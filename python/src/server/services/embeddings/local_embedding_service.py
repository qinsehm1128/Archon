"""
Local Embedding Service using Qwen3-Embedding-0.6B

This service provides local text embedding generation using the Qwen3-Embedding-0.6B model
from Hugging Face, eliminating the need for API calls.
"""

import asyncio
import torch
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer
import numpy as np

from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)


@dataclass
class LocalEmbeddingBatchResult:
    """Result of local batch embedding creation."""

    embeddings: list[list[float]] = field(default_factory=list)
    failed_items: list[dict] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    texts_processed: list[str] = field(default_factory=list)

    def add_success(self, embedding: list[float], text: str):
        """Add a successful embedding."""
        self.embeddings.append(embedding)
        self.texts_processed.append(text)
        self.success_count += 1

    def add_failure(self, text: str, error: Exception, batch_index: int | None = None):
        """Add a failed item with error details."""
        self.failed_items.append({
            "text": text[:200] if text else None,
            "error": str(error),
            "error_type": type(error).__name__,
            "batch_index": batch_index,
        })
        self.failure_count += 1

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    @property
    def total_requested(self) -> int:
        return self.success_count + self.failure_count


class LocalEmbeddingService:
    """
    Service for generating embeddings locally using Qwen3-Embedding-0.6B model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the local embedding service.

        Args:
            model_name: Hugging Face model name or local path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or "./models"

        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        logger.info(f"LocalEmbeddingService initialized with model: {model_name}, device: {self.device}")

    async def initialize(self):
        """
        Lazy initialization of the model and tokenizer.
        Downloads the model on first use.
        """
        async with self._init_lock:
            if self._initialized:
                return

            try:
                logger.info(f"Loading embedding model: {self.model_name}")

                # Run model loading in executor to avoid blocking
                loop = asyncio.get_event_loop()

                def load_model():
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )

                    model = AutoModel.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    ).to(self.device)

                    # Set model to evaluation mode
                    model.eval()

                    return tokenizer, model

                self.tokenizer, self.model = await loop.run_in_executor(None, load_model)
                self._initialized = True

                logger.info(f"Successfully loaded model: {self.model_name}")

            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise RuntimeError(f"Failed to initialize embedding model: {e}")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to get sentence embeddings.
        """
        token_embeddings = model_output[0]  # First element of model output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts synchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Qwen3-Embedding supports up to 512 tokens
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        model_output = self.model(**encoded_input)

        # Apply mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists
        return embeddings.cpu().numpy().tolist()

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        # Ensure model is initialized
        await self.initialize()

        try:
            # Run embedding generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._generate_embeddings_batch,
                [text]
            )
            return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise RuntimeError(f"Failed to create embedding: {e}")

    async def create_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Optional[callable] = None,
    ) -> LocalEmbeddingBatchResult:
        """
        Create embeddings for multiple texts with batch processing.

        Args:
            texts: List of texts to create embeddings for
            batch_size: Number of texts to process at once
            progress_callback: Optional callback for progress reporting

        Returns:
            LocalEmbeddingBatchResult with successful embeddings and failure details
        """
        if not texts:
            return LocalEmbeddingBatchResult()

        # Ensure model is initialized
        await self.initialize()

        result = LocalEmbeddingBatchResult()

        with safe_span(
            "create_local_embeddings_batch",
            text_count=len(texts),
            batch_size=batch_size,
            device=self.device
        ) as span:
            try:
                # Process texts in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_index = i // batch_size

                    try:
                        # Generate embeddings for this batch
                        loop = asyncio.get_event_loop()
                        batch_embeddings = await loop.run_in_executor(
                            None,
                            self._generate_embeddings_batch,
                            batch
                        )

                        # Add successful embeddings
                        for text, embedding in zip(batch, batch_embeddings):
                            result.add_success(embedding, text)

                    except Exception as e:
                        logger.error(f"Batch {batch_index} failed: {e}")
                        # Track failures for this batch
                        for text in batch:
                            result.add_failure(text, e, batch_index)

                    # Progress reporting
                    if progress_callback:
                        processed = result.success_count + result.failure_count
                        progress = (processed / len(texts)) * 100

                        message = f"Processed {processed}/{len(texts)} texts locally"
                        if result.has_failures:
                            message += f" ({result.failure_count} failed)"

                        await progress_callback(message, progress)

                    # Yield control between batches
                    await asyncio.sleep(0.01)

                span.set_attribute("embeddings_created", result.success_count)
                span.set_attribute("embeddings_failed", result.failure_count)
                span.set_attribute("success", not result.has_failures)

                logger.info(
                    f"Local embedding batch completed: {result.success_count} successful, "
                    f"{result.failure_count} failed"
                )

                return result

            except Exception as e:
                span.set_attribute("catastrophic_failure", True)
                logger.error(f"Catastrophic failure in local batch embedding: {e}")

                # Mark remaining texts as failed
                processed_count = result.success_count + result.failure_count
                for text in texts[processed_count:]:
                    result.add_failure(text, e)

                return result


# Global singleton instance
_local_embedding_service: Optional[LocalEmbeddingService] = None


def get_local_embedding_service() -> LocalEmbeddingService:
    """
    Get or create the global local embedding service instance.
    """
    global _local_embedding_service
    if _local_embedding_service is None:
        _local_embedding_service = LocalEmbeddingService()
    return _local_embedding_service