"""
Embedding Service using Qwen3-Embedding-0.6B

This service exclusively uses the local Qwen3-Embedding-0.6B model for text embeddings,
eliminating all API dependencies.
"""

import asyncio
import torch
from typing import List, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer

from ...config.logfire_config import safe_span, search_logger

logger = search_logger


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding creation with success/failure tracking."""

    embeddings: list[list[float]] = field(default_factory=list)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
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
        error_dict = {
            "text": text[:200] if text else None,
            "error": str(error),
            "error_type": type(error).__name__,
            "batch_index": batch_index,
        }
        self.failed_items.append(error_dict)
        self.failure_count += 1

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    @property
    def total_requested(self) -> int:
        return self.success_count + self.failure_count


class QwenEmbeddingService:
    """Singleton service for Qwen3-Embedding-0.6B model."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model_name = "Qwen/Qwen3-Embedding-0.6B"
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.cache_dir = "./models"
            self.model = None
            self.tokenizer = None
            self._init_lock = asyncio.Lock()
            QwenEmbeddingService._initialized = True
            logger.info(f"QwenEmbeddingService initialized with device: {self.device}")

    async def initialize(self):
        """Lazy load the model on first use."""
        async with self._init_lock:
            if self.model is not None:
                return

            try:
                logger.info(f"Loading Qwen embedding model: {self.model_name}")

                loop = asyncio.get_event_loop()

                def load_model():
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

                    model.eval()
                    return tokenizer, model

                self.tokenizer, self.model = await loop.run_in_executor(None, load_model)
                logger.info(f"Successfully loaded Qwen model on {self.device}")

            except Exception as e:
                logger.error(f"Failed to initialize Qwen model: {e}")
                raise RuntimeError(f"Failed to initialize Qwen model: {e}")

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def _generate_embeddings_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings synchronously."""
        # Tokenize
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Generate embeddings
        model_output = self.model(**encoded_input)

        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()


# Global instance
_qwen_service = QwenEmbeddingService()


async def create_embedding(text: str, provider: str | None = None) -> list[float]:
    """
    Create an embedding for a single text using Qwen model.

    Args:
        text: Text to create an embedding for
        provider: Ignored, kept for compatibility

    Returns:
        List of floats representing the embedding
    """
    await _qwen_service.initialize()

    try:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            _qwen_service._generate_embeddings_batch_sync,
            [text]
        )
        return embeddings[0]
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise RuntimeError(f"Failed to create embedding: {e}")


async def create_embeddings_batch(
    texts: list[str],
    progress_callback: Any | None = None,
    provider: str | None = None,
) -> EmbeddingBatchResult:
    """
    Create embeddings for multiple texts using Qwen model.

    Args:
        texts: List of texts to create embeddings for
        progress_callback: Optional callback for progress reporting
        provider: Ignored, kept for compatibility

    Returns:
        EmbeddingBatchResult with successful embeddings and failure details
    """
    if not texts:
        return EmbeddingBatchResult()

    await _qwen_service.initialize()

    result = EmbeddingBatchResult()

    # Validate texts
    validated_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            try:
                validated_texts.append(str(text))
            except Exception:
                validated_texts.append("")
        else:
            validated_texts.append(text)

    texts = validated_texts

    with safe_span(
        "create_embeddings_batch",
        text_count=len(texts),
        device=_qwen_service.device
    ) as span:
        try:
            # Process in batches of 32 for memory efficiency
            batch_size = 32

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_index = i // batch_size

                try:
                    # Generate embeddings
                    loop = asyncio.get_event_loop()
                    batch_embeddings = await loop.run_in_executor(
                        None,
                        _qwen_service._generate_embeddings_batch_sync,
                        batch
                    )

                    # Add successful embeddings
                    for text, embedding in zip(batch, batch_embeddings):
                        result.add_success(embedding, text)

                except Exception as e:
                    logger.error(f"Batch {batch_index} failed: {e}")
                    for text in batch:
                        result.add_failure(text, e, batch_index)

                # Progress reporting
                if progress_callback:
                    processed = result.success_count + result.failure_count
                    progress = (processed / len(texts)) * 100

                    message = f"Processed {processed}/{len(texts)} texts"
                    if result.has_failures:
                        message += f" ({result.failure_count} failed)"

                    await progress_callback(message, progress)

                # Yield control
                await asyncio.sleep(0.01)

            span.set_attribute("embeddings_created", result.success_count)
            span.set_attribute("embeddings_failed", result.failure_count)
            span.set_attribute("success", not result.has_failures)

            return result

        except Exception as e:
            span.set_attribute("catastrophic_failure", True)
            logger.error(f"Catastrophic failure in batch embedding: {e}")

            # Mark remaining texts as failed
            processed_count = result.success_count + result.failure_count
            for text in texts[processed_count:]:
                result.add_failure(text, RuntimeError(f"Catastrophic failure: {str(e)}"))

            return result


# Deprecated function kept for compatibility
async def get_openai_api_key() -> str | None:
    """DEPRECATED: No longer needed with local model."""
    return None