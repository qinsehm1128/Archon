#!/usr/bin/env python3
"""
独立的嵌入模型测试脚本
跳过所有数据库和配置检查，直接测试 Qwen3-Embedding-0.6B 模型
"""

import asyncio
import sys
import os
import traceback
from pathlib import Path

async def test_direct_model():
    """直接测试模型，不依赖项目代码"""
    print("🤖 Direct model test starting...")

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        cache_dir = "./models"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"📥 Loading {model_name}")
        print(f"💾 Cache: {cache_dir}")
        print(f"🖥️  Device: {device}")

        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        ).to(device)

        model.eval()

        print("✅ Model loaded successfully!")

        # 测试嵌入生成
        test_texts = ["Hello world", "This is a test", "Machine learning"]

        with torch.no_grad():
            inputs = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)

            # Mean pooling
            token_embeddings = outputs[0]
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        print(f"✅ Generated embeddings!")
        print(f"   - Input texts: {len(test_texts)}")
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Sample embedding: {embeddings[0][:5].tolist()}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 Qwen Embedding Standalone Test")
    print("=" * 50)

    os.environ['HF_HOME'] = './models'

    success = asyncio.run(test_direct_model())

    if success:
        print("\n🎉 Test successful! Qwen model works!")
    else:
        print("\n❌ Test failed!")

    exit(0 if success else 1)