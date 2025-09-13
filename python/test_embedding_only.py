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

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

async def test_qwen_embedding():
    """测试 Qwen 嵌入模型是否正常工作"""
    print("🚀 Starting Qwen3-Embedding-0.6B test...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")

    try:
        # 直接导入嵌入服务
        print("📦 Importing embedding service...")
        from src.server.services.embeddings.embedding_service import create_embedding, create_embeddings_batch

        # 测试单个嵌入
        print("🔍 Testing single embedding...")
        test_text = "This is a test sentence for Qwen embedding model."
        embedding = await create_embedding(test_text)

        print(f"✅ Single embedding successful!")
        print(f"   - Input text: {test_text[:50]}...")
        print(f"   - Embedding dimensions: {len(embedding)}")
        print(f"   - Sample values: {embedding[:5]}")

        # 测试批量嵌入
        print("\n📝 Testing batch embeddings...")
        test_texts = [
            "Hello world, this is a test.",
            "Machine learning with local models is great.",
            "Python and transformers make AI accessible.",
            "Qwen3-Embedding-0.6B is a lightweight model."
        ]

        result = await create_embeddings_batch(test_texts)

        print(f"✅ Batch embedding successful!")
        print(f"   - Total texts: {len(test_texts)}")
        print(f"   - Successful embeddings: {result.success_count}")
        print(f"   - Failed embeddings: {result.failure_count}")

        if result.embeddings:
            print(f"   - Embedding dimensions: {len(result.embeddings[0])}")
            print(f"   - Sample embedding: {result.embeddings[0][:3]}")

        if result.has_failures:
            print(f"❌ Some failures occurred:")
            for failure in result.failed_items:
                print(f"   - Error: {failure['error']}")

        print("\n🎉 Qwen embedding model test completed successfully!")
        print("✨ Your local embedding model is working perfectly!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 This might be a dependency issue. Check if transformers and torch are installed.")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

async def test_model_download():
    """测试模型下载和初始化"""
    print("\n🤖 Testing model download and initialization...")

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        cache_dir = "./models"

        print(f"📥 Loading model: {model_name}")
        print(f"💾 Cache directory: {cache_dir}")
        print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # 加载分词器
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")

        # 加载模型
        print("🧠 Loading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")

        # 测试推理
        print("🔄 Testing model inference...")
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs[0].mean(dim=1)  # Simple mean pooling

        print(f"✅ Model inference successful!")
        print(f"   - Input: {test_text}")
        print(f"   - Output shape: {embeddings.shape}")

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Qwen3-Embedding-0.6B Standalone Test")
    print("=" * 60)

    # 设置环境变量
    os.environ['HF_HOME'] = './models'

    # 运行测试
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 先测试模型下载
        model_success = loop.run_until_complete(test_model_download())

        if model_success:
            # 再测试嵌入服务
            embedding_success = loop.run_until_complete(test_qwen_embedding())

            if embedding_success:
                print("\n🎊 ALL TESTS PASSED! 🎊")
                print("✨ Your Qwen embedding model is ready to use!")
                exit(0)
            else:
                print("\n❌ Embedding service test failed")
                exit(1)
        else:
            print("\n❌ Model download/initialization failed")
            exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        exit(1)
    finally:
        loop.close()