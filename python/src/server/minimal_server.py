"""
最小化服务器 - 仅用于测试Qwen嵌入模型
跳过所有Supabase连接
"""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# 添加路径
sys.path.append('/app/src')

app = FastAPI(title="Minimal Qwen Test Server")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Minimal server running"}

@app.post("/test-embedding")
async def test_embedding(text: str = "Hello world"):
    """测试Qwen嵌入模型"""
    try:
        from server.services.embeddings.embedding_service import create_embedding

        embedding = await create_embedding(text)

        return {
            "success": True,
            "text": text,
            "embedding_dimensions": len(embedding),
            "sample_values": embedding[:5]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create embedding"
        }

@app.get("/")
async def root():
    return {
        "message": "Minimal Qwen Test Server",
        "endpoints": {
            "/health": "Health check",
            "/test-embedding": "Test Qwen embedding model",
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8181)