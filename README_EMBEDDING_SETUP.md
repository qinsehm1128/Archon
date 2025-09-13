# 本地嵌入模型设置说明

## 概述
Archon 现在使用 **Qwen3-Embedding-0.6B** 本地嵌入模型，完全替代了之前的 OpenAI API 调用。这意味着：
- ✅ **零 API 成本**
- ✅ **完全离线运行**
- ✅ **数据隐私保护**
- ✅ **更快的处理速度**（本地GPU）

## Docker 运行方式（推荐）

### 1. 首次运行
```bash
# 创建模型目录（用于持久化存储模型）
mkdir models

# 使用 Docker Compose 启动所有服务
docker compose --profile full up -d --build
```

### 2. 模型下载
- 模型会在**第一次爬取网页时自动下载**（约1.2GB）
- 下载位置：`./models/` 目录
- 下载完成后，模型会被持久化保存，重启容器无需重新下载

### 3. 目录结构
```
Archon/
├── models/                  # 模型缓存目录（自动创建）
│   └── hub/                # Hugging Face 模型缓存
│       └── models--Qwen/  # Qwen 模型文件
├── docker-compose.yml      # Docker 配置
└── python/
    └── src/server/services/embeddings/
        └── embedding_service.py  # 嵌入服务实现
```

## 技术细节

### 模型规格
- **模型名称**: Qwen/Qwen3-Embedding-0.6B
- **参数量**: 0.6B（600M参数）
- **输出维度**: 可配置（默认与OpenAI兼容）
- **最大输入长度**: 512 tokens
- **设备支持**: CPU/CUDA 自动检测

### Docker 配置
```yaml
# docker-compose.yml 关键配置
volumes:
  - ./models:/app/models  # 持久化模型存储
environment:
  - HF_HOME=/app/models   # Hugging Face 缓存目录
  - TRANSFORMERS_CACHE=/app/models
```

### 性能优化
- **批处理**: 32个文本一批，优化内存使用
- **CPU优化**: 使用 PyTorch CPU 版本，减小镜像体积
- **懒加载**: 模型仅在需要时加载
- **单例模式**: 全局共享一个模型实例

## 常见问题

### Q: 第一次运行很慢？
A: 第一次需要下载1.2GB的模型文件，请耐心等待。下载完成后会自动缓存。

### Q: 如何确认模型已下载？
A: 检查 `./models/hub/` 目录，应该有 `models--Qwen` 文件夹。

### Q: 可以使用GPU吗？
A: 默认使用CPU版本以确保兼容性。如需GPU支持，需要修改 Dockerfile 安装 CUDA 版本的 PyTorch。

### Q: 如何清理模型缓存？
A: 删除 `./models` 目录即可，下次运行会重新下载。

### Q: 内存要求？
A: 建议至少 4GB 可用内存，模型加载约占用 2GB。

## 迁移说明

如果你之前使用的是 API 版本：
1. 不再需要 `OPENAI_API_KEY`
2. 所有嵌入生成现在都在本地完成
3. 接口保持兼容，无需修改调用代码

## 故障排除

### 模型下载失败
```bash
# 手动下载模型
docker exec -it archon-server python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', cache_dir='/app/models')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', cache_dir='/app/models')
print('Model downloaded successfully!')
"
```

### 查看日志
```bash
docker compose logs archon-server -f
```

### 重新构建
```bash
docker compose down
docker compose --profile full up -d --build --force-recreate
```

## 优势对比

| 特性 | OpenAI API | Qwen 本地模型 |
|------|------------|---------------|
| 成本 | $0.02/1M tokens | 免费 |
| 速度 | 依赖网络 | 本地处理，更快 |
| 隐私 | 数据发送到 OpenAI | 完全本地 |
| 离线 | 需要网络 | 完全离线 |
| 模型大小 | - | 1.2GB |

## 支持

如有问题，请查看服务日志或提交 Issue。