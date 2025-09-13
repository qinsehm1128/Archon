# 🚀 Archon 部署指南

## 本地Qwen嵌入模型版本

这个版本使用 **Qwen3-Embedding-0.6B** 本地嵌入模型，完全替代OpenAI API，实现：
- ✅ **零API费用**
- ✅ **完全离线运行**
- ✅ **数据隐私保护**
- ✅ **更快的处理速度**

## 🎯 快速开始

### **方法1：自动部署（推荐）**

**Linux/macOS:**
```bash
git clone https://github.com/your-username/Archon.git
cd Archon
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```cmd
git clone https://github.com/your-username/Archon.git
cd Archon
deploy.bat
```

### **方法2：手动部署**

#### **1. 克隆项目**
```bash
git clone https://github.com/your-username/Archon.git
cd Archon
```

#### **2. 配置环境变量**
创建 `.env` 文件：
```env
# Supabase配置
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here

# 本地嵌入模型配置
EMBEDDING_DIMENSIONS=1024
USE_CONTEXTUAL_EMBEDDINGS=false

# 端口配置
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_UI_PORT=3737

# 日志
LOG_LEVEL=INFO
```

#### **3. 创建模型缓存目录**
```bash
mkdir models
```

#### **4. 启动服务**
```bash
# 完整服务
docker compose --profile full up -d --build

# 仅后端+MCP (用于开发)
docker compose --profile backend up -d --build
```

#### **5. 初始化数据库**
1. 访问你的Supabase控制台
2. 进入 SQL 编辑器
3. 运行 `migration/complete_setup.sql` 中的脚本

## 🌐 访问地址

启动成功后，访问：

- **前端界面**: http://localhost:3737
- **API服务**: http://localhost:8181/health
- **MCP服务**: http://localhost:8051/health

## 📂 目录结构

```
Archon/
├── models/                 # Qwen模型缓存 (首次运行后自动创建)
├── archon-ui-main/        # React前端
├── python/                # Python后端
│   ├── src/server/        # 主API服务
│   ├── src/mcp_server/    # MCP服务
│   └── migration/         # 数据库迁移脚本
├── .env                   # 环境变量配置
├── docker-compose.yml     # Docker配置
├── deploy.sh              # Linux/macOS部署脚本
└── deploy.bat             # Windows部署脚本
```

## 🔧 配置说明

### **必需配置**
```env
SUPABASE_URL=            # Supabase项目URL
SUPABASE_SERVICE_KEY=    # Supabase Service Role密钥 (不是Anon密钥!)
```

### **嵌入模型配置**
```env
EMBEDDING_DIMENSIONS=1024              # Qwen模型输出维度
USE_CONTEXTUAL_EMBEDDINGS=false       # 禁用上下文增强（无需LLM）
```

### **服务端口**
```env
ARCHON_SERVER_PORT=8181   # 主API服务端口
ARCHON_MCP_PORT=8051      # MCP服务端口
ARCHON_UI_PORT=3737       # 前端端口
```

## 🤖 模型信息

- **模型**: Qwen/Qwen3-Embedding-0.6B
- **参数量**: 600M
- **输出维度**: 1024
- **大小**: ~1.2GB
- **首次下载**: 自动进行，存储在 `./models/` 目录
- **设备**: 自动检测 GPU/CPU

## 📋 使用步骤

1. **访问前端**: http://localhost:3737
2. **添加知识源**: 输入网页URL开始爬取
3. **模型下载**: 首次爬取时自动下载Qwen模型
4. **向量化**: 文档自动使用Qwen模型生成1024维向量
5. **智能搜索**: 通过MCP工具或前端界面搜索知识

## 🛠️ MCP工具使用

在支持MCP的IDE中（如Claude Code、Cursor）：

```bash
# RAG查询
archon:perform_rag_query(query="FastAPI异步编程", match_count=5)

# 代码示例搜索
archon:search_code_examples(query="React hooks useEffect", match_count=3)

# 查看知识源
archon:get_available_sources()
```

## 🚨 故障排除

### **Docker错误**
```bash
# 强制重建
docker compose down --rmi all
docker compose --profile full up -d --build --force-recreate
```

### **模型下载失败**
```bash
# 手动下载测试
docker exec -it archon-server python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', cache_dir='/app/models')
print('Model downloaded!')
"
```

### **数据库连接错误**
1. 检查Supabase URL和密钥是否正确
2. 确保使用 **Service Role** 密钥，不是 Anon 密钥
3. 检查Supabase项目是否激活

### **向量维度错误**
```sql
-- 修改数据库表结构为1024维
ALTER TABLE archon_crawled_pages DROP COLUMN embedding;
ALTER TABLE archon_crawled_pages ADD COLUMN embedding vector(1024);

ALTER TABLE archon_code_examples DROP COLUMN embedding;
ALTER TABLE archon_code_examples ADD COLUMN embedding vector(1024);
```

## 📊 性能对比

| 特性 | OpenAI API | Qwen 本地模型 |
|------|------------|---------------|
| **成本** | $0.02/1M tokens | 免费 |
| **速度** | 依赖网络 | 本地处理，更快 |
| **隐私** | 数据发送外部 | 完全本地 |
| **离线** | 需要网络 | 完全离线 |
| **维度** | 1536 | 1024 |

## 💡 生产环境

### **资源要求**
- **内存**: 至少 4GB (推荐 8GB+)
- **存储**: 至少 5GB (模型1.2GB + 数据)
- **CPU**: 支持 AVX2 指令集
- **GPU**: 可选，支持 CUDA 加速

### **环境变量优化**
```env
# 生产环境设置
LOG_LEVEL=WARNING
PROD=true
```

## 🔄 升级指南

1. **拉取最新代码**: `git pull origin main`
2. **重新构建**: `docker compose --profile full up -d --build`
3. **数据库迁移**: 运行新的迁移脚本（如有）

## 📞 支持

如遇问题，请检查：
1. [故障排除](#🚨-故障排除) 部分
2. Docker日志: `docker compose logs -f`
3. 提交 GitHub Issue

---

🎉 **恭喜！** 你现在拥有了一个完全本地化的AI知识管理系统！