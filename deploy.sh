#!/bin/bash
# Archon 一键部署脚本

set -e

echo "🚀 开始部署 Archon (本地Qwen嵌入版本)"
echo "=================================="

# 检查必要文件
echo "📋 检查环境..."
if [ ! -f ".env" ]; then
    echo "❌ 错误：缺少 .env 文件"
    echo "请创建 .env 文件并配置以下变量："
    echo "  SUPABASE_URL=https://your-project.supabase.co"
    echo "  SUPABASE_SERVICE_KEY=your-service-key"
    echo "  EMBEDDING_DIMENSIONS=1024"
    echo "  USE_CONTEXTUAL_EMBEDDINGS=false"
    exit 1
fi

# 创建模型目录
echo "📁 创建模型缓存目录..."
mkdir -p models

# 检查Docker
echo "🐳 检查Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ 错误：Docker 未安装"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误：docker-compose 未安装"
    exit 1
fi

# 构建并启动服务
echo "🔨 构建并启动服务..."
docker compose --profile full up -d --build

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker compose ps

echo ""
echo "🎉 部署完成！"
echo ""
echo "📍 访问地址："
echo "  前端: http://localhost:3737"
echo "  API:  http://localhost:8181/health"
echo "  MCP:  http://localhost:8051/health"
echo ""
echo "📋 下一步："
echo "  1. 访问前端界面"
echo "  2. 在Supabase中运行 migration/complete_setup.sql"
echo "  3. 开始爬取网页测试Qwen模型"
echo ""
echo "📝 注意："
echo "  - 首次爬取时会自动下载Qwen模型(1.2GB)"
echo "  - 模型缓存在 ./models 目录"
echo "  - 完全离线运行，无需API密钥"