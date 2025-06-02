# Ollama-Compatible API Server

基于 FastAPI 和 Hugging Face 的大模型部署服务，支持与 Open WebUI 集成。

## 功能特点

- 兼容 Ollama API 接口
- 支持多种 Hugging Face 模型
- 支持流式输出
- 支持聊天和生成两种模式
- 支持 GPU 加速
- 支持 Open WebUI 前端界面

## 环境要求

- Python 3.8+
- CUDA 支持（可选，用于 GPU 加速）
- 至少 4GB 显存（使用 GPU 时）

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd ollama
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

在 `main.py` 中配置可用模型：

```python
AVAILABLE_MODELS = {
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek:latest': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}
```

## 运行

启动服务器：
```bash
python main.py
```

服务器将在 `http://localhost:11434` 上运行。

## 与 Open WebUI 集成

1. 安装 Open WebUI：
```bash
docker run -d --name open-webui -p 3000:8080 -e OLLAMA_API_BASE_URL=http://host.docker.internal:11434/api ghcr.io/open-webui/open-webui:main
```

2. 访问 Open WebUI：
打开浏览器访问 `http://localhost:3000`

## API 接口

服务器提供以下 API 接口：

- `POST /api/generate` - 文本生成
- `POST /api/chat` - 聊天对话
- `GET /api/tags` - 获取可用模型列表
- `POST /api/show` - 获取模型详情

所有接口都兼容 Ollama API 规范。

## 模型支持

当前支持以下模型：
- DeepSeek-R1-Distill-Qwen-1.5B

可以根据需要添加更多模型。

## 性能优化

- 使用 float16 精度减少内存占用
- 支持 GPU 加速
- 流式输出减少响应延迟

## 注意事项

- 确保有足够的 GPU 显存（如果使用 GPU）
- 建议使用虚拟环境运行
- 生产环境部署时注意配置安全设置

## 许可证

MIT License 