# 第七章作业提交材料

## 1. 平台文本框可直接填写

```text
作业：第七章 智能应用开发

项目名称：智学助手 · Course Knowledge RAG
项目简介：基于 LangChain 的课程知识库问答应用，支持 PDF/Markdown/TXT 资料导入、向量检索、来源引用、资料不足拒答，以及在线大模型、Ollama 离线模型和无 Key 本地检索演示三种模式。

技术路线：LangChain + InMemoryVectorStore + Retriever + ChatPromptTemplate + ChatModel + Streamlit；默认使用零下载 Hashing Embedding 保证可复现，可选择 HuggingFace 中文 Embedding。

核心流程：课程资料加载 → 文本切分 → Embedding → 向量检索 → 提示词约束 → 大模型生成 → 答案与来源展示。

开源仓库：https://github.com/dusk5-Surtr/course-rag-assistant
README：见附件 README.md（同时附 README.pdf 便于预览）

自测情况：
- 12 个自动化测试通过（含 Streamlit 页面冒烟、检索问答、拒答、在线/离线协议 Mock 调用）；
- 8/8 固定问题集通过（本地检索演示模式，100%）；
- 真实浏览器页面已截图并人工检查；
- 仓库无真实 API Key，已配置 .env.example 与 .gitignore。

说明：本机没有配置真实在线 API Key，也未安装 Ollama，因此真实模型回答质量需要按 README 配置后验证；在线与离线 LangChain 调用路径已通过本地协议 Mock 服务完成集成测试。
```

## 2. 附加文件

上传以下文件：

1. `README.md`：仓库中的同一份 README；
2. `README.pdf`：README 的可预览版本；
3. 可选：`docs/images/app-home.png` 和 `docs/images/app-answer.png`，作为运行截图。

平台要求只上传 README 时，以 `README.pdf` 为主附件，并在文本提交中附上公开仓库链接。

## 3. 提交前最后检查

- [ ] GitHub 仓库已创建并设为 Public；
- [ ] 仓库首页可以看到 README、源代码、示例资料、测试和 LICENSE；
- [ ] 仓库链接在退出 GitHub 登录后仍可打开；
- [ ] 仓库中没有 `.env`、真实 API Key 或隐私资料；
- [ ] 在干净目录按 README 启动成功；
- [ ] 文本提交中的仓库链接与 README 中的链接一致；
- [ ] 上传 README.md 和 README.pdf；
- [ ] 点击提交后确认平台显示提交成功。

## 4. 如果真实模型暂未配置

默认 `LLM_PROVIDER=demo` 可以完整演示检索、来源和拒答。若需要展示真实大模型回答，按 README 第 6.4 节或第 6.5 节配置：

- 在线：DeepSeek/Qwen 等兼容 OpenAI 接口的服务；
- 离线：安装 Ollama 后拉取本地模型。

切换后重启 `streamlit run app.py`，界面的运行模式会变为“在线大模型”或“离线大模型”。
