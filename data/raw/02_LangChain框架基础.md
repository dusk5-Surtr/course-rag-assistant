# LangChain 框架基础

> 课程复习资料：梳理 LangChain 的定位、核心模块、LCEL 组合方式和常见工程问题。

## 1. LangChain 的定位

LangChain 是一个用于构建大语言模型应用的开源框架。它的目标不是训练模型，而是把模型、提示词、外部数据、检索器、工具和业务流程连接起来，降低多步骤应用开发与维护的复杂度。

只调用一次聊天 API 通常只能完成简单的单轮问答。真实应用往往还需要：

- 从本地文件或数据库加载上下文；
- 对长文本进行切分和向量化；
- 维护会话状态；
- 调用搜索、计算或业务 API；
- 记录中间步骤并进行调试；
- 在输出给用户前做格式解析和校验。

LangChain 通过模块化和组合式接口解决这些问题。它与具体模型解耦，可以在不同聊天模型、Embedding 模型、向量数据库和工具之间切换。

## 2. LangChain 的核心模块

LangChain 的常见核心模块包括：

1. **Model I/O（模型输入输出）**：负责提示词模板、聊天模型调用和输出解析。输入可以是字符串或消息列表，输出可解析为文本、JSON 或结构化对象。
2. **Retrieval（检索）**：负责文档加载、文本切分、Embedding、向量存储和相似度检索，让模型能够使用外部知识。
3. **Chains（链）**：把提示词、模型、解析器或其他处理步骤组合成可复用的处理流水线。
4. **Memory（记忆）**：在链执行过程中保存历史对话或业务状态，使应用可以多轮交互。实际项目中也可以把状态放在数据库中自行管理。
5. **Agents（智能体）**：让模型根据任务选择工具、规划步骤并循环执行，而不仅是生成一段文本。
6. **Callbacks（回调）**：记录链的中间步骤、耗时和异常，便于调试、监控和成本分析。

这些模块既可以单独使用，也可以通过 Runnable 接口组合。对于课程项目，至少应实际使用模型、提示词、检索器和输出解析器等组件，避免把 LangChain 当成单纯的 SDK 包装。

## 3. LangChain 生态

- **LangChain**：核心应用开发框架，负责任务编排和组件集成；
- **LangGraph**：面向有状态、多步骤和循环流程的 Agent 编排工具；
- **LangSmith**：用于追踪、调试和评估大模型应用；
- **LangChain 集成包**：把不同模型、向量库、文档加载器和工具接入统一接口，例如 OpenAI 兼容模型、Ollama、Chroma 等。

生态组件很多，但课程项目不应为了“组件齐全”而堆叠依赖。先确定用户任务，再选择最少的一组组件完成闭环。

## 4. LCEL 与链式组合

LCEL（LangChain Expression Language）使用管道符号把 Runnable 组合起来。一个典型的 RAG 链可以写成：

```text
检索器 -> 格式化上下文 -> 提示词模板 -> ChatModel -> 字符串解析器
```

每个步骤都接收上一步的输出，并返回下一步需要的输入。链式组合的好处是逻辑清晰、容易替换组件，也方便单独测试。

在实际实现中，可以先调用检索器获取 Document 列表，再把“上下文 + 问题”交给提示词模板和模型。这样既能得到答案，也能直接拿到来源文档用于界面展示。

## 5. 关键对象

- **Document**：包含 `page_content` 和 `metadata`，metadata 可保存文件名、页码、章节和片段编号；
- **TextSplitter**：把长文档切分为适合检索和模型上下文窗口的片段；
- **Embeddings**：把文本转换为向量，支持相似度计算；
- **VectorStore**：保存文本向量并提供相似度搜索；
- **Retriever**：面向检索场景的接口，输入问题，返回相关 Document；
- **PromptTemplate**：把变量填入提示词，控制模型角色、上下文和输出格式；
- **ChatModel**：接收消息并返回模型回复；
- **OutputParser**：把模型回复转换为字符串或结构化数据。

在 RAG 应用中，Document 的 metadata 很重要。没有来源信息，界面就无法告诉用户答案出自哪份资料，也难以评估检索是否正确。

## 6. 开发环境建议

课程项目建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 可执行 `.venv\Scripts\Activate.ps1`。应把虚拟环境目录加入 `.gitignore`，并把依赖版本固定到 `requirements.txt`。LangChain 生态拆分较细，常见包包括：

- `langchain-core`：Runnable、Prompt、Document 等基础抽象；
- `langchain-text-splitters`：文本切分；
- `langchain-openai`：OpenAI 兼容接口；
- `langchain-ollama`：Ollama 本地模型；
- `langchain-huggingface`：Hugging Face Embedding 集成。

安装前应确认包名和版本，不要照搬过时代码中的旧导入路径。框架升级后，部分类可能已迁移到独立的集成包。

## 7. 常见问题

**问题一：把模型调用写成一个大函数。**  
这会削弱 LangChain 的模块化优势。建议把配置、检索、提示词、链和界面分层。

**问题二：不保存来源元数据。**  
检索到文本却无法追溯文件或页码，用户无法核验答案。切分时就应补充稳定 citation。

**问题三：依赖版本不固定。**  
LangChain 组件更新较快，应固定依赖并记录验证过的 Python 版本。

**问题四：把密钥写进代码。**  
应使用 `.env` 和 `.env.example`，并在提交前扫描 Git 历史。

## 8. 本章要点

LangChain 的核心价值是把模型、数据和工具编排成可维护的应用。一个合格的课程项目应当让 LangChain 承担实际职责：使用 Retriever 完成检索，使用 PromptTemplate 约束输出，使用 ChatModel 接入在线或离线模型，并通过输出解析和来源元数据形成可测试、可展示的完整链路。
