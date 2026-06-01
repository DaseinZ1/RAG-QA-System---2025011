# RAG智能问答系统

基于本地知识库的RAG（检索增强生成）智能问答系统，支持文档上传、知识库构建和智能问答功能。

## 功能特点

- 📄 支持多种文档格式：PDF、DOCX、TXT
- 📚 本地知识库构建：文档分块、向量化存储
- 🔍 智能检索：基于Chroma向量数据库的相似性检索
- 💬 对话式问答：支持多轮对话，记忆上下文
- 🌐 Web界面：使用Streamlit构建友好的交互界面
- ⚡ 内存优化：支持多种模型选择，适应不同硬件配置

## 环境要求

- Python 3.8+
- Ollama（用于本地大模型部署）
- 推荐内存：8GB+（4GB可用内存）
- 推荐模型：deepseek-r1:7b（8GB+内存）或 qwen2:0.5b（4GB内存）

## 安装步骤

### 1. 安装Ollama

访问 [Ollama官方网站](https://ollama.com/download) 下载并安装Ollama。

### 2. 下载大语言模型

根据您的系统内存选择合适的模型：

```bash
# 方案一：8GB+内存（推荐，效果更好）
ollama pull deepseek-r1:7b

# 方案二：4GB内存（内存受限时推荐）
ollama pull qwen2:0.5b

# 方案三：7GB内存（中等配置）
ollama pull qwen2:7b

# 下载嵌入模型（必需）
ollama pull nomic-embed-text
```

**模型对比**：

| 模型 | 内存需求 | 模型大小 | 推荐场景 |
|------|---------|---------|---------|
| deepseek-r1:7b | 8GB+ | 4.7GB | 效果最佳，适合高配置设备 |
| qwen2:7b | 7GB+ | 4.2GB | 平衡性能和资源占用 |
| qwen2:0.5b | 4GB+ | 352MB | 内存受限时的最佳选择 |

### 3. 安装项目依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用说明

### 方式一：命令行版本

```bash
python rag_cli.py
```

### 方式二：Web界面版本

```bash
streamlit run app.py
```

打开浏览器访问显示的URL（通常是 http://localhost:8501）。

### 使用流程

1. **上传文档**：在侧边栏选择PDF、DOCX或TXT文件上传
2. **构建知识库**：点击"构建知识库"按钮，系统会自动解析文档并构建向量库
3. **提问**：在底部输入框中输入问题，点击发送
4. **查看答案**：系统会基于知识库内容给出回答，并显示参考来源

### 内存优化建议

- **首次构建知识库**：建议关闭其他占用内存的程序
- **小内存设备**：使用 `qwen2:0.5b` 模型，已默认配置
- **大内存设备**：可修改 `app.py` 中的 `model_name` 参数为 `deepseek-r1:7b`
- **构建时间**：首次构建可能需要1-3分钟，请耐心等待

## 项目结构

```
RAG-QA-System/
├── app.py                 # Streamlit Web应用入口
├── rag_cli.py             # 命令行版本入口
├── test_ollama.py         # Ollama连接测试脚本
├── test_kb.py             # 知识库测试脚本
├── requirements.txt       # 项目依赖列表
├── .gitignore             # Git忽略配置
├── docs/                  # 示例文档目录
│   ├── nlp_introduction.txt
│   ├── bert_introduction.txt
│   ├── transformer_introduction.txt
│   ├── sentiment_analysis.txt
│   └── question_answering.txt
├── src/                   # 核心模块
│   ├── knowledge_base.py  # 知识库管理模块
│   └── rag_chain.py       # RAG问答链模块
└── chroma_db/             # Chroma向量数据库（运行后自动生成）
```

## 关键技术点

### RAG流程

1. **文档加载**：支持PDF、DOCX、TXT格式文档的读取
2. **文本分块**：使用RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化**：使用Ollama的nomic-embed-text或SentenceTransformer的all-MiniLM-L6-v2
4. **向量存储**：使用Chroma向量数据库
5. **相似性检索**：基于余弦相似度的Top-K检索（K=3）
6. **答案生成**：基于Ollama大模型，结合检索结果生成答案

### 系统提示词设计

系统提示词要求模型：
- 仅使用参考文档中的信息回答问题
- 如果文档中没有相关信息，明确回答"文档中未找到相关答案"
- 不编造信息，不猜测

### 模型切换

如需切换模型，修改 `app.py` 中的以下参数：

```python
# 使用大模型（需要8GB+内存）
st.session_state.rag_chain = RAGChain(retriever, model_name="deepseek-r1:7b")

# 使用小模型（4GB内存即可）
st.session_state.rag_chain = RAGChain(retriever, model_name="qwen2:0.5b")
```

## 测试示例

### 相关问题

1. **问**：什么是自然语言处理？
   **答**：自然语言处理(NLP)是人工智能的一个分支，致力于使计算机能够理解、解释和生成人类语言。

2. **问**：BERT模型的核心创新是什么？
   **答**：BERT的核心创新包括：1)双向预训练，能够同时考虑左右上下文；2)Masked Language Model(MLM)，随机掩盖部分token并预测；3)Next Sentence Prediction(NSP)，预测两个句子是否连续。

3. **问**：Transformer架构的组成部分有哪些？
   **答**：Transformer的核心组成包括：1)多头自注意力机制；2)位置编码；3)前馈神经网络；4)残差连接和层归一化。

4. **问**：情感分析有哪些类型？
   **答**：情感分析的类型包括：1)文档级情感分析；2)句子级情感分析；3)方面级情感分析。

5. **问**：RAG的流程是什么？
   **答**：RAG的流程包括：1)文档预处理（加载、分块、向量化）；2)向量存储（存入向量数据库）；3)查询处理（向量化查询）；4)相似性检索（查找相关文档）；5)答案生成（基于检索结果生成答案）。

### 无关问题

1. **问**：今天天气怎么样？
   **答**：文档中未找到相关答案

2. **问**：如何制作蛋糕？
   **答**：文档中未找到相关答案

## 打包部署

使用pyinstaller将应用打包成exe文件：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed app.py
```

## 已知问题与改进方向

### 已知问题

- 首次加载模型时可能需要较长时间
- 大文档处理时间较长
- 低内存设备可能无法加载大模型

### 改进方向

- 支持更多文档格式（如Markdown、HTML）
- 添加夜间模式
- 支持批量文档上传
- 添加问答记录导出功能
- 支持多语言文档
- 添加模型自动选择功能（根据可用内存）

## 许可证

MIT License