import sys
sys.path.insert(0, './src')

from knowledge_base import KnowledgeBase

print("=== 测试知识库构建 ===")

kb = KnowledgeBase(persist_directory="./chroma_db")

documents = kb.load_documents("./docs")
print(f"加载了 {len(documents)} 个文档")

if documents:
    kb.build_vector_store(documents)
    print(f"向量库中有 {kb.get_document_count()} 个文本块")
    
    results = kb.search("什么是自然语言处理", k=3)
    print(f"\n检索到 {len(results)} 个相关文本块")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.metadata.get('source', '未知文档')}")
        print(f"   {result.page_content[:100]}...")
else:
    print("没有找到文档")
