import os
import sys
sys.path.insert(0, './src')

from knowledge_base import KnowledgeBase
from rag_chain import RAGChain

def main():
    print("=== RAG智能问答系统 (命令行版本) ===")
    
    kb = KnowledgeBase(persist_directory="./chroma_db")
    
    docs_folder = "./docs"
    if os.path.exists(docs_folder) and len(os.listdir(docs_folder)) > 0:
        print(f"正在从 {docs_folder} 加载文档...")
        documents = kb.load_documents(docs_folder)
        
        if documents:
            kb.build_vector_store(documents)
        else:
            print("未找到可加载的文档")
            if not kb.load_vector_store():
                print("错误：无法加载向量库且没有新文档")
                return
    else:
        if not kb.load_vector_store():
            print("错误：文档目录不存在且向量库也不存在")
            return
    
    retriever = kb.vector_store.as_retriever(search_kwargs={"k": 3})
    
    try:
        rag_chain = RAGChain(retriever)
    except Exception as e:
        print(f"初始化RAG链失败: {e}")
        print("请确保Ollama已安装并下载了模型")
        print("安装命令: ollama pull deepseek-r1:7b")
        return
    
    print("\n知识库构建完成！可以开始提问了。")
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        question = input("请输入问题: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("感谢使用RAG问答系统，再见！")
            break
        
        if not question.strip():
            print("请输入有效的问题")
            continue
        
        print("正在思考...")
        answer, sources = rag_chain.ask(question)
        
        print(f"\n答案: {answer}\n")
        
        if sources:
            print("参考来源:")
            for i, source in enumerate(sources, 1):
                doc_name = source.metadata.get('source', '未知文档')
                print(f"  {i}. {doc_name}")
        print("-" * 50)

if __name__ == "__main__":
    main()
