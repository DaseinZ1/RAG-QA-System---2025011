import streamlit as st
import os
import sys
sys.path.insert(0, './src')

from knowledge_base import KnowledgeBase
from rag_chain import RAGChain

st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="📚",
    layout="wide"
)

def init_session_state():
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase(persist_directory="./chroma_db")
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0

def build_knowledge_base(uploaded_files):
    temp_dir = "./temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    documents = st.session_state.kb.load_documents(temp_dir)
    
    if documents:
        if st.session_state.kb.vector_store is None:
            st.session_state.kb.build_vector_store(documents)
        else:
            splits = st.session_state.kb.split_documents(documents)
            st.session_state.kb.vector_store.add_documents(splits)
            st.session_state.kb.vector_store.persist()
        
        st.session_state.document_count = st.session_state.kb.get_document_count()
        
        retriever = st.session_state.kb.vector_store.as_retriever(search_kwargs={"k": 3})
        st.session_state.rag_chain = RAGChain(retriever)
        
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        
        return True
    return False

def main():
    init_session_state()
    
    st.title("📚 RAG智能问答系统")
    st.sidebar.title("系统设置")
    
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "上传文档",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if st.button("构建知识库"):
            if uploaded_files:
                with st.spinner("正在构建知识库..."):
                    success = build_knowledge_base(uploaded_files)
                    if success:
                        st.success(f"知识库构建成功！共 {st.session_state.document_count} 个文本块")
                    else:
                        st.error("构建失败，请检查文件格式")
            else:
                st.warning("请先上传文档")
        
        if st.session_state.document_count > 0:
            st.info(f"当前知识库文本块数量: {st.session_state.document_count}")
        
        if st.button("清空对话历史"):
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.session_state.chat_history = []
            st.success("对话历史已清空")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.rag_chain is None:
                st.warning("请先上传文档并构建知识库")
                st.session_state.chat_history.append({"role": "assistant", "content": "请先上传文档并构建知识库"})
            else:
                with st.spinner("正在思考..."):
                    answer, sources = st.session_state.rag_chain.ask(prompt)
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    if sources:
                        with st.expander("查看参考来源"):
                            for i, source in enumerate(sources, 1):
                                doc_name = source.metadata.get('source', '未知文档')
                                st.write(f"{i}. {doc_name}")

if __name__ == "__main__":
    main()
