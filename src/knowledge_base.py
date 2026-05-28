import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

class KnowledgeBase:
    def __init__(self, persist_directory="./chroma_db", embedding_model="nomic-embed-text"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = None
    
    def load_documents(self, folder_path):
        documents = []
        
        if not os.path.exists(folder_path):
            print(f"目录不存在: {folder_path}")
            return documents
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                elif filename.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                else:
                    print(f"不支持的文件格式: {filename}")
            except Exception as e:
                print(f"加载文件失败 {filename}: {e}")
        
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(documents)
        return splits
    
    def build_vector_store(self, documents):
        splits = self.split_documents(documents)
        
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.vector_store.persist()
        print(f"向量库构建完成，共 {len(splits)} 个文本块")
    
    def load_vector_store(self):
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
            except Exception as e:
                print(f"加载向量库失败: {e}")
                return False
        return False
    
    def search(self, query, k=3):
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_document_count(self):
        if self.vector_store is not None:
            return self.vector_store._collection.count()
        return 0
