from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self, retriever, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        self.retriever = retriever
        self.llm = None
        self.chain = None
        self.memory = None
        self._init_chain()
    
    def _init_chain(self):
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.1,
            num_ctx=8192
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        system_prompt = """
        You are a question-answering assistant based on a knowledge base. Please answer the user's questions according to the provided reference documents.
        
        Rules:
        1. Read the reference documents carefully and only use the information in the documents to answer questions
        2. If there is no relevant information in the documents, clearly answer "文档中未找到相关答案"
        3. Do not make up answers or guess
        4. Your answer should be concise and accurate
        5. If the user asks in Chinese, answer in Chinese
        
        Reference documents:
        {context}
        
        Question: {question}
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_prompt
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
    
    def ask(self, question):
        try:
            result = self.chain({"question": question})
            answer = result.get("answer", "")
            sources = result.get("source_documents", [])
            
            if "未找到相关答案" not in answer and not answer.strip():
                answer = "文档中未找到相关答案"
            
            return answer, sources
        except Exception as e:
            print(f"问答失败: {e}")
            return "问答过程中发生错误", []
    
    def clear_memory(self):
        if self.memory:
            self.memory.clear()
