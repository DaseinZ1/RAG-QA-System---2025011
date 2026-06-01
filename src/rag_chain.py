from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    def __init__(self, retriever, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        self.retriever = retriever
        self.llm = None
        self.chain = None
        self._init_chain()
    
    def _init_chain(self):
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.1,
                num_ctx=8192
            )
            
            # Test the LLM connection
            test_response = self.llm.invoke("Hello")
            print(f"LLM initialized successfully: {test_response.content[:30]}...")
            
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            self.llm = None
            return
        
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
        
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question):
        if self.llm is None:
            return "LLM模型未初始化，请检查Ollama服务是否正常运行", []
        
        if self.chain is None:
            return "问答链未初始化", []
        
        try:
            answer = self.chain.invoke(question)
            sources = self.retriever.invoke(question)
            
            if "未找到相关答案" not in answer and not answer.strip():
                answer = "文档中未找到相关答案"
            
            return answer, sources
        except Exception as e:
            print(f"问答失败: {e}")
            return f"问答过程中发生错误: {str(e)}", []
    
    def clear_memory(self):
        pass