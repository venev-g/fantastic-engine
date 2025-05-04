from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from document_processor import DocumentProcessor

class DocumentRetrievalSystem:
    def __init__(self, llm_path="models/llama-2-7b-chat.Q4_K_M.gguf"):
        """Initialize the document retrieval system."""
        # Initialize local LLM
        self.llm = LlamaCpp(
            model_path=llm_path,
            temperature=0.1,
            max_tokens=2048,
            n_ctx=4096
        )
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        self.document_processor.connect_to_databases()
        
        # Placeholders for vector stores
        self.vectorstores = {}
        
    def process_document(self, file_path):
        """Process a document and prepare it for retrieval."""
        self.vectorstores = self.document_processor.process_document(file_path)
        
    def build_retrieval_chain(self):
        """Build an optimized retrieval chain using LangChain."""
        # Create retrievers for each vector store
        retrievers = {}
        for name, vectorstore in self.vectorstores.items():
            # Create base retriever
            base_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Wrap with MultiQueryRetriever for query expansion
            retrievers[name] = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.llm
            )
        
        # Create a merged retriever that combines results from all sources
        merger_retriever = MergerRetriever(retrievers=list(retrievers.values()))
        
        # Create a custom prompt template
        template = """
        Answer the following question based on the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Provide a detailed answer that directly addresses the question.
        If the provided context doesn't contain enough information to answer the question,
        respond with "I don't have enough information to answer this question."
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=merger_retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def query_document(self, query):
        """Query the processed document."""
        if not self.vectorstores:
            raise ValueError("No document has been processed yet. Call process_document first.")
            
        # Build the retrieval chain
        qa_chain = self.build_retrieval_chain()
        
        # Execute the query
        result = qa_chain({"query": query})
        return result["result"]

    def query_neo4j(self, cypher_query):
        """Execute a direct Cypher query against Neo4j."""
        if not self.document_processor.neo4j_graph:
            raise ValueError("Neo4j connection not established.")
            
        return self.document_processor.neo4j_graph.query(cypher_query)