import os
from typing import List
import google.generativeai as genai
from langchain_core.documents import Document



class GeminiProcessor:
    """Enhanced class for processing queries with Gemini LLM."""

    def __init__(self, api_key=None):
        """Initialize the Gemini processor."""
        # Try to get API key from parameter, then environment variable
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            print("Warning: No API key provided for Gemini. Set GOOGLE_API_KEY environment variable.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
                print("Gemini model initialized successfully")
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.model = None
    
    def analyze_query_topics(self, query: str, num_topics: int = 3) -> List[str]:
        """Extract main topics from query to aid in document relevance."""
        if not self.model:
            return []
        
        try:
            prompt = f"""
            Identify the {num_topics} main topics or subjects in this query. 
            Return only a comma-separated list of topics without explanation.
            
            Query: {query}
            """
            
            response = self.model.generate_content(prompt)
            topics = [topic.strip() for topic in response.text.split(',')]
            return topics[:num_topics]  # Ensure we don't exceed requested number
        except Exception as e:
            print(f"Error analyzing query topics: {e}")
            return []

    def process_query(self, query: str, retrieved_docs: List[Document]) -> str:
        """Process a query using the Gemini LLM and retrieved documents."""
        if not self.model:
            return "Gemini model not initialized. Please provide a valid API key."

        try:
            # Prepare context with document metadata
            context_elements = []
            for i, doc in enumerate(retrieved_docs):
                # Extract document ID and other useful metadata
                doc_id = doc.metadata.get('doc_id', 'unknown')
                source = doc.metadata.get('source', 'unknown')
                content_type = doc.metadata.get('content_type', '')
                title = doc.metadata.get('title', '')
                
                # Format the context entry
                context_entry = f"Document {i+1} [{doc_id}]:\n"
                if title:
                    context_entry += f"Title: {title}\n"
                if content_type:
                    context_entry += f"Content type: {content_type}\n"
                context_entry += f"Content: {doc.page_content}\n"
                
                context_elements.append(context_entry)
                
            context = "\n\n".join(context_elements)

            # Enhanced prompt with metadata awareness
            prompt = f"""
            You are an intelligent assistant that answers questions based on provided context.
            The context consists of information from various documents, each with its own ID and metadata.

            Context information:
            {context}
            
            User question: {query}
            
            Please answer the question based only on the provided context. If the context doesn't contain enough information
            to fully answer the question, acknowledge what you can answer based on the context and what information is missing.
            When referring to specific information, mention which document it comes from.
            """

            # Generate response
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error processing query with Gemini: {e}")
            return f"Error processing query: {str(e)}"