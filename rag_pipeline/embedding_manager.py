import os
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document



class EmbeddingManager:
    """Class to manage embeddings for document chunks."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"Initialized embedding model: {model_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # The HuggingFaceEmbeddings class follows the LangChain embedding interface
        return self.embeddings.embed_query(text)

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_documents(self, documents: List[Document]) -> Tuple[List[List[float]], List[Document]]:
        """
        Embed LangChain Document objects.

        Args:
            documents: List of Document objects to embed

        Returns:
            Tuple of (embeddings, documents)
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.batch_embed(texts)
        return embeddings, documents

    def embed_triplets(self, triplets: List[Dict]) -> List[Dict]:
        """
        Embed triplets for Neo4j graph.

        Args:
            triplets: List of triplet dictionaries

        Returns:
            Triplets with embeddings added
        """
        for triplet in triplets:
            # Create a concatenated text representation of the triplet
            triplet_text = f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"
            triplet['embedding'] = self.generate_embedding(triplet_text)

        return triplets