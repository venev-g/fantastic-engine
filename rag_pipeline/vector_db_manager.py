from langchain_community.vectorstores import Milvus
from langchain_neo4j import Neo4jVector, Neo4jGraph
from rag_pipeline.embedding_manager import EmbeddingManager
from typing import List, Dict
from langchain_core.documents import Document
import re



class VectorDBManager:
    """Class to manage vector database connections and operations."""

    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize with the embedding manager."""
        self.embedding_manager = embedding_manager
        self.milvus_client = None
        self.neo4j_client = None

        # Collections in Milvus (document_id -> collection)
        self.collections = {}
        
        # Document metadata store
        self.doc_metadata = {}

    def setup_milvus(self, host='localhost', port='2379'):
        """Set up connection to Milvus."""
        try:
            # Define Milvus connection parameters
            self.milvus_connection_params = {
                "host": host,
                "port": port
            }
            
            print("Milvus connection parameters configured.")
            self.milvus_available = True
            
            # Check if connection works
            from pymilvus import connections
            connections.connect(host=host, port=port)
            print("Milvus connection test successful")
            connections.disconnect(alias="default")
            
        except Exception as e:
            print(f"Error setting up Milvus: {e}")
            self.milvus_available = False

    def setup_neo4j(self, uri='bolt://localhost:7687', username='neo4j', password='venev'):
        """Set up connection to Neo4j."""
        try:
            # Store Neo4j connection parameters
            self.neo4j_connection_params = {
                "url": uri,
                "username": username,
                "password": password
            }
            
            # Initialize Neo4j graph client
            self.neo4j_client = Neo4jGraph(
                url=uri,
                username=username,
                password=password
            )
            
            # Initialize Neo4j vector client
            self.neo4j_vector = Neo4jVector(
                url=uri,
                username=username,
                password=password,
                embedding=self.embedding_manager.embeddings,
                index_name="documentVectors",
                node_label="DocumentChunk"
            )
            
            # Set up schema for our document graph (only once)
            self._create_neo4j_schema()
            print("Neo4j connection established and schema created.")
            self.neo4j_available = True
        except Exception as e:
            print(f"Error setting up Neo4j: {e}")
            self.neo4j_client = None
            self.neo4j_vector = None
            self.neo4j_available = False

    def _create_neo4j_schema(self):
        """Create Neo4j schema for document storage."""
        # Create constraints that work with Community Edition
        self.neo4j_client.query("""
            CREATE CONSTRAINT document_id IF NOT EXISTS
            FOR (d:Document) REQUIRE d.id IS UNIQUE
        """)
        
        self.neo4j_client.query("""
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE
        """)
        
        # Create vector index if needed
        self.neo4j_client.query("""
            CREATE VECTOR INDEX documentVectors IF NOT EXISTS
            FOR (c:DocumentChunk) 
            ON c.embedding
            OPTIONS {indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }}
        """)

    def store_in_milvus(self, documents: List[Document], doc_id: str, content_type: str):
        """Store documents in Milvus using doc_id for collection naming."""
        if not hasattr(self, 'milvus_available') or not self.milvus_available:
            print("Skipping Milvus storage - Milvus not available")
            return
            
        try:
            # Sanitize the document ID and content type to create a valid collection name
            # Replace spaces and special characters with underscores
            sanitized_doc_id = re.sub(r'[^a-zA-Z0-9_]', '_', doc_id)
            sanitized_content_type = re.sub(r'[^a-zA-Z0-9_]', '_', content_type)
            
            # Create a collection name that includes sanitized document ID and content type
            collection_name = f"doc_{sanitized_doc_id}_{sanitized_content_type}"
            
            print(f"Using sanitized collection name: {collection_name}")
            
            # Generate embeddings for documents
            embeddings = [self.embedding_manager.generate_embedding(doc.page_content) for doc in documents]

            # Create Milvus collection
            vector_store = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_manager.embeddings,
                collection_name=collection_name,
                connection_args=self.milvus_connection_params
            )

            # Store the collection reference with original doc_id
            if doc_id not in self.collections:
                self.collections[doc_id] = {}
            
            self.collections[doc_id][content_type] = vector_store

            print(f"Stored {len(documents)} documents in Milvus collection: {collection_name}")

        except Exception as e:
            print(f"Error storing documents in Milvus: {e}")

    def store_in_neo4j(self, triplets: List[Dict], doc_id: str):
        """Store triplets in Neo4j with document ID."""
        if not hasattr(self, 'neo4j_available') or not self.neo4j_available:
            print("Skipping Neo4j storage - Neo4j not available")
            return
            
        try:
            # Get the Neo4j driver session properly
            session = self.neo4j_client._driver.session()
            
            # Process in smaller batches for better error handling
            batch_size = 50
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i+batch_size]
                try:
                    # Begin a transaction for this batch
                    with session.begin_transaction() as tx:
                        # Convert parameters for Neo4j
                        params = {
                            "batch_size": len(batch),
                            "subjects": [t["subject"] for t in batch],
                            "predicates": [t["predicate"] for t in batch],
                            "objects": [t["object"] for t in batch],
                            "embeddings": [t["embedding"] for t in batch],
                            "doc_id": doc_id
                        }

                        # Enhanced Cypher query with document ID
                        query = """
                        UNWIND range(0, $batch_size - 1) as i
                        MERGE (d:Document {id: $doc_id})
                        MERGE (s:Subject {name: $subjects[i], doc_id: $doc_id})
                        MERGE (o:Object {name: $objects[i], doc_id: $doc_id})
                        WITH d, s, o, i
                        CREATE (s)-[r:HAS_PROPERTY {name: $predicates[i]}]->(o)
                        CREATE (s)-[:BELONGS_TO]->(d)
                        CREATE (o)-[:BELONGS_TO]->(d)
                        WITH d, s, o, i
                        CREATE (c:DocumentChunk {
                            id: randomUUID(),
                            text: $subjects[i] + ' ' + $predicates[i] + ' ' + $objects[i],
                            subject: $subjects[i],
                            predicate: $predicates[i],
                            object: $objects[i],
                            doc_id: $doc_id
                        })
                        SET c.embedding = $embeddings[i]
                        CREATE (c)-[:SUBJECT_OF]->(s)
                        CREATE (c)-[:OBJECT_OF]->(o)
                        CREATE (c)-[:PART_OF]->(d)
                        """

                        # Execute query within transaction
                        result = tx.run(query, params)
                        result.consume()  # Ensure execution completes
                    
                    print(f"Processed batch {i//batch_size + 1}/{(len(triplets) + batch_size - 1)//batch_size}")
                        
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Continue to the next batch even if this one fails
            
            # Close the session when done
            session.close()
            print("Stored triplets in Neo4j (completed batches)")

        except Exception as e:
            print(f"Error initializing Neo4j transaction: {e}")
            # Make sure session is closed if an error occurs
            if 'session' in locals():
                session.close()

    def register_document_metadata(self, doc_id: str, metadata: Dict):
        """Register document metadata for later reference."""
        self.doc_metadata[doc_id] = metadata
        
    def get_document_metadata(self, doc_id: str) -> Dict:
        """Get metadata for a document."""
        return self.doc_metadata.get(doc_id, {})
        
    def list_documents(self) -> List[str]:
        """List all documents in the system."""
        return list(self.collections.keys())

    def document_exists(self, doc_id: str) -> bool:
        """Check if document already exists in the system."""
        return doc_id in self.collections