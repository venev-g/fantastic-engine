from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from rag_pipeline.document_analyzer import DocumentAnalyzer
import pandas as pd


class DocumentChunker:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def hierarchical_chunking(self, document_structure: Dict) -> List[Document]:
        """
        Perform Title + Content (Hierarchical) chunking.

        Args:
            document_structure: Document structure dictionary

        Returns:
            List of LangChain Document objects
        """
        # First convert the document structure to a format suitable for hierarchical processing
        markdown_content = self._convert_to_markdown_with_headers(document_structure)

        # Split using the markdown header splitter
        md_header_splits = self.md_header_splitter.split_text(markdown_content)

        # Further split long content chunks if necessary
        final_chunks = []
        for doc in md_header_splits:
            # If content is too long, split further
            content = doc.page_content
            if len(content) > 1000:  # adjust threshold as needed
                sub_chunks = self.recursive_splitter.split_text(content)
                # Copy metadata to each sub-chunk
                for chunk in sub_chunks:
                    final_chunks.append(Document(
                        page_content=chunk,
                        metadata={**doc.metadata, 'chunk_type': 'hierarchical'}
                    ))
            else:
                doc.metadata['chunk_type'] = 'hierarchical'
                final_chunks.append(doc)

        return final_chunks
    
    def paragraph_based_chunking(self, document_structure: Dict) -> List[Document]:
        """
        Perform Paragraph-Based chunking with line breaks preserved.

        Args:
            document_structure: Document structure dictionary

        Returns:
            List of LangChain Document objects
        """
        chunks = []

        # Extract paragraphs from the document
        paragraphs = []
        current_section = {'title': 'Document', 'content': ''}

        for element in document_structure['elements']:
            if element['type'] == 'heading':
                # If we have content in the current section, save it
                if current_section['content'].strip():
                    paragraphs.append(current_section)

                # Start a new section
                current_section = {
                    'title': element['content'],
                    'content': ''
                }
            elif element['type'] == 'paragraph':
                current_section['content'] += element['content'] + "\n\n"

        # Add the last section if it has content
        if current_section['content'].strip():
            paragraphs.append(current_section)

        # Create Document objects for each paragraph
        for para in paragraphs:
            # Split content if it's too long
            if len(para['content']) > 1000:
                sub_chunks = self.recursive_splitter.split_text(para['content'])
                for i, chunk in enumerate(sub_chunks):
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            'title': para['title'],
                            'chunk_type': 'paragraph',
                            'chunk_index': i,
                            'source': document_structure['metadata']['source']
                        }
                    ))
            else:
                chunks.append(Document(
                    page_content=para['content'],
                    metadata={
                        'title': para['title'],
                        'chunk_type': 'paragraph',
                        'source': document_structure['metadata']['source']
                    }
                ))

        return chunks

    def table_aware_chunking(self, document_structure: Dict) -> List[Dict]:
        """
        Perform Table-Aware chunking for Neo4j graph.

        Args:
            document_structure: Document structure dictionary

        Returns:
            List of dictionaries representing table triplets for Neo4j
        """
        # Extract tables and convert to triplets
        analyzer = DocumentAnalyzer()
        return analyzer.extract_tables_as_triplets(document_structure)

    def metadata_aware_chunking(self, document_structure: Dict) -> List[Document]:
        """
        Perform Metadata-Aware chunking with Milvus compatibility.
        """
        chunks = []
        
        # Identify semantic sections
        semantic_sections = self._identify_semantic_sections(document_structure)
        
        for section in semantic_sections:
            # Calculate chunk size based on content type
            if section['type'] == 'rule_explanation':
                chunk_size = 1500
            elif section['type'] == 'user_scenario':
                chunk_size = 1000
            elif section['type'] == 'business_logic':
                chunk_size = 800
            else:
                chunk_size = 1000
                
            custom_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=min(100, chunk_size // 10),
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split content
            text_chunks = custom_splitter.split_text(section['content'])
            
            # Convert keywords and categories to strings for Milvus compatibility
            keywords_str = ",".join(section.get('keywords', []))
            categories_str = ",".join(section.get('categories', []))
            
            # Create Document objects with compatible metadata
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        'title': section.get('title', 'Untitled Section'),
                        'content_type': section['type'],
                        'source': document_structure['metadata']['source'],
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'chunk_type': 'metadata_aware',
                        # Store keywords and categories as strings
                        'keywords_str': keywords_str,
                        'categories_str': categories_str
                    }
                ))
        
        return chunks

    def _convert_to_markdown_with_headers(self, document_structure: Dict) -> str:
        """
        Convert document structure to markdown format with headers.

        Args:
            document_structure: Document structure dictionary

        Returns:
            Markdown string representation of the document
        """
        markdown = []
        current_heading = ""

        for element in document_structure['elements']:
            if element['type'] == 'heading':
                level = element.get('level', 1)
                heading = '#' * level + ' ' + element['content']
                markdown.append(heading)
                current_heading = element['content']
            elif element['type'] == 'paragraph':
                markdown.append(element['content'])
            elif element['type'] == 'table':
                if isinstance(element['content'], pd.DataFrame):
                    table_str = element['content'].to_markdown()
                    markdown.append(table_str)
                else:
                    markdown.append(str(element['content']))
            elif element['type'] in ['strikeout', 'highlight', 'image']:
                markdown.append(element['content'])

        return '\n\n'.join(markdown)

    def _identify_semantic_sections(self, document_structure: Dict) -> List[Dict]:
        """
        Identify semantic sections in the document.

        Args:
            document_structure: Document structure dictionary

        Returns:
            List of semantic section dictionaries
        """
        sections = []
        current_section = None

        # Patterns to identify content types
        rule_patterns = ['rule', 'regulation', 'policy', 'guidelines']
        logic_patterns = ['workflow', 'process', 'logic', 'procedure', 'business logic']
        scenario_patterns = ['scenario', 'example', 'use case', 'user story']
        interaction_patterns = ['interaction', 'interface', 'system', 'integration', 'api']

        for element in document_structure['elements']:
            if element['type'] == 'heading':
                # If there's a current section with content, save it
                if current_section and current_section.get('content'):
                    sections.append(current_section)

                # Determine the section type based on heading content
                heading_lower = element['content'].lower()

                if any(pattern in heading_lower for pattern in rule_patterns):
                    section_type = 'rule_explanation'
                elif any(pattern in heading_lower for pattern in logic_patterns):
                    section_type = 'business_logic'
                elif any(pattern in heading_lower for pattern in scenario_patterns):
                    section_type = 'user_scenario'
                elif any(pattern in heading_lower for pattern in interaction_patterns):
                    section_type = 'system_interaction'
                else:
                    section_type = 'general'

                # Create a new section
                current_section = {
                    'title': element['content'],
                    'type': section_type,
                    'content': '',
                    'keywords': self._extract_keywords(element['content']),
                    'categories': [section_type]
                }
            elif element['type'] in ['paragraph', 'strikeout', 'highlight']:
                if current_section:
                    current_section['content'] += element['content'] + "\n\n"
                else:
                    # Create a default section if none exists
                    current_section = {
                        'title': 'Introduction',
                        'type': 'general',
                        'content': element['content'] + "\n\n",
                        'keywords': [],
                        'categories': ['general']
                    }
            elif element['type'] == 'table':
                # Tables are handled separately for Neo4j, but we still include their text representation
                # in the content for completeness
                if current_section:
                    if isinstance(element['content'], pd.DataFrame):
                        current_section['content'] += "Table content:\n"
                        current_section['content'] += str(element['content']) + "\n\n"
                    else:
                        current_section['content'] += "Table content:\n" + str(element['content']) + "\n\n"

        # Add the last section if it exists
        if current_section and current_section.get('content'):
            sections.append(current_section)

        return sections

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        # Simple implementation - extract meaningful words and filter stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                     'on', 'in', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]

        # Return unique keywords
        return list(set(keywords))