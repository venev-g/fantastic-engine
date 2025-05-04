import os
import re
import PyPDF2
import pandas as pd
import numpy as np
from docx import Document
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.docstore.document import Document as LangchainDocument
from langchain.schema import Document as LCDocument
from langchain_community.retrievers import MilvusRetriever
from langchain_community.graphs import Neo4jGraph
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp




