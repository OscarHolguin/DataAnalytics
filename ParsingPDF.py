# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:45:22 2023

@author: o_hol
"""


from dataclasses import dataclass
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

@dataclass
class PDFParser:
    """
    Sample ussage : 
        path
        parser = PDFParser(path,**kwargs)
        documents = parser.parse_pdf()
    """
    
    pdf_path: str
    extract_images: bool = False
    chunked: bool = False
    online: bool = False
    max_tokens:int = 2048
    chunk_overlap: int = 64
    huggingembed: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        self.loader = self.get_loader()

    def get_loader(self):
        if self.chunked:
            return UnstructuredPDFLoader(self.pdf_path)
        elif self.online:
            return OnlinePDFLoader(self.pdf_path)
        else:
            return PyPDFLoader(self.pdf_path, extract_images=self.extract_images)

    def parse_pdf(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = self.loader.load_and_split(text_splitter=text_splitter)
        return documents
    
    def get_embeddings(self,documents=None,lib='langchain'):
#         if not documents:
#             documents = pass
        #huggingface or langchain
        if lib=='langchain':
            from langchain.embeddings import OpenAIEmbeddings
            embeddings_model = OpenAIEmbeddings()
            embeddings = embeddings_model.embed_documents(documents)
            return embeddings
        else:
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=self.huggingembed)
            return embeddings
        


def preprocess_text(text):
    from string import punctuation
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words_en = set(stopwords.words("english"))
    stop_words_es = set(stopwords.words("spanish"))
    stop_words = stop_words_en.union(stop_words_es)
    text = ' '.join([word for word in word_tokenize(text) if word not in punctuation and word not in stop_words])
    return text



def get_pdf_url(uploaded_file):
    import base64
    # Function to get the URL for the uploaded PDF file
    # Convert the bytes object to base64-encoded string
    pdf_data_base64 = base64.b64encode(uploaded_file.getvalue()).decode()
    pdf_data_url = f"data:application/pdf;base64,{pdf_data_base64}"
    return pdf_data_url

#example 
if __name__ == "main":
        path = r""
        parser = PDFParser(path,extract_images=True,max_tokens=2000)
        documents = parser.parse_pdf()
    
    

        



