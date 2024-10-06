# Ensure pandas is installed
# If you're using a virtual environment or need to install pandas,
# uncomment the following line and run it separately:
# !pip install pandas
from langchain_community.document_loaders import CSVLoader  # For loading CSV files from directories
from langchain_core.documents import Document  # Base document class for working with text data
from langchain_chroma import Chroma  # Vector store for efficient similarity search
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # For creating embeddings using HuggingFace models
from langchain_openai import OpenAIEmbeddings  # For creating embeddings using OpenAI models

import re
import os
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv('.env') # looks for .env in Python script directory unless path is provided
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Document locations (relative to this py file)
folder_paths = ['data']

def convert_to_text(folders):
    """
    Convert CSV documents from specified folders to a list of LangChain documents.

    This function iterates through the provided folder paths, loads all CSV documents
    found in each folder, and converts them to LangChain Document objects. Each Document
    object contains the text content of a CSV row and associated metadata.

    Parameters:
    -----------
    folders : list of str
        A list of folder paths containing CSV documents to be processed.

    Returns:
    --------
    list of Document
        A list of LangChain Document objects, where each object represents a row
        from the processed CSV files.

    Notes:
    ------
    - The function uses CSVLoader to load CSV files.
    - Each Document object includes:
        - page_content: The text content of a CSV row.
        - metadata: Information about the source document, including the row number.
    - Missing or empty folders are silently ignored.
    """
    all_docs = []
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                file_path = os.path.join(folder, file)
                loader = CSVLoader(file_path)
                all_docs.extend(loader.load())
    
    return all_docs 

# function to clean text -> redact SSN/TIN
def prepare_text(docs):
    '''
     Performs various data clean up steps

            Parameters:
                    docs (langchain docs): Text 

            Returns:
                    clean_docs (langchain docs): Clean text
     '''
    # pattern = re.compile(r'.{4}-.{2}')
    
    # clean_docs = list(map(lambda v: Document(page_content=pattern.sub('REDACTED', v.page_content), metadata=v.metadata), docs))
    clean_docs = docs
    return clean_docs

# Function to chunk text (CSV-specific chunking)
def chunk_text(unchunked_docs):
    """
    Chunk CSV documents into meaningful segments.

    This function uses a CSV-specific approach to split documents. It first combines
    all rows for each unique combination of non-numeric columns, then splits the
    resulting text based on a maximum chunk size.

    Parameters:
        unchunked_docs (list of Document): List of CSV documents to be chunked.

    Returns:
        list of Document: List of chunked documents.
    """
    chunked_docs = []
    
    # Group documents by non-numeric columns (assumed to be identifiers)
    grouped_docs = {}
    for doc in unchunked_docs:
        df = pd.read_json(doc.page_content)
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        key = tuple(df[non_numeric_cols].iloc[0])
        if key not in grouped_docs:
            grouped_docs[key] = []
        grouped_docs[key].append(doc)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Process each group
    for key, docs in grouped_docs.items():
        combined_text = "\n".join([doc.page_content for doc in docs])
        chunks = text_splitter.split_text(combined_text)
        
        for chunk in chunks:
            metadata = docs[0].metadata.copy()  # Use metadata from the first document in the group
            metadata['group_key'] = key
            chunked_docs.append(Document(page_content=chunk, metadata=metadata))
    
    return chunked_docs


def insert_into_vector_db(chunked_docs):
    """
    Inserts chunked text documents into a vector database.

    This function takes pre-chunked text documents and inserts them into a Chroma vector database.
    It uses OpenAI's text embedding model for creating vector representations of the documents.

    Parameters:
        chunked_docs (list of Document): List of chunked text documents to be inserted.

    Note:
        The embedding model used here may differ from the one used for chunking.
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large')

    # Initialize Chroma vector store
    alcon_vectorstore = Chroma(
        collection_name='alcon_collection_financial_statements',
        embedding_function=embeddings,
        collection_metadata={'hnsw:space': 'cosine'},
        persist_directory='./chroma_langchain_db'
    )

    # Insert documents into the vector store
    for i, doc in enumerate(chunked_docs):
        document = Document(
            page_content=doc.page_content,
            metadata=doc.metadata,  # Preserve original metadata
            id=i
        )
        alcon_vectorstore.add_documents(documents=[document], ids=[str(i)])

def main():
    """
    Main function to process documents and populate the vector database.

    This function orchestrates the entire process of converting documents to text,
    preparing the text, chunking it, and inserting it into the vector database.
    It also measures and prints the execution time for each step and the total process.
    """
    start_time = time.time()

    # Step 1: Convert documents from various folders to text
    step1_start = time.time()
    all_docs = convert_to_text(folder_paths)
    step1_end = time.time()
    print(f"Step 1 (Convert to text) took {step1_end - step1_start:.2f} seconds")

    # Step 2: Clean and prepare the text documents
    step2_start = time.time()
    clean_docs = prepare_text(all_docs)
    step2_end = time.time()
    print(f"Step 2 (Prepare text) took {step2_end - step2_start:.2f} seconds")

    # Step 3: Chunk the cleaned text into smaller, manageable pieces
    step3_start = time.time()
    chunked_docs = chunk_text(clean_docs)
    step3_end = time.time()
    print(f"Step 3 (Chunk text) took {step3_end - step3_start:.2f} seconds")

    # Step 4: Insert the chunked documents into the vector database
    step4_start = time.time()
    insert_into_vector_db(chunked_docs)
    step4_end = time.time()
    print(f"Step 4 (Insert into vector DB) took {step4_end - step4_start:.2f} seconds")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()