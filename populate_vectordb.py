from langchain_community.document_loaders import PyPDFDirectoryLoader  # For loading PDF files from directories
from langchain_core.documents import Document  # Base document class for working with text data
from langchain_chroma import Chroma  # Vector store for efficient similarity search
from langchain_experimental.text_splitter import SemanticChunker  # For splitting text into semantic chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # For creating embeddings using HuggingFace models
from langchain_openai import OpenAIEmbeddings  # For creating embeddings using OpenAI models

import re
import os
from dotenv import load_dotenv
import numpy as np

#from langchain_community.document_loaders import PyPDFLoader
#from langchain.schema import Document
#from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv('.env') # looks for .env in Python script directory unless path is provided
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Document locations (relative to this py file)
folder_paths = ['data']

def convert_to_text(folders):
    """
    Convert PDF documents from specified folders to a list of LangChain documents.

    This function iterates through the provided folder paths, loads all PDF documents
    found in each folder, and converts them to LangChain Document objects. Each Document
    object contains the text content of a PDF page and associated metadata.

    Parameters:
    -----------
    folders : list of str
        A list of folder paths containing PDF documents to be processed.

    Returns:
    --------
    list of Document
        A list of LangChain Document objects, where each object represents a page
        from the processed PDF files.

    Notes:
    ------
    - The function uses PyPDFDirectoryLoader to load PDF files.
    - Each Document object includes:
        - page_content: The text content of a PDF page.
        - metadata: Information about the source document, including an index
          (starting from 0) that does not necessarily correspond to the PDF page number.
    - Missing or empty folders are silently ignored.
    - For more details on PDF loading, see:
      https://python.langchain.com/docs/how_to/document_loader_pdf/
    """
    all_docs = []
    for folder in folders:
        # Load PDFs from each folder
        # Note: Missing or empty folders are ignored without raising errors
        loader_folder = PyPDFDirectoryLoader(folder)
        
        # Add loaded documents to the list
        # Using load() instead of load_and_split() for basic page-level splitting
        all_docs.extend(loader_folder.load())
        
        # Alternative with more refined splitting (commented out):
        # all_docs.extend(loader_folder.load_and_split())
    
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
    pattern = re.compile(r'.{4}-.{2}')
    
    clean_docs = list(map(lambda v: pattern.sub('REDACTED', v.page_content), docs)) 
    return clean_docs

# Function to chunk text (semantic chunking)
def chunk_text(unchunked_doc):
    """
    Chunk text documents into semantically meaningful segments.

    This function uses a semantic chunking approach to split documents based on
    content similarity. It employs the gradient of distance method along with
    percentile-based thresholding, which is particularly effective for
    domain-specific content (e.g., legal or medical texts).

    Parameters:
        unchunked_doc (list of str): List of text documents to be chunked.

    Returns:
        list of Document: List of chunked documents.

    References:
        - Semantic Chunker: https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/
        - Text Splitting Techniques: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """    

    # Define the embedding model for chunking
    # chunking_model_name = "nvidia/nv-embed-v2" # Alternative model: https://huggingface.co/nvidia/NV-Embed-v2
    chunking_model_name = 'sentence-transformers/all-mpnet-base-v2' # Default model

    # Initialize the semantic chunker with the chosen embedding model
    text_splitter = SemanticChunker(
        HuggingFaceEmbeddings(model_name=chunking_model_name),
        breakpoint_threshold_type='gradient'
    ) 

    # Note: May need to install sentence-transformers (pip install sentence-transformers)
    # This might replace tokenizers-0.20.0 with tokenizers-0.19.1

    # Perform the chunking operation
    # TODO: Investigate long processing time (45 minutes reported) and potential metadata loss
    chunked_docs = []
    for doc in unchunked_doc:
        try:
            chunks = text_splitter.split_text(doc)
            # Convert each chunk to a Document object
            chunked_docs.extend([Document(page_content=chunk) for chunk in chunks])
        except IndexError:
            # If an IndexError occurs, skip this document and continue with the next
            print(f"Warning: Skipping a document due to IndexError. Document content: {doc[:100]}...")
            continue

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
    # Reference: https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
    uw_vectorstore = Chroma(
        collection_name='uw_collection',
        embedding_function=embeddings,
        collection_metadata={'hnsw:space': 'cosine'},
        persist_directory='./chroma_langchain_db'
    )
    # Note: The cosine parameter may not affect results unless using Chroma's similarity function

    # Insert documents into the vector store
    for i, doc in enumerate(chunked_docs):
        document = Document(
            page_content=doc.page_content,
            metadata={'source': 'IRS'},  # TODO: Determine appropriate metadata source (e.g., 'IRS', 'UW')
            id=i
        )
        uw_vectorstore.add_documents(documents=[document], ids=[str(i)])

    # Commented out alternative implementation:
    # for i in range(1, len(chunked_docs)):
    #     document = Document(page_content=chunked_docs[i].page_content, metadata={"source": "IRS"}, id=i)
    #     uw_vectorstore.add_documents(documents=[document], ids=[str(i)])

def main():
    """
    Main function to process documents and populate the vector database.

    This function orchestrates the entire process of converting documents to text,
    preparing the text, chunking it, and inserting it into the vector database.
    """
    # Step 1: Convert documents from various folders to text
    all_docs = convert_to_text(folder_paths)

    # Step 2: Clean and prepare the text documents
    clean_docs = prepare_text(all_docs)

    # Step 3: Chunk the cleaned text into smaller, manageable pieces
    chunked_docs = chunk_text(clean_docs)

    # Step 4: Insert the chunked documents into the vector database
    insert_into_vector_db(chunked_docs)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()