import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "./data"
DB_FAISS_PATH = "./vectorstore/db_faiss"

def run_ingestion():
    print(f"Scanning directory: {DATA_PATH}...")
    
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    all_files = pdf_files + txt_files
    
    print(f"Found {len(all_files)} files to process ({len(pdf_files)} PDFs, {len(txt_files)} TXTs).")
    
    documents = []
    for file_path in all_files:
        print(f"Loading {file_path}...")
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not documents:
        print("No documents found to ingest!")
        return False
        
    print(f"Loaded {len(documents)} document pages/sections. Splitting text...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from documents. Initializing embeddings model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Generating embeddings and creating FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to {DB_FAISS_PATH}...")
    db.save_local(DB_FAISS_PATH)
    print("Ingestion complete!")
    return True

if __name__ == "__main__":
    run_ingestion()
