from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.vector_databse import save_vector_store, load_vector_store


def build_retriever(source: str = "resume"):
    """Load PDF files from a directory or a single PDF path and build a retriever."""
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix.lower() == ".pdf":
        pdf_paths = [source_path]
    elif source_path.is_dir():
        pdf_paths = sorted(source_path.glob("*.pdf"))
    else:
        raise FileNotFoundError(f"No valid PDF source found at '{source}'.")

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{source}'.")

    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        for document in documents:
            document.metadata["source"] = pdf_path.name
            all_documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If saved vectorstore exists, load and return its retriever to avoid re-indexing
    store_dir = "vectorstore"
    try:
        retriever = load_vector_store(store_dir=store_dir, embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), k=4)
        print(f"Loaded retriever from saved vectorstore at {store_dir}")
        return retriever
    except Exception:
        # Fall back to building a new vectorstore
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        # Persist the vector store (FAISS index + metadata CSV + embeddings)
        try:
            embeddings = embedding_model.embed_documents([c.page_content for c in chunks])
            documents_text = [c.page_content for c in chunks]
            sources = [c.metadata.get("source", "") for c in chunks]
            save_vector_store(embeddings, documents_text, sources, store_dir=store_dir)
        except Exception as exc:
            print(f"Warning: failed to save vector store: {exc}")

        return vectorstore.as_retriever(search_kwargs={"k": 4})


# Note: earlier helper `load_and_split_pdf` removed — indexing and splitting
# are handled inside `build_retriever()` which loads PDFs, splits them into
# chunks, and builds or loads the FAISS retriever. Keeping this file focused
# on the retriever builder avoids duplicated functionality.