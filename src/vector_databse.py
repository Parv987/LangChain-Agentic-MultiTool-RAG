import faiss
import numpy as np
import csv
import os


def save_vector_store(embeddings, documents_text, sources, store_dir="vectorstore"):
    """Save embeddings and metadata to a FAISS index and CSV file."""
    if not embeddings:
        raise ValueError("embeddings list is empty")

    embedding_dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(np.array(embeddings, dtype="float32"))

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    # write metadata CSV without requiring pandas
    with open(f"{store_dir}/docs.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["documents", "source"])
        for doc, src in zip(documents_text, sources):
            writer.writerow([doc, src])
    # also save embeddings for later reconstruction
    np.save(f"{store_dir}/embeddings.npy", np.array(embeddings, dtype="float32"))

    faiss.write_index(index, f"{store_dir}/vector_db.index")

    print("FAISS index and embeddings saved successfully")


def load_vector_store(store_dir="vectorstore", embedding_model=None, k=4):
    """Load FAISS index and metadata and return a simple retriever.

    The returned object implements `get_relevant_documents(query)` which
    returns a list of objects with `page_content` and `metadata` attributes.
    """
    index_path = os.path.join(store_dir, "vector_db.index")
    docs_path = os.path.join(store_dir, "docs.csv")
    emb_path = os.path.join(store_dir, "embeddings.npy")

    if not (os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(emb_path)):
        raise FileNotFoundError("Saved vector store not found in %s" % store_dir)

    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    # read metadata CSV
    rows = []
    with open(docs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    class SimpleDoc:
        def __init__(self, text, meta):
            self.page_content = text
            self.metadata = meta

    class SimpleFaissRetriever:
        def __init__(self, index, embeddings, df, embedding_model, k=4):
            self.index = index
            self.embeddings = embeddings
            self.df = df
            self.embedding_model = embedding_model
            self.k = k

        def _embed_query(self, query):
            if self.embedding_model is None:
                raise ValueError("Embedding model required to run query through retriever")
            if hasattr(self.embedding_model, 'embed_query'):
                return self.embedding_model.embed_query(query)
            # fallback to embed_documents
            return self.embedding_model.embed_documents([query])[0]

        def get_relevant_documents(self, query):
            q_emb = self._embed_query(query)
            q_emb = np.array([q_emb], dtype='float32')
            D, I = self.index.search(q_emb, self.k)
            results = []
            for idx in I[0]:
                if idx < 0 or idx >= len(rows):
                    continue
                row = rows[idx]
                meta = {'source': row.get('source', '')}
                results.append(SimpleDoc(row.get('documents', ''), meta))
            return results

    return SimpleFaissRetriever(index, embeddings, df, embedding_model, k=k)
