# LangChain-Agentic-MultiTool-RAG

## Vectorstore and running the agent

Rebuild the FAISS vectorstore (indexes embeddings and metadata):


Files will be saved under the `vectorstore/` folder:
- `vector_db.index` — FAISS index
- `embeddings.npy` — saved embeddings
- `docs.csv` — documents text and source metadata



Notes:
- `build_retriever()` will attempt to load the saved vectorstore from `vectorstore/` first. If not found, it will build and save the index automatically.
- To force reindexing, delete the `vectorstore/` folder and run the rebuild command above.
