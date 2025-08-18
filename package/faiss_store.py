# faiss_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss


class FaissStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=50):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # create empty FAISS indices per data type
        dim = len(self.embeddings.embed_query("hello world"))
        self.stores = {
            "issues": FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(dim),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            ),
            "prs": FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(dim),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            ),
            "code": FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(dim),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            ),
        }

        # one splitter for all text types
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",  # still fine for English text, preserves code semantics
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_issues(self, issues_df):
        """Index issues into FAISS with chunking."""
        docs = []
        for _, row in issues_df.iterrows():
            text = f"{row['issue_title']} {row['issue_body']}"
            for chunk in self.splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"issue_number": row["issue_number"], "node_id": row["node_id"]},
                    )
                )
        if docs:
            self.stores["issues"].add_documents(docs)

    def add_prs(self, prs_df):
        """Index PRs into FAISS with chunking."""
        docs = []
        for _, row in prs_df.iterrows():
            text = f"{row['pr_title']} {row['pr_body']}"
            for chunk in self.splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"pr_number": row["pr_number"], "node_id": row["node_id"]},
                    )
                )
        if docs:
            self.stores["prs"].add_documents(docs)

    def add_code(self, code_df):
        """Index code functions into FAISS with chunking."""
        docs = []
        for _, row in code_df.iterrows():
            for chunk in self.splitter.split_text(row["code"]):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"func_id": row["func_id"], "node_id": row["node_id"]},
                    )
                )
        if docs:
            self.stores["code"].add_documents(docs)

    def search(self, query, index_type="issues", top_k=5):
        """Search one of the FAISS stores."""
        return self.stores[index_type].similarity_search(query, k=top_k)

    def save(self, path):
        """Save all FAISS indices to disk."""
        for key, store in self.stores.items():
            store.save_local(f"{path}/{key}")

    def load(self, path):
        """Load all FAISS indices from disk."""
        for key in self.stores.keys():
            self.stores[key] = FAISS.load_local(
                f"{path}/{key}", self.embeddings, allow_dangerous_deserialization=True
            )