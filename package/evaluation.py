import pandas as pd
from ast import literal_eval

from .kg_rag import RepositoryRAG


def calculate_precision_at_k(retrieved, relevant, k):
    if k > len(retrieved):
        k = len(retrieved)
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k & relevant_set) / k if k > 0 else 0.0


def calculate_recall_at_k(retrieved, relevant, k):
    if k > len(retrieved):
        k = len(retrieved)
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k & relevant_set) / len(relevant_set) if relevant_set else 0.0


def calculate_f1_at_k(retrieved, relevant, k):
    precision = calculate_precision_at_k(retrieved, relevant, k)
    recall = calculate_recall_at_k(retrieved, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class RAGEvaluator:
    def __init__(self, df: pd.DataFrame, rag_model: RepositoryRAG, k_values=[3, 5, 10]):
        self.df = df.copy()
        self.rag = rag_model
        self.k_values = k_values
        self._prepare_columns()

    def _prepare_columns(self):
        for k in self.k_values:
            self.df[f'precision_{k}'] = None
            self.df[f'recall_{k}'] = None
            self.df[f'f1_{k}'] = None

    def _get_top_functions(self, question: str, top_n: int):
        print("\nRetrieving top results...")

        functions_df = self.rag.retrieve_code_functions(question, top_n=top_n)
        issues_df = self.rag.retrieve_issues(question, top_n=top_n)
        prs_df = self.rag.retrieve_prs(question, top_n=top_n)

        reranked = self.rag.rerank_functions(functions_df, issues_df, prs_df, top_n=top_n)

        return list(reranked["combinedName"].values)

    def evaluate(self, verbose=True):
        for idx, row in self.df.iterrows():
            question = row["question"]
            context = row["edit_functions"]

            if verbose:
                print(f"Question: {question}")
                print(f"Golden context: {context}")

            
            top_functions = self._get_top_functions(question, top_n=max(self.k_values))
            
            if verbose:
                print(f"Retrieved functions: {top_functions}")

            for k in self.k_values:
                precision = calculate_precision_at_k(top_functions, context, k)
                recall = calculate_recall_at_k(top_functions, context, k)
                f1 = calculate_f1_at_k(top_functions, context, k)

                self.df.at[idx, f'precision_{k}'] = precision
                self.df.at[idx, f'recall_{k}'] = recall
                self.df.at[idx, f'f1_{k}'] = f1

                if verbose:
                    print(f"Precision@{k}: {precision}, Recall@{k}: {recall}, F1@{k}: {f1}")
            if verbose:
                print("-" * 40)

    def print_summary(self):
        print("Evaluation metrics:")
        for k in self.k_values:
            precision_mean = self.df[f'precision_{k}'].mean()
            recall_mean = self.df[f'recall_{k}'].mean()
            f1_mean = self.df[f'f1_{k}'].mean()
            print(f"Precision@{k}: {precision_mean:.4f}, Recall@{k}: {recall_mean:.4f}, F1@{k}: {f1_mean:.4f}")

    def export(self, path):
        self.df.to_csv(path, index=False)
