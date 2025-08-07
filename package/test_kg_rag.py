# Data
import pandas as pd
import argparse
import numpy as np
import torch
from ast import literal_eval
import os
import sys
from kg_rag import RepositoryRAG
from evaluation import RAGEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG model using a test DataFrame.")
    parser.add_argument(
            "eval_df_path",
            default="../graph/sklearn/eval_df.csv",
            type=str,
            help="Path to the CSV file containing the evaluation dataframe."
        )
    args = parser.parse_args()
    
    eval_df_path = args.eval_df_path

    if not os.path.isfile(eval_df_path):
        print(f"Error: File '{eval_df_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading dataset from: {eval_df_path}")
    dataset = pd.read_csv(eval_df_path)
    dataset["edit_functions"] = dataset["edit_functions"].apply(literal_eval)

    sklearn_hier_json = pd.read_pickle("../graph/sklearn/sklearn.pkl")
    tool = RepositoryRAG(data_dict=sklearn_hier_json)

    evaluator = RAGEvaluator(df=dataset, rag_model=tool, k_values=[3, 5, 10])
    evaluator.evaluate(verbose=True)
    evaluator.print_summary()
    
    df_with_eval = evaluator.df
    # Save to same folder with _w_metrics.csv
    base_path, _ = os.path.splitext(eval_df_path)
    output_path = f"{base_path}_w_metrics.csv"
    df_with_eval.to_csv(output_path, index=False)
if __name__ == "__main__":
    main()