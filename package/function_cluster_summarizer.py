import pickle
import pandas as pd
import torch
from typing import List, Optional, Dict
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class FunctionClusterSummarizer:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", quantize:bool=False, huggingface_token: Optional[str] = None):
        self.llm_chain = self._load_chain(model_id, quantize, huggingface_token)

    def generate_summaries(self, df: pd.DataFrame, label_col:str = "label", fn_name_col:str = "combinedName") -> pd.DataFrame:
        summaries = {}
        for label, group_df in df.groupby(label_col):
            function_names = group_df[fn_name_col].tolist()
            fn_list_str = ", ".join(function_names)
            try:
                summary = self.llm_chain.invoke({"function_list": fn_list_str})
            except Exception as e:
                print(f"Error generating summary for label '{label}': {e}")
                summary = "Error generating summary."
            summaries[label] = summary

        df['summary'] = df[label_col].map(summaries)
        return df
        

    def _load_chain(self, model_id: str, quantize:bool, token: Optional[str]) -> LLMChain:
        """
        Loads the HuggingFacePipeline and wraps it into a LangChain LLMChain.

        :param model_id: Hugging Face model ID.
        :param token: Optional Hugging Face token for private models.
        :return: LLMChain instance for running prompts.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if quantize:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",quantization_config=bnb_config,torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            return_full_text=False,
        ))

        prompt = """You are analyzing a cluster of related Python functions extracted from a GitHub repository.

Here is a list of function names in this cluster:
{function_list}

Based on the names above, generate a short but informative summary (1 sentence, maximum of 10 words) describing the main purpose or theme of this cluster. 
If the functions seem very unrelated or random, still try to write a useful and coherent summary that reflects that — do not return empty or vague output.
The summary should be understandable and helpful for someone trying to understand what kind of functions are in the cluster.
        """
        prompt = PromptTemplate.from_template(prompt)

        llm_chain = prompt | llm
        return llm_chain

def main():
    function_cluster_summarizer = FunctionClusterSummarizer(model_id="mistralai/Mistral-7B-Instruct-v0.3", quantize=True)

    with open('sklearn.pkl', 'rb') as f:
        sklearn_hier_json = pickle.load(f)

    sklearn_hier_df = sklearn_hier_json["cg_nodes"].copy()
    sklearn_hier_df = sklearn_hier_df.merge(sklearn_hier_json["cluster_edges"], on="func_id", how="left")
    
    cluster_sum_df = function_cluster_summarizer.generate_summaries(sklearn_hier_df, label_col="cluster", fn_name_col="combinedName")
    cluster_sum_df = cluster_sum_df.rename(columns={"cluster":"cluster_id"})
    sklearn_hier_json["cg_nodes"] = cluster_sum_df

    # save the updated json
    with open('sklearn_with_summaries.pkl', 'wb') as f:
        pickle.dump(sklearn_hier_json, f)

if __name__ == "__main__":
    main()