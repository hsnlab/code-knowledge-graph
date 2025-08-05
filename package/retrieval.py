# Data
import pandas as pd
import numpy as np
import torch

import networkx as nx
import seaborn as sns
from pyvis.network import Network

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from sklearn import metrics

import spacy



class Retrieval():

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")



    def retrieve(self, text: str, df: pd.DataFrame, column_name: str, threshold=5) -> pd.DataFrame:
        """        
        Retrieve the most relevant rows from a DataFrame based on a text query.
        Args:
            text (str): The input text query.
            df (pd.DataFrame): The DataFrame to search in.
            column_name (str): The column name in the DataFrame to search.
            threshold (int/float): If int, returns the top_k results. If float (between 0 and 1), returns results above the threshold similarity.
        Returns:
            pd.DataFrame: A DataFrame containing the top results.
        """


        # Extract keywords from the input text
        keywords = self._extract_keywords(text)

        search_text = df[column_name].astype(str).tolist()

        # Compute embeddings for the search text
        search_embeddings = self.model.encode(search_text, convert_to_tensor=True).cpu().tolist()

        # Compute cosine similarities between the query and search text
        cosine_similarities = metrics.pairwise.cosine_similarity(keywords, search_embeddings)

        cos_dict = dict(zip(search_text, cosine_similarities))

        if isinstance(threshold, int):
            # If threshold is an integer, return the top k results
            top_k = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)[:threshold]
            top_k_df = pd.DataFrame(top_k, columns=[column_name, 'similarity'])
            return top_k_df

        elif isinstance(threshold, float):
            # If threshold is a float, return results above the threshold similarity
            filtered = {k: v for k, v in cos_dict.items() if v > threshold}
            filtered_df = pd.DataFrame(filtered.items(), columns=[column_name, 'similarity'])
            return filtered_df
        

        

    def _extract_keywords(self, text: str, min_token_len: int = 2) -> list:
        """
        Extract important words from a programming-related question.
        Filters out stopwords, punctuation, and selects key POS tags (nouns, verbs, adjectives).
        
        Args:
            question (str): Input question string.
            min_token_len (int): Minimum length of a word to be considered.

        Returns:
            List of important keywords.
        """
        doc = self.nlp(text)
        
        # Keep tokens that are nouns, proper nouns, verbs, adjectives
        keywords = [
            token.text for token in doc
            if token.is_alpha and not token.is_stop and len(token.text) >= min_token_len
            and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
        ]

        # concat keywords to one string with spaces
        keyword_string = ' '.join(keywords)

        keyword_embedding = self.model.encode(keyword_string, convert_to_tensor=True).cpu().tolist()
        
        return keyword_embedding