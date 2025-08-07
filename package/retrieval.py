# Data
import pandas as pd
import numpy as np
import torch

import networkx as nx
import seaborn as sns
from pyvis.network import Network

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from sklearn.metrics.pairwise import cosine_similarity

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
        keyword_embeddings = self._extract_keywords(text)

        search_text = df[column_name].astype(str).tolist()

        # Compute embeddings for the search text
        search_embeddings = self.model.encode(search_text, convert_to_tensor=True).cpu().tolist()

        # Compute cosine similarities between the query and search text
        cosine_similarities = cosine_similarity([keyword_embeddings], search_embeddings)[0]

        df_result = df.copy()
        df_result['similarity'] = cosine_similarities

        df_result = df_result.sort_values(by='similarity', ascending=False)

        if isinstance(threshold, int):
            # If threshold is an integer, return top k results
            df_result = df_result.head(threshold).reset_index(drop=True)
        elif isinstance(threshold, float) and 0 < threshold < 1:
            # If threshold is a float, filter results based on similarity score
            df_result = df_result[df_result['similarity'] >= threshold].reset_index(drop=True)
        else:
            raise ValueError("Threshold must be an integer (top k results) or a float (between 0 and 1).")
            
        return df_result
        

        

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