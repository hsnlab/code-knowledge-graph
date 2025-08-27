from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import torch
from transformers import pipeline

import re

class SemanticClustering():

    def __init__(self, llm_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503"):
        self.llm_model = llm_model
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-generation", 
            model=self.llm_model, 
            device=device,
            token=""
        )

    def cluster_text(self, df, column, max_clusters=50):

        df['split_names'] = df[column].apply(self._preprocess_strings)
        processed_text = df['split_names'].tolist()
        
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        X = vectorizer.fit_transform(processed_text)

        cluster_num_array = range(2, max_clusters + 1)
        best_silhouette = -1
        best_num_clusters = 0

        for num_clusters in cluster_num_array:
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_

            silhouette_avg = metrics.silhouette_score(X, labels)

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_num_clusters = num_clusters

        kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_

        print(f"Number of clusters: {best_num_clusters} with silhouette score: {best_silhouette}")


        cluster_df = pd.DataFrame({
            'original': df[column],
            'text': processed_text,
            'cluster': labels
        })

        cluster_summaries = {}
        for c in cluster_df['cluster'].unique():
            cluster_texts = cluster_df[cluster_df['cluster'] == c]['text'].tolist()
            sample_texts = "; ".join(cluster_texts[:50])

            prompt = (
                f"These function names belong to one cluster:\n{sample_texts}\n\n"
                f"Write a very short one-sentence summary describing the content of this cluster."
            )

            response = self.pipe(prompt, max_new_tokens=30, temperature=0.3)
            summary = response[0]["generated_text"].replace(prompt, "").strip()
            cluster_summaries[c] = summary

        cluster_df['cluster_summary'] = cluster_df['cluster'].map(cluster_summaries)

        return cluster_df
    
    def _preprocess_strings(self, name):
        
         # Ha pont van benne, bontsuk külön részekre
        parts = name.split('.')
        split_parts = []

        for part in parts:
            # Snake_case -> szóköz
            part = part.replace('_', ' ')
            # camelCase vagy PascalCase -> szóköz a kisbetű és nagybetű közt
            part = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', part)
            # Nagybetűk közé kisbetű -> pl. HTMLResponse -> HTML Response
            part = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', part)
            split_parts.append(part)

        # Egyesítsük, kisbetűsre konvertálva
        return ' '.join(split_parts).lower().strip()
        