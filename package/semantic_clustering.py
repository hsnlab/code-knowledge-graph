from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
import numpy as np
import networkx as nx
import re
from cdlib import algorithms
from scipy.optimize import linear_sum_assignment
import re


class SemanticClustering():

    def __init__(self, hugging_face_token=None, llm_model="mistralai/mistral-7b-instruct-v0.3"):

        self.llm_model = llm_model
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model,
            token=hugging_face_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation", 
            model=self.llm_model, 
            device=device,
            tokenizer=tokenizer,
            token=hugging_face_token
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

        prompts = []
        for c in cluster_df['cluster'].unique():
            cluster_texts = cluster_df[cluster_df['cluster'] == c]['text'].tolist()
            sample_texts = "; ".join(cluster_texts[:50])

            prompt = (
                f"These function names belong to one cluster:\n{sample_texts}\n\n"
                f"Write a very short one-sentence summary describing the content of this cluster."
            )
            prompts.append((c, prompt))

        # csak a promptokat küldjük a modellnek, egyszerre
        responses = self.pipe(
            [p for _, p in prompts],
            max_new_tokens=50,
            temperature=0.3,
            batch_size=16
        )

        # hozzárendeljük a válaszokat a clusterekhez
        cluster_summaries = {}
        for (c, prompt), resp in zip(prompts, responses):
            summary = resp[0]["generated_text"].replace(prompt, "").strip()
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
    
    def create_graph_from_dfs(self, edges_df, nodes_df, number_of_edges=None):
        """Load a NetworkX graph from CSV edge and node dataframes."""
        # Map func_id -> combinedName (labels for visualization)
        label_dict = pd.Series(nodes_df["combinedName"].values, index=nodes_df["func_id"]).to_dict()

        # Limit edges if requested (useful for debugging)
        if number_of_edges is not None:
            edges_df = edges_df.head(number_of_edges)

        # Build edge list
        edges = list(zip(edges_df['source_id'], edges_df['target_id']))
        
        # Create undirected graph
        G = nx.Graph()
        G.add_edges_from(edges)

        # Attach human-readable labels to nodes
        for node in G.nodes:
            if node in label_dict:
                G.nodes[node]['label'] = label_dict[node]

        return G
        
    def apply_methods(self, edges_df, nodes_df):
            """Apply multiple community detection algorithms from CDlib."""
            G = self.create_graph_from_dfs(edges_df, nodes_df)
            
            # Run multiple clustering/community detection methods
            algos = {
                "louvain": algorithms.louvain(G, resolution=1., randomize=False),
                "surprise": algorithms.surprise_communities(G),
                "leiden": algorithms.leiden(G),
                "walktrap": algorithms.walktrap(G)
            }

            results = {}
            for name, partition in algos.items():
                data = []
                # Convert each partition into a DataFrame: node_id -> cluster assignment
                for com_id, community in enumerate(partition.communities):
                    for n in community:
                        label = G.nodes[n].get('label', str(n))
                        data.append({'node_id': n, 'label': label, 'cluster': com_id})
                df = pd.DataFrame(data).rename(columns={"cluster": f"cluster_{name}"})
                results[name] = df.drop_duplicates(subset=["node_id"])
                
            return results
            
            
    def ensemble(self, df_sem, df_algos, project="default"):
        """Ensemble clustering:
        - Align labels across methods
        - Build co-association matrix
        - Cluster on agreement matrix with AgglomerativeClustering
        """
        df = df_sem.copy()
        
        # Merge semantic clusters with graph-based algorithm outputs
        for name, df_algo in df_algos.items():
            df = pd.merge(df, df_algo[["node_id", f"cluster_{name}"]], on="node_id", how="inner")

        df = df.drop_duplicates(subset=["node_id"]).reset_index(drop=True)

        # Align predicted labels to semantic clusters using Hungarian algorithm
        def align_labels(true_labels, pred_labels):
            contingency = pd.crosstab(true_labels, pred_labels)
            row_ind, col_ind = linear_sum_assignment(-contingency.values)
            mapping = {contingency.columns[c]: contingency.index[r] for r, c in zip(row_ind, col_ind)}
            return pred_labels.map(mapping)

        # Align each graph-based algorithm to semantic clusters
        for name in df_algos.keys():
            df[f"cluster_{name}_mapped"] = align_labels(df["cluster"], df[f"cluster_{name}"])

        # Build co-association matrix:
        #   co_matrix[i,j] = fraction of methods where nodes i and j are in same cluster
        N = len(df)
        label_sources = [df["cluster"]] + [df[f"cluster_{name}_mapped"] for name in df_algos.keys()]

        co_matrix = np.zeros((N, N), dtype=float)
        for labels in label_sources:
            arr = labels.to_numpy()
            # Broadcasting: (N,1) == (1,N) → boolean matrix of shape (N,N)
            mask = (arr[:, None] == arr[None, :]).astype(float)
            co_matrix += mask

        co_matrix /= len(label_sources)  # normalize by number of methods

        # Run AgglomerativeClustering on the similarity matrix (1 - co_matrix = distance)
        desired_k = len(np.unique(df["cluster"]))  # match number of semantic clusters
        ensemble = AgglomerativeClustering(
            n_clusters=desired_k,
            metric="precomputed",
            linkage="average"
        )
        final_labels = ensemble.fit_predict(1 - co_matrix)

        df["ensemble_cluster"] = final_labels
        return df
        
        
    def agreement_graph(self, df_sem, df_algos, threshold=0.5, weighted=False):
        """Build an agreement graph from semantic + algorithmic clusterings.

        Nodes are taken from the semantic dataframe (df_sem with 'node_id' and 'cluster').
        Edges are created between node pairs whose co-association (fraction of methods that put
        them in the same cluster) meets the threshold.

        :param df_sem: DataFrame returned by cluster_text() with columns ['node_id','cluster']
        :param df_algos: dict of DataFrames returned by apply_methods() with keys like 'louvain'
        :param threshold: Fraction threshold in [0,1] to include an edge (default 0.5).
        :param weighted: If True, include weight column with co-association value for every pair (no thresholding).
        :return: edges_df (DataFrame with columns ['source_id','target_id','weight']).
        """
        # Merge semantic clusters with graph-based algorithm outputs (same logic as ensemble)
        df = df_sem.copy()
        for name, df_algo in df_algos.items():
            df = pd.merge(df, df_algo[["node_id", f"cluster_{name}"]], on="node_id", how="inner")
        df = df.drop_duplicates(subset=["node_id"]).reset_index(drop=True)

        # Align predicted labels to semantic clusters using Hungarian algorithm
        def align_labels(true_labels, pred_labels):
            contingency = pd.crosstab(true_labels, pred_labels)
            row_ind, col_ind = linear_sum_assignment(-contingency.values)
            mapping = {contingency.columns[c]: contingency.index[r] for r, c in zip(row_ind, col_ind)}
            return pred_labels.map(mapping)

        for name in df_algos.keys():
            df[f"cluster_{name}_mapped"] = align_labels(df["cluster"], df[f"cluster_{name}"])

        # Build co-association matrix
        N = len(df)
        if N == 0:
            return pd.DataFrame(columns=['source_id','target_id','weight'])
        co_matrix = np.zeros((N, N))
        label_sources = [df["cluster"]] + [df[f"cluster_{name}_mapped"] for name in df_algos.keys()]
        for labels in label_sources:
            arr = labels.values
            for i in range(N):
                for j in range(N):
                    if arr[i] == arr[j]:
                        co_matrix[i, j] += 1
        co_matrix /= len(label_sources)

        node_ids = df['node_id'].values
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                weight = float(co_matrix[i, j])
                if weighted:
                    edges.append({'source_id': int(node_ids[i]), 'target_id': int(node_ids[j]), 'weight': weight})
                else:
                    if weight >= threshold:
                        edges.append({'source_id': int(node_ids[i]), 'target_id': int(node_ids[j]), 'weight': weight})
        edges_df = pd.DataFrame(edges)
        return edges_df

