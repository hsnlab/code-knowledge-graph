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
            print(resp)
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

    def apply_clustering_methods(self, edges_df, nodes_df):
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
            
    def _merge_and_align(self, df_sem, df_algos):
        """Common helper: merge semantic clusters with algo outputs and align labels."""
        df = df_sem.copy()
        for name, df_algo in df_algos.items():
            df = pd.merge(
                df,
                df_algo[["node_id", f"cluster_{name}"]],
                left_on="func_id",
                right_on="node_id",
                how="inner",
                suffixes=("", f"_{name}")
            )
            df = df.drop(columns=["node_id"])

        def align_labels(true_labels, pred_labels):
            contingency = pd.crosstab(true_labels, pred_labels)
            row_ind, col_ind = linear_sum_assignment(-contingency.values)
            mapping = {contingency.columns[c]: contingency.index[r] for r, c in zip(row_ind, col_ind)}
            return pred_labels.map(mapping)

        for name in df_algos.keys():
            df[f"cluster_{name}_mapped"] = align_labels(df["cluster"], df[f"cluster_{name}"])
        return df


    def _build_co_matrix(self, df, df_algos):
        """Common helper: build normalized co-association matrix."""
        N = len(df)
        label_sources = [df["cluster"]] + [df[f"cluster_{name}_mapped"] for name in df_algos.keys()]
        co_matrix = np.zeros((N, N), dtype=float)
        for labels in label_sources:
            arr = labels.to_numpy()
            mask = (arr[:, None] == arr[None, :]).astype(float)
            co_matrix += mask
        return co_matrix / len(label_sources)
    
    
    def ensemble(self, df_algos, df_sem, project="default"):
        df = self._merge_and_align(df_sem, df_algos)
        N = len(df)
        co_matrix = self._build_co_matrix(df, df_algos)

        desired_k = len(np.unique(df["cluster"]))
        ensemble = AgglomerativeClustering(
            n_clusters=desired_k, metric="precomputed", linkage="average"
        )
        final_labels = ensemble.fit_predict(1 - co_matrix)
        df["ensemble_cluster"] = final_labels
        return df


    def agreement_graph(self, df_sem, df_algos, threshold=0.5, weighted=False):
        df = self._merge_and_align(df_sem, df_algos)
        N = len(df)
        if N == 0:
            return pd.DataFrame(columns=['source_id','target_id','weight'])
        co_matrix = self._build_co_matrix(df, df_algos)

        node_ids = df['node_id'].values if 'node_id' in df else df['func_id'].values
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                weight = float(co_matrix[i, j])
                if weighted or weight >= threshold:
                    edges.append({
                        'source_id': int(node_ids[i]),
                        'target_id': int(node_ids[j]),
                        'weight': weight
                    })
        return pd.DataFrame(edges)

   

