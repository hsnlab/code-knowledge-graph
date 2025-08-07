import pandas as pd
import torch
import networkx as nx
from pyvis.network import Network

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
sys.path.append("../package")
from retrieval import Retrieval


class RepositoryRAG():
    def __init__(self, data_dict: dict, model_name: str = 'all-MiniLM-L6-v2', llm_model: str = "mistralai/mistral-7b-instruct-v0.3"):
        """
        Initialize the RepositoryRAG class with in-memory data and models.

        Args:
            data_dict (dict): Dictionary of DataFrames representing repo data.
            model_name (str): Name of the SentenceTransformer model.
            llm_model (str): Hugging Face model for LLM-based answer generation.
        """
        self.data_dict = data_dict
        self.retriever = Retrieval(model_name)

        # LLM initialization
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model).to(device)
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def search(self, top_n: int = 10):
        try:
            while True:
                question = input("\nPlease enter your question (Ctrl+C to exit): ").strip()
                if not question:
                    continue

                print("\nRetrieving top results...")
                functions_df = self.retrieve_code_functions(question, top_n=top_n)
                issues_df = self.retrieve_issues(question, top_n=top_n)
                prs_df = self.retrieve_prs(question, top_n=top_n)

                reranked_functions = self.rerank_functions(functions_df, issues_df, prs_df, top_n=top_n)
                
                print("\nBuilding enriched knowledge graph...")
                subgraph = self._filter_knowledge_graph(reranked_functions, issues_df, prs_df)

                print("\nGenerating answer...")
                answer = self._generate_answer(question, subgraph)
                print("\nAnswer:", answer)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting search. Goodbye!")


    def retrieve_code_functions(self, question, top_n=5):
        df = self.data_dict['cg_nodes']
        results = self.retriever.retrieve(question, df, column_name='combinedName', threshold=top_n)
        #merged = results.merge(df, on='combinedName', how='left')
        return results

    def retrieve_issues(self, question, top_n=5):
        df = self.data_dict['issues'].copy()
        df['problem_statement'] = df['issue_title'].fillna('') + " " + df['issue_body'].fillna('')
        results = self.retriever.retrieve(question, df, column_name='problem_statement', threshold=top_n)
        #merged = results.merge(df, on='problem_statement', how='left')
        return results

    def retrieve_prs(self, question, top_n=5):
        df = self.data_dict['prs']
        results = self.retriever.retrieve(question, df, column_name='pr_title', threshold=top_n)
        #merged = results.merge(df, on='pr_title', how='left')
        return results

    def rerank_functions(self, functions_df, issues_df, prs_df, top_n=10):
        """
        Expand and rerank functions based on:
        - PR and issue links
        - Cluster membership
        """
        data = self.data_dict

        # Get base set
        base_funcs = set(functions_df['func_id'].tolist())
        issue_nums = set(issues_df['issue_number'].tolist())
        pr_nums = set(prs_df['pr_number'].tolist())

        # 1. From issue_to_pr_function_edges
        issue_links = data['issue_to_pr_function_edges'].copy()
        issue_links = issue_links[issue_links['issue_number'].isin(issue_nums)]
        
        issue_sim = issues_df[['issue_number', 'similarity']]
        issue_links = issue_links.merge(issue_sim, on='issue_number', how='left')
        issue_links['similarity'] = issue_links['similarity'].fillna(0.0)
        
        issue_funcs = issue_links['func_id'].tolist()

        # 2. From pr_edges
        pr_edges = data['pr_edges'].copy()
        pr_edges = pr_edges[pr_edges['pr_number'].isin(pr_nums)]

        pr_sim = prs_df[['pr_number', 'similarity']]
        pr_edges = pr_edges.merge(pr_sim, on='pr_number', how='left')
        pr_edges['similarity'] = pr_edges['similarity'].fillna(0.0)
        pr_func_pairs = pd.concat([
            pr_edges[['pr_number', 'func_id_1']].rename(columns={'func_id_1': 'func_id'}),
            pr_edges[['pr_number', 'func_id_2']].rename(columns={'func_id_2': 'func_id'})
        ])

        # Drop duplicates
        pr_func_pairs = pr_func_pairs.drop_duplicates().reset_index(drop=True)
        pr_funcs = pr_func_pairs["func_id"].tolist()

        # 3. From same cluster
        func_cluster_map = data['cg_nodes'].set_index('func_id')['cluster_id'].to_dict()
        base_clusters = set(func_cluster_map[fid] for fid in base_funcs if fid in func_cluster_map)
        cluster_funcs = [fid for fid, cid in func_cluster_map.items() if cid in base_clusters]

        # Union all function candidates
        all_func_ids = set(base_funcs) | set(issue_funcs) | set(pr_funcs) | set(cluster_funcs)

        # Score initialization
        scores = {}
        for fid in all_func_ids:
            scores[int(fid)] = 0.0

        # Add base similarity scores
        for _, row in functions_df.iterrows():
            val = row.get('similarity', 0.0)
            scores[int(row['func_id'])] += val

        # Add issue links
        for fid in issue_funcs:
            val = issue_links[issue_links['func_id'] == fid]['similarity'].values[0]
            scores[int(fid)] += val

        for fid in set(pr_funcs):  # or use np.unique(pr_funcs) if it's a NumPy array
            val = pr_edges[(pr_edges['func_id_1'] == fid) | (pr_edges['func_id_2'] == fid)].drop_duplicates().reset_index(drop=True)['similarity'].values[0]
            scores[int(fid)] += val

        # Add cluster membership
        for fid in cluster_funcs:
            scores[fid] += 0.15  # Arbitrary score for cluster membership

        # Create new ranked dataframe
        all_nodes = data['cg_nodes']
        scored_df = all_nodes[all_nodes['func_id'].isin(all_func_ids)].copy()
        scored_df['relevance_score'] = scored_df['func_id'].map(scores)      
        scored_df = scored_df.sort_values(by='relevance_score', ascending=False).head(top_n)

        return scored_df


    def _filter_knowledge_graph(self, functions_df, issues_df, prs_df):
        data = self.data_dict
        

        # --- Inputs ---
        func_ids = set(functions_df['func_id'])
        issue_nums = set(issues_df['issue_number'])
        pr_nums = set(prs_df['pr_number'])

        # --- Filter nodes ---
        cgn = data['cg_nodes']
        cge = data['cg_edges']
        sgn = data['sg_nodes']
        sge = data['sg_edges']
        h1e = data['hier_1']
        h2e = data['hier_1']
        issues = data['issues']
        prs = data['prs']
        issue_edges = data['issue_to_pr_function_edges']
        pr_edges = data['pr_edges']

        # Filter function nodes based on top functions
        cgn = cgn[cgn['func_id'].isin(func_ids)].reset_index(drop=True)
        cge = cge[cge['source_id'].isin(cgn['func_id'].tolist()) & cge['target_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)

        sgn = sgn[sgn['func_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)
        sge = sge[sge['source_id'].isin(sgn['node_id'].tolist()) & sge['target_id'].isin(sgn['node_id'].tolist())].reset_index(drop=True)

        h1e = h1e[h1e['source_id'].isin(sgn['node_id'].tolist()) & h1e['target_id'].isin(cgn['func_id'].tolist())].reset_index(drop=True)
        h2e = h2e[h2e['source_id'].isin(cgn['func_id'].tolist()) & h2e['target_id'].isin(sgn['node_id'].tolist())].reset_index(drop=True)

        # Filter issue nodes
        issues = issues[issues['issue_number'].isin(issue_nums)].reset_index(drop=True)

        # Filter PRs and PR edges
        prs = prs[prs['pr_number'].isin(pr_nums)].copy()
        pr_edges = pr_edges[pr_edges['pr_number'].isin(prs['pr_number'].tolist())].reset_index(drop=True)


        # Filter issue → function links (only those involving current funcs and issues)
        issue_edges = issue_edges[issue_edges['func_id'].isin(cgn['func_id'].tolist()) & issue_edges['issue_number'].isin(issues['issue_number'].tolist())].reset_index(drop=True)


        G = nx.Graph()
        # Add call graph nodes (functions)
        for _, row in cgn.iterrows():
            node_id = f"F_{row['func_id']}"
            label = f"[F] {row['combinedName']}"
            G.add_node(node_id, label=label, title=label, color="#1f78b4")  # blue

        # Add structure graph nodes
        for _, row in sgn.iterrows():
            node_id = f"S_{row['node_id']}"
            label = f"[S] {row['code'][:25]}..." if len(row['code']) > 25 else f"[S] {row['code']}"
            G.add_node(node_id, label=label, title=label, color="#33a02c")  # green

        # Add call edges
        for _, row in cge.iterrows():
            source = f"F_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#a6cee3")  # light blue

        # Add structure edges
        for _, row in sge.iterrows():
            source = f"S_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#b2df8a")  # light green

        # Add hierarchy edges (function -> structure)
        for _, row in h1e.iterrows():
            source = f"S_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#ff7f00")

        for _, row in h2e.iterrows():
            source = f"F_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#ff7f00")

        # --- Add issue nodes ---
        for _, row in issues.iterrows():
            node_id = f"I_{row['issue_number']}"
            label = f"[I] {row['issue_title'][:40]}"
            G.add_node(node_id, label=label, title=row['issue_body'], color="#e31a1c")

        # --- Add PR edges (function–function) ---
        for _, row in pr_edges.iterrows():
            src = f"F_{row['func_id_1']}"
            tgt = f"F_{row['func_id_2']}"
            label = f"[PR] {row['pr_number']}"
            G.add_edge(src, tgt, color="#1f78b4", dashes=True, label=label, title=row['pr_title'])

        # --- Add issue → function edges via PR ---
        for _, row in issue_edges.iterrows():
            issue_node = f"I_{row['issue_number']}"
            func_node = f"F_{row['func_id']}"
            label = f"[PR] {row['pr_number']}"
            pr_row = prs[prs['pr_number'] == row['pr_number']]
            pr_title = pr_row['pr_title'].values[0] if not pr_row.empty else ""
            G.add_edge(issue_node, func_node, color="#fb9a99", dashes=True, label=label, title=pr_title)


        # --- Visualization (optional) ---
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph('./filtered_graph.html')
        print("Graph saved to ./filtered_graph.html")

        return G


    def _generate_answer(self, question: str, subgraph: nx.Graph, max_items_per_section:int=10):
        """
        Generate an answer to the user's question based on the filtered subgraph.
        Includes enriched context: functions, issues, PRs, and edge summaries.
        """
        def truncate(text, max_len=80):
            return text[:max_len] + "..." if len(text) > max_len else text
    
        function_nodes = set()
        issue_nodes = set()
        pr_datas = set()
        pr_edge_summary = set()
        call_edge_summary = set()
    
        # --- Parse nodes ---
        for node_id, data in subgraph.nodes(data=True):
            label = truncate(data.get("label", node_id))
            if node_id.startswith("F_"):
                func_id = node_id[2:]
                function_nodes.add(f"{label} (func_id: {func_id})")
            elif node_id.startswith("I_"):
                issue_num = node_id[2:]
                body = truncate(data.get("title", ""))
                issue_nodes.add(f"Issue #{issue_num}: {label} | {body}")
    
        # --- Parse edges ---
        for u, v, data in subgraph.edges(data=True):
            if data.get("dashes", True):
                pr_number = data.get("label", "")
                pr_title = truncate(data.get("title", ""))
                pr_datas.add(f"PR #{pr_number}: {pr_title}")
                pr_edge_summary.add(f"PR #{pr_number}: connects {u} ↔ {v}")
            elif u.startswith("F_") and v.startswith("F_"):
                call_edge_summary.add(f"{u} ↔ {v}")
    
        # --- Helper to limit output ---
        def limit(section):
            return sorted(section)[:max_items_per_section]
    
        # --- Build context blocks ---
        context_parts = []
    
        if function_nodes:
            context_parts.append("### Relevant Functions\n" + "\n".join(limit(function_nodes)))
        if issue_nodes:
            context_parts.append("### Related Issues\n" + "\n".join(limit(issue_nodes)))
        if pr_datas:
            context_parts.append("### Pull Requests\n" + "\n".join(limit(pr_datas)))
        if pr_edge_summary:
            context_parts.append("### PR-Based Function Links\n" + "\n".join(limit(pr_edge_summary)))
        if call_edge_summary:
            context_parts.append("### Function Call Edges\n" + "\n".join(limit(call_edge_summary)))
    
        context = "\n\n".join(context_parts)

        # Prompt
        prompt = f"""<s>[INST] You are a helpful machine learning assistant.

Use the context below to answer the question about this software repository.
You are given function nodes, issue nodes, PR titles, and edges between them.
Use the node/edge IDs when relevant.

If you're unsure, say so.

### Question
{question}

{context}

### Answer
[/INST]"""

        # Generate output
        response = self.generation_pipeline(
            prompt,
            max_new_tokens=100,
            return_full_text=False,
        )
        return response[0]['generated_text'].strip()

if __name__ == "__main__":
    sklearn_hier_json = pd.read_pickle("../graph/sklearn/sklearn.pkl")
    tool = RepositoryRAG(data_dict=sklearn_hier_json)
    tool.search(top_n=10)