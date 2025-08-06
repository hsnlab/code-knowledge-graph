import pandas as pd
import torch
import networkx as nx
from pyvis.network import Network

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
sys.path.append("./package")
from retrieval import Retrieval  # Assuming you saved the new Retrieval class in retrieval.py


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
                function_ids = reranked_functions['func_id'].tolist()

                print("\nBuilding enriched knowledge graph...")
                subgraph = self._filter_knowledge_graph(function_ids)

                print("\nGenerating answer...")
                answer = self._generate_answer(question, subgraph)
                print("\nAnswer:", answer)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting search. Goodbye!")


    def retrieve_code_functions(self, question, top_n=5):
        df = self.data_dict['cg_nodes']
        results = self.retriever.retrieve(question, df, column_name='summary', threshold=top_n)
        merged = results.merge(df, on='summary', how='left')
        return merged

    def retrieve_issues(self, question, top_n=5):
        df = self.data_dict['issues'].copy()
        df['combined'] = df['issue_title'].fillna('') + " " + df['issue_body'].fillna('')
        results = self.retriever.retrieve(question, df, column_name='combined', threshold=top_n)
        merged = results.merge(df, on='combined', how='left')
        return merged

    def retrieve_prs(self, question, top_n=5):
        df = self.data_dict['prs']
        results = self.retriever.retrieve(question, df, column_name='pr_title', threshold=top_n)
        merged = results.merge(df, on='pr_title', how='left')
        return merged

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
        
        issue_funcs = issue_links['func_id'].tolist()

        # 2. From pr_edges
        pr_edges = data['pr_edges'].copy()
        pr_edges = pr_edges[pr_edges['pr_number'].isin(pr_nums)]

        pr_sim = prs_df[['pr_number', 'similarity']]
        pr_edges = pr_edges.merge(pr_sim, on='pr_number', how='left')
        pr_funcs = pd.concat([pr_edges['func_id_1'], pr_edges['func_id_2']]).tolist()

        # 3. From same cluster
        func_cluster_map = data['cg_nodes'].set_index('func_id')['cluster_id'].to_dict()
        base_clusters = set(func_cluster_map[fid] for fid in base_funcs if fid in func_cluster_map)
        cluster_funcs = [fid for fid, cid in func_cluster_map.items() if cid in base_clusters]

        # Union all function candidates
        all_func_ids = set(base_funcs) | set(issue_funcs) | set(pr_funcs) | set(cluster_funcs)

        # Score initialization
        scores = {}
        for fid in all_func_ids:
            scores[fid] = 0.0

        # Add base similarity scores
        for _, row in functions_df.iterrows():
            scores[row['func_id']] += row.get('similarity', 0.0)

        # Add issue links
        for fid in issue_funcs:
            scores[fid] += issue_links[issue_links['func_id'] == fid]['similarity'].sum()

        # Add PR links
        for fid in pr_funcs:
            scores[fid] += pr_edges[(pr_edges['func_id_1'] == fid) | (pr_edges['func_id_2'] == fid)]['similarity'].sum()

        # Add cluster membership
        for fid in cluster_funcs:
            scores[fid] += 0.5  # Arbitrary score for cluster membership

        # Create new ranked dataframe
        all_nodes = data['cg_nodes']
        scored_df = all_nodes[all_nodes['func_id'].isin(all_func_ids)].copy()
        scored_df['relevance_score'] = scored_df['func_id'].map(scores)
        scored_df = scored_df.sort_values(by='relevance_score', ascending=False).head(top_n)

        return scored_df


    def _filter_knowledge_graph(self, function_ids):
        data = self.data_dict
        G = nx.Graph()

        cgn = data['cg_nodes']
        cge = data['cg_edges']
        sgn = data['sg_nodes']
        sge = data['sg_edges']
        h1e = data['hier_1']
        h2e = data['hier_2']
        issues = data['issues']
        issue_edges = data['issue_to_pr_function_edges']
        pr_edges = data['pr_edges']
        prs = data['prs']

        # Filter relevant functions
        cgn = cgn[cgn['func_id'].isin(function_ids)].copy()

        # Call graph
        func_ids = set(cgn['func_id'])
        cge = cge[cge['source_id'].isin(func_ids) & cge['target_id'].isin(func_ids)]

        # Structure graph
        sgn = sgn[sgn['func_id'].isin(func_ids)]
        sge = sge[sge['source_id'].isin(sgn['node_id']) & sge['target_id'].isin(sgn['node_id'])]

        # Hierarchy
        h1e = h1e[h1e['target_id'].isin(func_ids)]
        h2e = h2e[h2e['source_id'].isin(func_ids)]

        # Issues
        issue_edges = issue_edges[issue_edges['func_id'].isin(func_ids)]
        relevant_issues = issue_edges['issue_number'].unique()
        issues = issues[issues['issue_number'].isin(relevant_issues)]

        # PRs
        relevant_prs = issue_edges['pr_number'].unique()
        prs = prs[prs['pr_number'].isin(relevant_prs)]
        pr_edges = pr_edges[pr_edges['pr_number'].isin(relevant_prs)]

        # Add function nodes
        for _, row in cgn.iterrows():
            node_id = f"F_{row['func_id']}"
            label = f"[F] {row['combinedName']}"
            G.add_node(node_id, label=label, title=row['summary'], color="#1f78b4")

        # Add issue nodes
        for _, row in issues.iterrows():
            node_id = f"I_{row['issue_number']}"
            label = f"[I] {row['issue_title'][:40]}"
            G.add_node(node_id, label=label, title=row['issue_body'], color="#e31a1c")

        # Add PR edges as dashed blue
        for _, row in pr_edges.iterrows():
            src = f"F_{row['func_id_1']}"
            tgt = f"F_{row['func_id_2']}"
            label = f"[PR] {row['pr_number']}"
            if src != tgt:
                G.add_edge(src, tgt, color="#1f78b4", dashes=True, label=label, title=row['pr_title'])

        # Add issue → function edges
        for _, row in issue_edges.iterrows():
            issue_node = f"I_{row['issue_number']}"
            func_node = f"F_{row['func_id']}"
            label = f"[PR] {row['pr_number']}"
            G.add_edge(issue_node, func_node, color="#fb9a99", dashes=True, label=label, title=row['pr_title'])

        # Add call graph edges
        for _, row in cge.iterrows():
            src = f"F_{row['source_id']}"
            tgt = f"F_{row['target_id']}"
            G.add_edge(src, tgt, color="#a6cee3", title=row['call'])
        # Visualize
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph('./filtered_graph.html')
        print("Graph saved to ./filtered_graph.html")

        return G


    def _generate_answer(self, question: str, subgraph: nx.Graph):
        """
        Generate an answer to the user's question based on the filtered subgraph.
        Includes enriched context: functions, issues, PRs, and edge summaries.
        """
        function_nodes = []
        issue_nodes = []
        pr_datas = set()

        # Parse nodes
        for node_id, data in subgraph.nodes(data=True):
            label = data.get("label", node_id)
            

            if node_id.startswith("F_"):
                func_id = node_id[2:]
                function_nodes.append(f"{label}(func_id: {func_id})")
            elif node_id.startswith("I_"):
                issue_num = node_id[2:]
                body = data.get("title", "")
                issue_nodes.append(f"issue: {issue_num}, title: {label}, body: {body}")

        # Parse edges
        pr_edge_summary = []
        call_edge_summary = []

        for u, v, data in subgraph.edges(data=True):
            if data.get("dashes", True):
                pr_label = data.get("label", "")
                pr_title = data.get("title", "")
                pr_datas.add(f"pr: {pr_label}, title: {pr_title}")
                pr_edge_summary.append(
                    f"{pr_number}: connects {u}, {v} | {issue_str}"
                )

            elif u.startswith("F_") and v.startswith("F_"):
                label = data.get("label", "No label")
                call_edge_summary.append(f"{u},{v}")

        # Create context blocks
        context_parts = []

        if function_nodes:
            context_parts.append("### Relevant Functions\n" + ";".join(function_nodes))

        if issue_nodes:
            context_parts.append("### Related Issues\n" + ";".join(issue_nodes))

        if pr_titles:
            context_parts.append("### Pull Requests (from edges)\n" + ";".join(list(pr_titles)))

        if pr_edge_summary:
            context_parts.append("### PR-Based Function Links\n" + ";".join(pr_edge_summary))

        if call_edge_summary:
            context_parts.append("### Function Call Edges\n" + ";".join(call_edge_summary))

        # Combine context
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
            max_new_tokens=512,
            return_full_text=False,
        )
        return response[0]['generated_text'].strip()

