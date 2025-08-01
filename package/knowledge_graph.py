from github import Github

import pandas as pd
import networkx as nx
from pyvis.network import Network
from .hierarchical_graph import HierarchicalGraphBuilder


class KnowledgeGraphBuilder():

    git = None
    repository = None

    def __init__(self, github_token=None):
        self.git = Github(github_token) if github_token else Github()



    def build_knowledge_graph(self, repo_path, repo_name, graph_type="CFG", num_of_PRs=5):
        """
        Builds a knowledge graph from the given repository path.
        
        :param repo_path: Path to the repository.
        :param repo_name: Name of the repository. Must match the format "owner/repo_name", as it is used for github API calls.
        :param graph_type: Type of subgraph to build. Can be "CFG" (Control Flow Graph) or "AST" (Abstract Syntax Tree). Default is "CFG".
        :param num_of_PRs: Number of pull requests to retrieve in detail. Defaults to 5. 0 means all PRs.
        :return: A dictionary containing nodes, edges, imports, and other parts of the hierarchical graph.
        """

        self.repository = self.git.get_repo(repo_name)

        issues = self.__get_repo_issues(self.repository)
        pulls = self.__get_repo_PRs(self.repository, num_of_PRs=num_of_PRs)
        artifacts = self.__get_repo_CI_artifacts(self.repository)

        hg = HierarchicalGraphBuilder()
        cg_nodes, cg_edges, sg_nodes, sg_edges, hier_1, hier_2, imports = hg.create_hierarchical_graph(repo_path, graph_type=graph_type)

        pulls = self.__set_PR_file_ids(pulls, cg_nodes)

        pr_edges = self.__create_PR_edges(pulls, cg_nodes)

        imports, imp_edges = self.__create_import_edges(imports, cg_nodes)

        self.knowledge_graph = {
            "cg_nodes": cg_nodes,
            "cg_edges": cg_edges,
            "sg_nodes": sg_nodes,
            "sg_edges": sg_edges,
            "hier_1": hier_1,
            "hier_2": hier_2,
            "imports": imports,
            "imp_edges": imp_edges,
            "issues": issues,
            "prs": pulls,
            "pr_edges": pr_edges,
            "artifacts": artifacts
        }
        
        return self.knowledge_graph



    def visualize_graph(self, knowledge_graph=None):

        if knowledge_graph is None:
            knowledge_graph = self.knowledge_graph

        G = nx.Graph()

        # ---------------- Nodes ----------------

        # Add call graph nodes (functions)
        for _, row in knowledge_graph['cg_nodes'].iterrows():
            node_id = f"F_{row['func_id']}"
            label = f"[F] {row['combinedName']}"
            G.add_node(node_id, label=label, title=label, color="#1a70aa")  # blue

        # Add structure graph nodes
        for _, row in knowledge_graph['sg_nodes'].iterrows():
            node_id = f"S_{row['node_id']}"
            label = f"[S] {row['code'][:25]}..." if len(row['code']) > 25 else f"[S] {row['code']}"
            G.add_node(node_id, label=label, title=label, color="#33a02c")  # green

        # Add import nodes
        for _, row in knowledge_graph['imports'].iterrows():
            node_id = f"S_{row['import_id']}"
            label = f"[I] {row['import_name']} as {row['import_as_name']}"
            G.add_node(node_id, label=label, title=label, color="#b12ad3")  # green

        # Add PR nodes
        for _, row in knowledge_graph['prs'].iterrows():
            node_id = f"PR_{row['pr_number']}"
            label = f"[PR] {row['pr_title'][:25]}..." if len(row['pr_title']) > 25 else f"[PR] {row['pr_title']}"
            G.add_node(node_id, label=label, title=label, color="#e31a1c")


        # ---------------- Edges ----------------

        # Add call edges
        for _, row in knowledge_graph['cg_edges'].iterrows():
            source = f"F_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#a6cee3")  # light blue

        # Add structure edges
        for _, row in knowledge_graph['sg_edges'].iterrows():
            source = f"S_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#b2df8a")  # light green

        # Add hierarchy edges (function -> structure)
        for _, row in knowledge_graph['hier_1'].iterrows():
            source = f"S_{row['source_id']}"
            target = f"F_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#21c795", dashes=True)

        for _, row in knowledge_graph['hier_2'].iterrows():
            source = f"F_{row['source_id']}"
            target = f"S_{row['target_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#21c795", dashes=True)

        # Add import edges
        for _, row in knowledge_graph['imp_edges'].iterrows():
            source = f"I_{row['import_id']}"
            target = f"F_{row['func_id'].astype(int)}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#bd75cf", dashes=True)

        # Add PR edges
        for _, row in knowledge_graph['pr_edges'].iterrows():
            source = f"PR_{row['pr_number']}"
            target = f"F_{row['func_id']}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#e07375", dashes=True)

        # Visualize with pyvis
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph('./filtered_graph.html')
        print(f"Filtered sklearn graph. visualization saved to ./filtered_graph.html")
        return G





    def __get_repo_issues(self, repo):
        """
        Retrieves issues from the repository.
        
        :param repo: The repository object.
        :return: A list of issues.
        """
        issues = repo.get_issues(state='open', sort='created', direction='desc', labels=["bug"])

        data = []

        for issue in issues:
            issue_number = issue.number
            issue_title = issue.title
            issue_body = issue.body if issue.body else ""
            issue_labels = [label.name for label in issue.labels]
            issue_state = issue.state

            data.append({
                "issue_number": issue_number,
                "issue_title": issue_title,
                "issue_body": issue_body,
                "issue_labels": issue_labels,
                "issue_state": issue_state
            })

        issues_df = pd.DataFrame(data)
        return issues_df
    

    def __get_repo_PRs(self, repo, num_of_PRs):
        """
        Retrieves pull requests from the repository and saves details to a DataFrame.
        
        :param repo: The repository object.
        :param num_of_PRs: Number of pull requests to retrieve. Defaults to 5. 0 means all PRs.
        :return: DataFrame with PR file change details.
        """
        pulls = repo.get_pulls(state='open', sort='created', direction='desc')

        print("Pulls done")
        data = []

        num_pulls = 0

        for pull in pulls:
            print(f"Processing PR # {num_pulls}")
            pr = repo.get_pull(pull.number)
            pr_number = pr.number
            pr_title = pr.title

            files = pr.get_files()
            for f in files:
                data.append({
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "filename": f.filename,
                    "status": f.status,        # 'modified', 'added', 'removed', etc.
                    "additions": f.additions,
                    "deletions": f.deletions
                })
            
            num_pulls += 1
            if num_of_PRs > 0:
                if num_pulls >= num_of_PRs:  # Limit to the specified number of PRs
                    break

        df = pd.DataFrame(data)
        return df


    def __get_repo_CI_artifacts(self, repo):
        """
        Retrieves CI artifacts from the repository.
        
        :param repo: The repository object.
        :return: A list of CI artifacts.
        """
        artifacts = repo.get_artifacts()
        data = []

        for artifact in artifacts:
            data.append({
                "artifact_id": artifact.id,
                "artifact_name": artifact.name,
                "artifact_size": artifact.size_in_bytes,
                "created_at": artifact.created_at,
                "updated_at": artifact.updated_at
            })

        artifacts_df = pd.DataFrame(data)
        return artifacts_df


    def __set_PR_file_ids(self, pulls, nodes):
        """
        Maps file IDs to pull requests.
        
        :param pulls: DataFrame containing pull request details.
        :param nodes: DataFrame containing node details.
        :return: DataFrame with file IDs added to pull requests.
        """
        filepaths = pulls['filename'].dropna().drop_duplicates().tolist()
        file_ids = []
        for filepath in filepaths:
            temp = nodes.loc[nodes['function_location'].str.contains(filepath, na=False, case=False)]
            if not temp.empty:
                file_ids.append({
                    "filepath": filepath,
                    "file_id": temp.iloc[0]['file_id'],
                })
        
        pulls['file_id'] = pulls['filename'].map(lambda x: next((item['file_id'] for item in file_ids if item['filepath'] == x), None))
        return pulls


    def __create_PR_edges(self, pulls, cg_nodes):
        """
        Creates edges between pull requests and call graph nodes.
        
        :param pulls: DataFrame containing pull request details.
        :param cg_nodes: DataFrame containing call graph nodes.
        :return: DataFrame with PR edges.
        """
        pr_edges = pulls[['pr_number', 'pr_title', 'status', 'additions', 'deletions', 'file_id']].copy()
        pr_edges = pr_edges.loc[pr_edges['file_id'].notna()]
        pr_edges = pr_edges.merge(cg_nodes[['file_id', 'func_id']], on='file_id', how='left')[['pr_number', 'func_id', 'status', 'additions', 'deletions']]
        return pr_edges


    def __create_import_edges(self, import_df, cg_nodes):
        """
        Creates edges for imports in the graph.
        
        :param imports: DataFrame containing import details.
        :return: DataFrame with import edges.
        """
        imports_grouped = (
            import_df.groupby(['name', 'from', 'as_name'])['file_id']
            .apply(lambda x: list(set(x)))
            .reset_index()
        )

        imports_grouped.insert(0, 'import_id', range(1, len(imports_grouped)+1))

        imports = imports_grouped.rename(columns={'name': 'import_name', 'from': 'import_from', 'as_name': 'import_as_name', 'file_id': 'import_file_ids'})

        imp_edges = imports[['import_id', 'import_file_ids']].explode('import_file_ids')
        imp_edges = imp_edges.rename(columns={'import_file_ids': 'file_id'})

        imp_edges = imp_edges.merge(cg_nodes[['func_id', 'file_id']], on='file_id', how='left')
        imp_edges = imp_edges[['import_id', 'func_id']].dropna().reset_index(drop=True)

        return imports, imp_edges
