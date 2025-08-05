from github import Github

import pandas as pd
import networkx as nx
from pyvis.network import Network
from itertools import combinations
import yaml
import re

from .hierarchical_graph import HierarchicalGraphBuilder
from .pr_function_collector import extract_changed_functions_from_pr
from .semantic_clustering import SemanticClustering

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
        actions = self.__get_repo_actions()

        hg = HierarchicalGraphBuilder()
        cg_nodes, cg_edges, sg_nodes, sg_edges, hier_1, hier_2, imports = hg.create_hierarchical_graph(repo_path, graph_type=graph_type)

        pulls = self.__set_PR_file_ids(pulls, cg_nodes)

        pr_node_edges = self.__create_PR_node_edges(cg_nodes)

        pr_edges = self.__create_PR_edges(pr_node_edges)

        imports, imp_edges = self.__create_import_edges(imports, cg_nodes)

        # Issue to PR edges
        linked_issues = []

        for pr_number in pulls['pr_number'].drop_duplicates().tolist():
            linked_issue_nums = self.__get_linked_issues(pr_number)
            linked_issues.append({
                'pr_number': pr_number,
                'linked_issues': linked_issue_nums
            })

        issue_to_pr_edges = self.__validated_linked_issues_to_dataframe(
            linked_issues,
            issues['issue_number'].tolist()
        )

        issue_to_pr_function_edges = issue_to_pr_edges.merge(pr_node_edges[['pr_number', 'func_id']], on='pr_number', how='left')
        issue_to_pr_function_edges = issue_to_pr_function_edges.dropna().reset_index(drop=True)

        cluster_edges = self.__cluster_function_nodes(cg_nodes)

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
            "issue_to_pr_edges": issue_to_pr_edges,
            "issue_to_pr_function_edges": issue_to_pr_function_edges,
            "artifacts": artifacts,
            "actions": actions,
            "cluster_edges": cluster_edges
        }

        return self.knowledge_graph



    def visualize_graph(self, knowledge_graph=None, show_subgraph_nodes=False):

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
        if show_subgraph_nodes:
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
        for _, row in knowledge_graph['issues'].iterrows():
            node_id = f"ISSUE_{row['issue_number']}"
            label = f"[IS] {row['issue_title'][:25]}..." if len(row['issue_title']) > 25 else f"[IS] {row['issue_title']}"
            G.add_node(node_id, label=label, title=label, color="#e31a1c")


        # ---------------- Edges ----------------

        # Add call edges
        for _, row in knowledge_graph['cg_edges'].iterrows():
            source = f"F_{row['source_id'].astype(int)}"
            target = f"F_{row['target_id'].astype(int)}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#a6cee3")  # light blue

        if show_subgraph_nodes:
            # Add structure edges
            for _, row in knowledge_graph['sg_edges'].iterrows():
                source = f"S_{row['source_id'].astype(int)}"
                target = f"S_{row['target_id'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#b2df8a")  # light green

            # Add hierarchy edges (function -> structure)
            for _, row in knowledge_graph['hier_1'].iterrows():
                source = f"S_{row['source_id'].astype(int)}"
                target = f"F_{row['target_id'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#21c795", dashes=True)

            for _, row in knowledge_graph['hier_2'].iterrows():
                source = f"F_{row['source_id'].astype(int)}"
                target = f"S_{row['target_id'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#21c795", dashes=True)

        # Add import edges
        for _, row in knowledge_graph['imp_edges'].iterrows():
            source = f"I_{row['import_id'].astype(int)}"
            target = f"F_{row['func_id'].astype(int)}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#bd75cf", dashes=True)

        # Add PR edges
        for _, row in knowledge_graph['pr_edges'].iterrows():
            source = f"F_{int(round(row['func_id_1']))}"
            target = f"F_{int(round(row['func_id_2']))}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#02096b", dashes=True, label= f"PR#{row['pr_number']} - {row['pr_title'][:25]}...")

        # Add issue to PR function edges
        for _, row in knowledge_graph['issue_to_pr_function_edges'].iterrows():
            source = f"ISSUE_{int(row['issue_number'])}"
            target = f"F_{int(row['func_id'])}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#860000", dashes=True, label=f"Linked to PR#{row['pr_number']}")

        # Visualize with pyvis
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph('./filtered_graph.html')
        print(f"Filtered sklearn graph. visualization saved to ./filtered_graph.html")
        return G





    def __get_repo_issues(self, repo, labels=["bug"]):
        """
        Retrieves issues from the repository.
        
        :param repo: The repository object.
        :return: A list of issues.
        """
        open_issues = repo.get_issues(state='open', sort='created', direction='desc', labels=labels)

        data = []

        for issue in open_issues:
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

        open_issues_df = pd.DataFrame(data)

        closed_issues = repo.get_issues(state='closed', sort='created', direction='desc', labels=["bug"])

        data = []

        for issue in closed_issues:
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

        closed_issues_df = pd.DataFrame(data)

        issues_df = pd.concat([open_issues_df, closed_issues_df], ignore_index=True)

        return issues_df
    

    def __get_repo_PRs(self, repo, num_of_PRs):
        """
        Retrieves pull requests from the repository and saves details to a DataFrame.
        
        :param repo: The repository object.
        :param num_of_PRs: Number of pull requests to retrieve. Defaults to 5. 0 means all PRs.
        :return: DataFrame with PR file change details.
        """

        # ---------------- Open Pull Requests ----------------

        pulls = repo.get_pulls(state='open', sort='created', direction='desc')
        changed_functions = []

        print("Pulls done")
        data = []

        num_pulls = 0

        for pull in pulls:
            print(f"Processing PR # {num_pulls}")
            pr = repo.get_pull(pull.number)
            pr_number = pr.number
            pr_title = pr.title

            try:
                changed = extract_changed_functions_from_pr(pr)
                for fn in changed:
                    file_path, class_and_function = fn.split(":", 1) if fn else ("", "")
                    changed_functions.append({
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_open": True,
                        "file_path": file_path,
                        "class_and_function": class_and_function
                    })
            except:
                print("Warning: Could not extract changed functions from PR #", pr_number)
                continue

            files = pr.get_files()
            for f in files:
                data.append({
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "pr_open": True,
                    "filename": f.filename,
                    "status": f.status,        # 'modified', 'added', 'removed', etc.
                    "additions": f.additions,
                    "deletions": f.deletions
                })
            
            num_pulls += 1
            if num_of_PRs > 0:
                if num_pulls >= num_of_PRs:  # Limit to the specified number of PRs
                    break

        df_open = pd.DataFrame(data)




        # ---------------- Closed Pull Requests ----------------

        pulls_closed = repo.get_pulls(state='closed', sort='created', direction='desc')

        print("Closed pulls done")
        data = []

        num_pulls = 0

        for pull in pulls_closed:
            print(f"Processing PR # {num_pulls}")
            pr = repo.get_pull(pull.number)
            pr_number = pr.number
            pr_title = pr.title

            try:
                changed = extract_changed_functions_from_pr(pr)
                for fn in changed:
                    file_path, class_and_function = fn.split(":", 1) if fn else ("", "")
                    changed_functions.append({
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_open": True,
                        "file_path": file_path,
                        "class_and_function": class_and_function
                    })
            except:
                print("Warning: Could not extract changed functions from PR #", pr_number)
                continue

            files = pr.get_files()
            for f in files:
                data.append({
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "pr_open": False,
                    "filename": f.filename,
                    "status": f.status,        # 'modified', 'added', 'removed', etc.
                    "additions": f.additions,
                    "deletions": f.deletions
                })
            
            num_pulls += 1
            if num_of_PRs > 0:
                if num_pulls >= num_of_PRs:  # Limit to the specified number of PRs
                    break

        df_closed = pd.DataFrame(data)

        # Combine open and closed PRs
        df = pd.concat([df_open, df_closed], ignore_index=True).reset_index(drop=True)

        self.changed_functions = pd.DataFrame(changed_functions)
        
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
    

    def __get_repo_actions(self):
        
        workflow_data = []

        for wf in self.repository.get_workflows():
            try:
                content = self.repository.get_contents(wf.path)
                yaml_text = content.decoded_content.decode()
                wf_yaml = yaml.safe_load(yaml_text)

                name = wf.name
                path = wf.path
                trigger = list(wf_yaml.get("on", {}).keys()) if isinstance(wf_yaml.get("on"), dict) else wf_yaml.get("on")
                jobs = wf_yaml.get("jobs", {})

                used_actions = set()
                platforms = set()

                for job_id, job_data in jobs.items():
                    if isinstance(job_data, dict):
                        runs_on = job_data.get("runs-on")
                        if runs_on:
                            if isinstance(runs_on, list):
                                platforms.update(runs_on)
                            else:
                                platforms.add(runs_on)

                        steps = job_data.get("steps", [])
                        for step in steps:
                            if isinstance(step, dict):
                                uses = step.get("uses")
                                if uses:
                                    used_actions.add(uses)

                workflow_data.append({
                    "name": name,
                    "path": path,
                    "triggers": trigger,
                    "platforms": list(platforms),
                    "actions_used": list(used_actions)
                })

            except Exception as e:
                print(f"Hiba a(z) {wf.name} feldolgozásakor: {e}")

        actions_df = pd.DataFrame(workflow_data)

        return actions_df


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


    def __create_PR_node_edges(self, cg_nodes):
        """
        Creates edges between pull requests and call graph nodes.
        
        :param pulls: DataFrame containing pull request details.
        :param cg_nodes: DataFrame containing call graph nodes.
        :return: DataFrame with PR edges.
        """
        temp_nodes = cg_nodes[['function_location', 'combinedName', 'func_id']].drop_duplicates()
        temp_nodes['function_location'] = temp_nodes['function_location'].str.replace('./sklearn/', '', regex=False)
        temp_changed_functions = self.changed_functions.copy()
        temp_changed_functions = temp_changed_functions.merge(temp_nodes, left_on='class_and_function', right_on='combinedName', how='left')

        pr_edges = temp_changed_functions[['pr_number', 'func_id', 'pr_title', 'pr_open', 'file_path']].copy().dropna()
        return pr_edges


    def __create_PR_edges(self, pr_node_edges):

        # Groupby pr_number, majd generáljuk a func_id párosításokat
        pair_results = []

        for pr_number, group in pr_node_edges.groupby('pr_number'):
            func_ids = group['func_id'].tolist()
            
            if len(func_ids) == 1:
                # Csak egy func_id van, tehát az önmagával párban van
                pair_results.append((pr_number, func_ids[0], func_ids[0]))
            else:
                # Minden lehetséges párosítás
                for pair in combinations(func_ids, 2):
                    pair_results.append((pr_number, pair[0], pair[1]))
                    pair_results.append((pr_number, pair[1], pair[0]))

        # Az eredmény egy új DataFrame
        pairs_df = pd.DataFrame(pair_results, columns=['pr_number', 'func_id_1', 'func_id_2'])

        pairs_df = pairs_df.merge(pr_node_edges[['pr_number', 'pr_title']], on='pr_number', how='left')

        return pairs_df
    

    def __create_PR_edges_files(self, pulls, cg_nodes):
        """
        Creates edges between pull requests and call graph nodes.
        
        :param pulls: DataFrame containing pull request details.
        :param cg_nodes: DataFrame containing call graph nodes.
        :return: DataFrame with PR edges.
        """
        pr_edges = pulls[['pr_number', 'pr_title', 'pr_open', 'status', 'additions', 'deletions', 'file_id']].copy()
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


    def __cluster_function_nodes(self, cg_nodes):
        """
        Clusters function nodes based on their names using semantic clustering.
        
        :param cg_nodes: DataFrame containing call graph nodes.
        :return: DataFrame with clustered function nodes.
        """
        sc = SemanticClustering()
        cluster_df = sc.cluster_text(cg_nodes, 'combinedName', max_clusters=50)

        # Create edges for the clusters
        cluster_df = cluster_df.merge(cg_nodes[['func_id', 'combinedName']], left_on='original', right_on='combinedName', how='left')

        return cluster_df[['cluster', 'func_id']]


    def __extract_issue_references(self, text):
        """
        Extract issue numbers from a text like #123
        """
        return list(set(map(int, re.findall(r'#(\d+)', text or ""))))

    def __get_linked_issues(self, pr_number):
        """
        Get all issues mentioned in the pull request body or comments
        """
        pr = self.repository.get_pull(pr_number)
        linked_issue_numbers = set()

        # PR leírásban keres issue hivatkozásokat
        linked_issue_numbers.update(self.__extract_issue_references(pr.body))

        # PR kommentjeiben is keres
        for comment in pr.get_issue_comments():
            linked_issue_numbers.update(self.__extract_issue_references(comment.body))
        
        return list(linked_issue_numbers)

    def __validated_linked_issues_to_dataframe(self, linked_issues_data, existing_issue_numbers):
        existing_set = set(existing_issue_numbers)
        records = []

        for item in linked_issues_data:
            pr_number = item['pr_number']

            for issue_number in item['linked_issues']:
                if issue_number in existing_set:
                    records.append({
                        'issue_number': issue_number,
                        'pr_number': pr_number,
                    })

        return pd.DataFrame(records)
