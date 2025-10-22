from github import Github

import pandas as pd
import networkx as nx
from pyvis.network import Network
from itertools import combinations
import yaml
import re
import sys, os

from neo4j import GraphDatabase
from git import Repo

from .hierarchical_graph import HierarchicalGraphBuilder
from .semantic_clustering import SemanticClustering
from .pr_function_collector import extract_changed_functions_from_pr

from package.adapters import LanguageAstAdapterRegistry


class KnowledgeGraphBuilder():

    git = None
    repository = None

    pr_function_data = []
    pr_file_data = []

    def __init__(self, github_token=None, hugging_face_token=None):
        self.git = Github(github_token) if github_token else Github()
        self.hugging_face_token = hugging_face_token

    def build_knowledge_graph(
        self, 
        repo_name: str, 
        graph_type: str = "CFG", 
        num_of_PRs: int = 0, 
        done_prs: list = None,
        create_embedding: bool = False, 
        repo_path_modifier: str = None,
        URI: str = None,
        user: str = None,
        password: str = None,
        project_language: str | None = None,
        developer_mode: str | None = None,  # None (default) to preserve behavior; or 'commit_authors', 'pr_authors', 'contributors'
        max_commits: int = 200  # used when developer_mode == 'commit_authors'
    ):
        """
        Builds a knowledge graph from the given repository.
        
        :param repo_name: Name of the repository. Must match the format "owner/repo_name", as it is used for github API calls.
        :param graph_type (optional): Type of subgraph to build. Can be "CFG" (Control Flow Graph) or "AST" (Abstract Syntax Tree). Default is "CFG".
        :param num_of_PRs (optional): Number of pull requests to retrieve in detail. Defaults to 0 (all).
        :param create_embedding (optional): Whether to create embeddings for the nodes. Defaults to False.
        :param repo_path_modifier (optional): Path modifier for the repository for cases when only a subfolder is meant to be parsed.
        :param URI (optional): URI for the Neo4J data saving.
        :param user (optional): Username for the Neo4J data saving.
        :param password (optional): Password for the Neo4J data saving.
        :param developer_mode (optional): If provided, extracts developer nodes and edges.
               Options: 'commit_authors' (map commits -> files -> functions),
                        'pr_authors' (map PR authors -> changed functions),
                        'contributors' (contributors only; no edges unless PR mapping is available).
        :param max_commits (optional): Max commits scanned when developer_mode='commit_authors'.
        :return: By defult, it returns a dictionary containing nodes, edges, imports, and other parts of the hierarchical graph. If the URI, user and password data is given, it saves it into a Neo4J database.
        """

        # Clone repository
        repo_path = self.__clone_github_repo(repo_name)
        if repo_path_modifier:
            repo_path = repo_path + '/' + repo_path_modifier if repo_path_modifier[0] != '/' else repo_path + repo_path_modifier
            repo_path = repo_path if repo_path[-1] == '/' else repo_path + '/'

        # Initialize repository
        self.repository = self.git.get_repo(repo_name)

        # Build hierarchical graph (CG + CFG/AST)        
        hg = HierarchicalGraphBuilder()
        (
            cg_nodes,
            cg_edges,
            sg_nodes,
            sg_edges,
            hier_1,
            hier_2,
            imports,
            function_version_nodes,
            version_edges,
            functionversion_function_edges,
        ) = hg.create_hierarchical_graph(
            repo_path,
            graph_type=graph_type,
            create_embedding=create_embedding,
            project_language=project_language
        )

        
        # Get repository issues, pull requests, artifacts and actions
        if create_embedding:
            cluster_nodes, cluster_edges = self.__cluster_function_nodes(cg_nodes)
            print('Function nodes clustered.')
        else:
            # Skip clustering
            cluster_nodes = pd.DataFrame(columns=['ID', 'summary'])
            cluster_edges = pd.DataFrame(columns=['source', 'target'])
            print('Clustering skipped (create_embedding=False).')
        """
        print('Skipping GitHub data (issues, PRs, artifacts, actions).')
        issues = pd.DataFrame(columns=['ID', 'issue_title', 'issue_body', 'issue_labels', 'issue_state'])
        prs = pd.DataFrame(columns=['ID', 'pr_title', 'pr_body', 'pr_open'])
        pr_edges = pd.DataFrame(columns=['source', 'target'])
        artifacts = pd.DataFrame(columns=['ID', 'artifact_name', 'artifact_size', 'created_at', 'updated_at'])
        actions = pd.DataFrame(columns=['name', 'path', 'triggers', 'platforms', 'actions_used'])
        issue_to_pr_edges = pd.DataFrame(columns=['source', 'target'])   
        """ 
        issues = self.__get_repo_issues(self.repository)
        print('Issues scraped.')
        prs, pr_edges = self.__get_repo_PRs(self.repository, cg_nodes, num_of_PRs=num_of_PRs, done_prs=done_prs)
        print('PRs scraped.')
        artifacts = self.__get_repo_CI_artifacts(self.repository)
        print('Artifacts scraped.')
        actions = self.__get_repo_actions()
        print('Actions scraped.')
       
       
        issue_to_pr_edges = self.__get_issue_to_pr_edges(issues, prs)
        print('Issue to PR edges created.')
        
        developers_df = pd.DataFrame(columns=['ID', 'dev_name', 'dev_email', 'dev_full'])
        dev_edges_df = pd.DataFrame(columns=['source', 'target'])  # developer -> function

        if developer_mode:
            try:
                if developer_mode == 'commit_authors':
                    dev_nodes, dev_edges_raw = self.__get_developers_from_commits(self.repository, cg_nodes, max_commits=max_commits)
                    # normalize columns
                    if not dev_nodes.empty:
                        dev_nodes = dev_nodes.rename(columns={'dev_id': 'ID'})
                        dev_nodes['ID'] = dev_nodes['ID'].astype(int)
                    if not dev_edges_raw.empty:
                        dev_edges_df = dev_edges_raw.rename(columns={'dev_id': 'source', 'func_id': 'target'})
                elif developer_mode == 'pr_authors':
                    dev_nodes, dev_edges_raw = self.__get_repo_developers_github(self.repository, prs, pr_edges)
                    if not dev_nodes.empty:
                        dev_nodes = dev_nodes.rename(columns={'dev_id': 'ID'})
                        dev_nodes['ID'] = dev_nodes['ID'].astype(int)
                    if not dev_edges_raw.empty:
                        dev_edges_df = dev_edges_raw.rename(columns={'dev_id': 'source', 'func_id': 'target'})
                elif developer_mode == 'contributors':
                    dev_nodes, dev_edges_raw = self.__get_repo_developers_github(self.repository, None, None)
                    if not dev_nodes.empty:
                        dev_nodes = dev_nodes.rename(columns={'dev_id': 'ID'})
                        dev_nodes['ID'] = dev_nodes['ID'].astype(int)
                    # no edges unless PR mapping was available
                else:
                    print(f"Unknown developer_mode '{developer_mode}', skipping developer extraction.")
                    dev_nodes = pd.DataFrame()
                    dev_edges_raw = pd.DataFrame()
                developers_df = dev_nodes if not dev_nodes.empty else developers_df
                if not dev_edges_df.empty and 'commit_sha' in dev_edges_df.columns:
                    # keep commit_sha as property but not required; leave as-is
                    pass
            except Exception as e:
                print(f"Developer extraction failed ({developer_mode}): {e}")


        # todo remove cpp specific import_edge_creation after demo
        if project_language == "cpp":
            adapter_class = LanguageAstAdapterRegistry.get_adapter(project_language)
            adapter = adapter_class()
            imports, imp_edges = adapter.create_import_edges(imports, cg_nodes)
        else:
            imports, imp_edges = self.__create_import_edges(imports, cg_nodes)
        cg_nodes, cg_edges, sg_nodes, sg_edges, imports, imp_edges, hier_1, hier_2 = self.__format_dfs(cg_nodes, cg_edges, sg_nodes, sg_edges, imports, imp_edges, hier_1, hier_2)

        question_nodes, question_edges = self.__create_question_nodes(cluster_nodes)

        self.knowledge_graph = {
            "function_nodes": cg_nodes,
            "function_edges": cg_edges,
            "subgraph_nodes": sg_nodes,
            "subgraph_edges": sg_edges,
            "subgraph_function_edges": hier_1,
            "function_subgraph_edges": hier_2,
            "import_nodes": imports,
            "import_function_edges": imp_edges,
            "pr_nodes": prs,
            "pr_function_edges": pr_edges,
            "issue_nodes": issues,
            "issue_pr_edges": issue_to_pr_edges,
            "artifacts": artifacts,
            "actions": actions,
            "cluster_nodes": cluster_nodes,
            "cluster_function_edges": cluster_edges,
            "functionversion_nodes": function_version_nodes,
            "functionversion_edges": version_edges,
            "functionversion_function_edges": functionversion_function_edges,
            "developer_node": developers_df,
            "developer_function_edges": dev_edges_df,
            "question_nodes": question_nodes,
            "question_cluster_edges": question_edges,
        }
        
        if developer_mode and not developers_df.empty:
            self.knowledge_graph["developer_nodes"] = developers_df

        if developer_mode and not dev_edges_df.empty:
            # Ensure correct dtype
            for col in ('source', 'target'):
                if col in dev_edges_df.columns:
                    dev_edges_df[col] = pd.to_numeric(dev_edges_df[col], errors='coerce').astype('Int64')
            dev_edges_df = dev_edges_df.dropna(subset=['source', 'target']).astype({'source':'int','target':'int'}).reset_index(drop=True)
            self.knowledge_graph["developer_function_edges"] = dev_edges_df

        if URI and user and password:
            self.store_knowledge_graph_in_neo4j(URI, user, password, self.knowledge_graph)

        else:
            return self.knowledge_graph



    def visualize_graph(self, knowledge_graph=None, show_subgraph_nodes=False, save_path="./graph.html"):
        """
        Create a HTML visualizaiton of the graph with the `visualize_graph` function. NOTE: for large graphs, it is advised to only plot a fraction of the nodes, othervise the visualization might not render properly. Parameters:
        :param repograph: The dictionary containing the created repository graph.
        :param show_subgraph_nodes (optional): Whether to plot the subgraph (CFG or AST) nodes. Defaults to False.
        :param save_path (optional): The file path to save the visualization. Defaults to "filtered_graph.html".
        """


        if knowledge_graph is None:
            knowledge_graph = self.knowledge_graph

        G = nx.Graph()

        # ---------------- Nodes ----------------

        # Add call graph nodes (functions)
        for _, row in knowledge_graph['function_nodes'].iterrows():
            node_id = f"F_{row['ID']}"
            label = f"[F] {row['combinedName']}"
            G.add_node(node_id, label=label, title=label, color="#1a70aa")  # blue

        # Add structure graph nodes
        if show_subgraph_nodes:
            for _, row in knowledge_graph['subgraph_nodes'].iterrows():
                node_id = f"S_{row['ID']}"
                label = f"[S] {row['code'][:25]}..." if len(row['code']) > 25 else f"[S] {row['code']}"
                G.add_node(node_id, label=label, title=label, color="#33a02c")  # green

        # Add import nodes
        for _, row in knowledge_graph['import_nodes'].iterrows():
            node_id = f"I_{row['ID']}"
            label = f"[I] {row['import_name']} as {row['import_as_name']}"
            G.add_node(node_id, label=label, title=label, color="#b12ad3")  # green

        # Add PR nodes
        for _, row in knowledge_graph['issue_nodes'].iterrows():
            node_id = f"ISSUE_{row['ID']}"
            label = f"[IS] {row['issue_title'][:25]}..." if len(row['issue_title']) > 25 else f"[IS] {row['issue_title']}"
            G.add_node(node_id, label=label, title=label, color="#e31a1c")

         # Add developer nodes (NEW)
        if 'developer_nodes' in knowledge_graph:
            for _, row in knowledge_graph['developer_nodes'].iterrows():
                node_id = f"DEV_{row['ID']}"
                label = f"[DEV] {row.get('dev_name', row['ID'])}"
                G.add_node(node_id, label=label, title=label, color="#ff9800")  # orange

        # ---------------- Edges ----------------

        # Add call edges
        for _, row in knowledge_graph['function_edges'].iterrows():
            source = f"F_{row['source'].astype(int)}"
            target = f"F_{row['target'].astype(int)}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#a6cee3")  # light blue

        if show_subgraph_nodes:
            # Add structure edges
            for _, row in knowledge_graph['subgraph_edges'].iterrows():
                source = f"S_{row['source'].astype(int)}"
                target = f"S_{row['target'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#b2df8a")  # light green

            # Add hierarchy edges (function -> structure)
            for _, row in knowledge_graph['subgraph_function_edges'].iterrows():
                source = f"S_{row['source'].astype(int)}"
                target = f"F_{row['target'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#21c795", dashes=True)

            for _, row in knowledge_graph['function_subgraph_edges'].iterrows():
                source = f"F_{row['source'].astype(int)}"
                target = f"S_{row['target'].astype(int)}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#21c795", dashes=True)

        # Add import edges
        for _, row in knowledge_graph['import_function_edges'].iterrows():
            source = f"I_{row['source'].astype(int)}"
            target = f"F_{row['target'].astype(int)}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#bd75cf", dashes=True)

        # Add PR edges
        for _, row in knowledge_graph['pr_function_edges'].iterrows():
            source = f"F_{int(round(row['sourrce']))}"
            target = f"F_{int(round(row['target']))}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#02096b", dashes=True, label= f"PR#{row['pr_number']} - {row['pr_title'][:25]}...")

        # Add issue to PR function edges
        for _, row in knowledge_graph['issue_pr_edges'].iterrows():
            source = f"ISSUE_{int(row['source'])}"
            target = f"F_{int(row['target'])}"
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, color="#860000", dashes=True, label=f"Linked to PR#{row['pr_number']}")
        
        
        if 'developer_function_edges' in knowledge_graph:
            for _, row in knowledge_graph['developer_function_edges'].iterrows():
                source = f"DEV_{int(row['source'])}"
                target = f"F_{int(row['target'])}"
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, color="#ff9800", dashes=True, title="Developer → Function")

        # Visualize with pyvis
        net = Network(height='1000px', width='100%', notebook=False, directed=False)
        net.from_nx(G)
        net.force_atlas_2based()
        net.save_graph(f'./{save_path}')
        print(f"Filtered sklearn graph. visualization saved to {save_path}")
        return G

    

    def store_knowledge_graph_in_neo4j(self, uri, user, password, knowledge_graph):
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

            # Load nodes with global id
            for key, df in knowledge_graph.items():
                if key.endswith("_nodes"):
                    label = key.replace("_nodes", "").upper()

                    if "id" in df.columns:
                        df["id"] = df["id"].astype(int)
                    elif "ID" in df.columns:
                        df["ID"] = df["ID"].astype(int)

                    for _, row in df.iterrows():
                        props = {}
                        for k, v in row.items():
                            # Csak NaN float-ot szűrjük ki, minden más marad
                            if not (isinstance(v, float) and pd.isna(v)):
                                props[k] = v

                        local_id = props.get("id") or props.get("ID")
                        if local_id is None:
                            raise ValueError(f"A(z) {label} node-nak nincs id mezője.")

                        props["global_id"] = f"{label}:{local_id}"

                        session.run(
                            f"CREATE (n:{label} $props)",
                            props=props
                        )

            # Load edges
            for key, df in knowledge_graph.items():
                if key.endswith("_edges"):
                    if df.empty or 'source' not in df.columns or 'target' not in df.columns:
                        continue
                    rel_type = key.replace("_edges", "").upper()

                    # Determine source and target node labels based on the key
                    parts = key.replace("_edges", "").split("_")
                    if len(parts) == 1:
                        src_label = tgt_label = parts[0].upper()
                    elif len(parts) == 2:
                        src_label, tgt_label = parts[0].upper(), parts[1].upper()
                    else:
                        raise ValueError(f"Nem tudom értelmezni az edge nevet: {key}")

                    df["source"] = df["source"].astype(int)
                    df["target"] = df["target"].astype(int)

                    for _, row in df.iterrows():
                        start_id = f"{src_label}:{row['source']}"
                        end_id = f"{tgt_label}:{row['target']}"
                        edge_props = {k: v for k, v in row.items() if k not in ["source", "target"] and pd.notna(v)}

                        session.run(
                            f"""
                            MATCH (a:{src_label} {{global_id: $start_id}}),
                                (b:{tgt_label} {{global_id: $end_id}})
                            CREATE (a)-[r:{rel_type} $props]->(b)
                            """,
                            start_id=start_id,
                            end_id=end_id,
                            props=edge_props
                        )

        driver.close()



    # ---------------------------------------------------------------
    #                     Private Methods
    # ---------------------------------------------------------------

    def __clone_github_repo(self, repo_identifier: str, target_dir: str = "./repos"):
        """
        Clones a GitHub repository.
        :param repo_identifier: The repository identifier (e.g., "torvalds/linux").
        :param target_dir: The directory where the repository will be cloned.
        :return: The local path of the cloned repository.
        """

        # Repo URL összeállítása
        url = f"https://github.com/{repo_identifier}.git"

        # Lokális könyvtár előkészítése
        repo_name = repo_identifier.split("/")[-1]
        local_path = os.path.join(target_dir, repo_name)

        os.makedirs(target_dir, exist_ok=True)

        if os.path.exists(local_path):
            print(f"Repo already exists here: {local_path}")
        else:
            print(f"Cloning: {url} -> {local_path}")
            Repo.clone_from(url, local_path)

        return local_path





    def __get_repo_issues(self, repo, labels=["bug"]):
        """
        Retrieves issues from the repository.
        
        :param repo: The repository object.
        :return: A list of issues.
        """

        # Get open issues
        open_issues_df = self.__get_issue_from_git(repo, labels, issue_state='open')

        # Get closed issues
        closed_issues_df = self.__get_issue_from_git(repo, labels, issue_state='closed')

        issues_df = pd.concat([open_issues_df, closed_issues_df], ignore_index=True)
        return issues_df
    

    def __get_issue_from_git(self, repo, labels, issue_state):

        issues = repo.get_issues(state=issue_state, sort='created', direction='desc', labels=labels)

        data = []

        for issue in issues:
            issue_number = issue.number
            issue_title = issue.title
            issue_body = issue.body if issue.body else ""
            issue_labels = [label.name for label in issue.labels]
            issue_state = issue.state

            data.append({
                "ID": issue_number,
                "issue_title": issue_title,
                "issue_body": issue_body,
                "issue_labels": issue_labels,
                "issue_state": issue_state
            })

        issues_df = pd.DataFrame(data)

        return issues_df





    def __get_repo_PRs(self, repo, cg_nodes, num_of_PRs, done_prs):
        """
        Retrieves pull requests from the repository and saves details to a DataFrame.
        
        :param repo: The repository object.
        :param num_of_PRs: Number of pull requests to retrieve. Defaults to 5. 0 means all PRs.
        :return: DataFrame with PR file change details.
        """

        # ---------------- Open Pull Requests ----------------

        df_open, changed_functions_df_open = self.__get_PR_from_git(repo, num_of_PRs, 'open', done_prs)

        # ---------------- Closed Pull Requests ----------------

        df_closed, changed_functions_df_closed = self.__get_PR_from_git(repo, num_of_PRs, 'closed', done_prs)


        PR_df = pd.concat([df_open, df_closed], ignore_index=True).reset_index(drop=True)
        changed_functions_df = pd.concat([changed_functions_df_open, changed_functions_df_closed], ignore_index=True).reset_index(drop=True)

        # handle repositories without pr-s
        if changed_functions_df.empty:
            PR_df = PR_df.rename(columns={'pr_number': 'ID'})
            PR_df = PR_df[['ID', 'pr_title', 'pr_body', 'pr_open']].drop_duplicates().reset_index(drop=True) if not PR_df.empty else pd.DataFrame(columns=['ID', 'pr_title', 'pr_body', 'pr_open'])
            return PR_df, pd.DataFrame(columns=['source', 'target'])

        # String formatting
        changed_functions_df['file_path'] = changed_functions_df['file_path'].str.replace(r'\\', '/', regex=True)
        cg_nodes['function_location'] = (
            cg_nodes['function_location']
            .str.replace(r'\\', '/', regex=True)
            .str.replace('./repos/', '', regex=False)
            .str.split('/')
            .str[1:]
            .str.join('/')
        )
        changed_functions_df = changed_functions_df.merge(cg_nodes[['func_id', 'combinedName', 'function_location']], left_on=['file_path','class_and_function'], right_on=['function_location', 'combinedName'], how='left')
        changed_functions_df = changed_functions_df[['pr_number', 'func_id']].dropna().reset_index(drop=True).rename(columns={'pr_number': 'source', 'func_id': 'target'})

        PR_df = PR_df.rename(columns={
            'pr_number': 'ID'
        })
        PR_df = PR_df[['ID', 'pr_title', 'pr_body', 'pr_open']].drop_duplicates().reset_index(drop=True)

        return PR_df, changed_functions_df
    

    def __get_PR_from_git(self, repo, num_of_PRs: int, PR_state: str, done_prs: list):

        pulls = repo.get_pulls(state=PR_state, sort='created', direction='desc')

        changed_functions = []
        data = []

        num_pulls = 0

        for pull in pulls:

            if done_prs and pull.number in done_prs:
                continue

            pr = repo.get_pull(pull.number)
            pr_number = pr.number
            pr_title = pr.title
            pr_body = pr.body if pr.body else ""

            try:
                changed = extract_changed_functions_from_pr(pr)
                for fn in changed:
                    file_path, class_and_function = fn.split(":", 1) if fn else ("", "")
                    changed_functions.append({
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_body": pr_body,
                        "pr_open": PR_state == 'open',
                        "file_path": file_path,
                        "class_and_function": class_and_function
                    })
                    self.pr_function_data.append({
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_body": pr_body,
                        "pr_open": PR_state == 'open',
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
                    "pr_body": pr_body,
                    "pr_open": PR_state == 'open',
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions
                })
                self.pr_file_data.append({
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "pr_body": pr_body,
                    "pr_open": PR_state == 'open',
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions
                })
            
            num_pulls += 1
            if num_of_PRs > 0:
                if num_pulls >= num_of_PRs:
                    break

        PR_df = pd.DataFrame(data)
        changed_functions_df = pd.DataFrame(changed_functions)

        return PR_df, changed_functions_df





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
                "ID": artifact.id,
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
        imp_edges = imp_edges[['import_id', 'func_id']].dropna().reset_index(drop=True).rename(columns={'import_id': 'source', 'func_id': 'target'})

        return imports, imp_edges





    def __cluster_function_nodes(self, cg_nodes):
        """
        Clusters function nodes based on their names using semantic clustering.
        
        :param cg_nodes: DataFrame containing call graph nodes.
        :return: DataFrame with clustered function nodes.
        """
        sc = SemanticClustering(hugging_face_token=self.hugging_face_token)
        cluster_df = sc.cluster_text(cg_nodes, 'combinedName', max_clusters=50)

        # Create edges for the clusters
        cluster_df = cluster_df.merge(cg_nodes[['func_id', 'combinedName']], left_on='original', right_on='combinedName', how='left')

        cluster_nodes = cluster_df[['cluster', 'cluster_summary']].drop_duplicates().rename(columns={'cluster': 'ID', 'cluster_summary': 'summary'})
        cluster_nodes['ID'] = cluster_nodes['ID'].astype(int) + 1
        cluster_edges = cluster_df[['cluster', 'func_id']].drop_duplicates().rename(columns={'cluster': 'source', 'func_id': 'target'})

        return cluster_nodes, cluster_edges





    def __get_issue_to_pr_edges(self, issue_df, pr_df):
        
        if issue_df.empty or 'ID' not in issue_df.columns or pr_df.empty or 'ID' not in pr_df.columns:
                return pd.DataFrame(columns=['source', 'target'])

        # Issue to PR edges
        linked_issues = []

        for pr_number in pr_df['ID'].drop_duplicates().tolist():
            linked_issue_nums = self.__get_linked_issues(pr_number)
            linked_issues.append({
                'pr_number': pr_number,
                'linked_issues': linked_issue_nums
            })

        issue_to_pr_edges = self.__validated_linked_issues_to_dataframe(
            linked_issues,
            issue_df['ID'].tolist()
        )

        issue_to_pr_edges = issue_to_pr_edges.rename(columns={
            'issue_number': 'source',
            'pr_number': 'target'
        })

        return issue_to_pr_edges

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

        return pd.DataFrame(records, columns=['issue_number', 'pr_number'])



    def __get_repo_developers_github(self, repo, pr_nodes=None, pr_function_edges=None):
        """Retrieve contributors and optionally map PR authors to changed functions.
        Returns: (developers_df with dev_id,dev_name,dev_email,dev_full,
                dev_edges_df with dev_id,func_id[,pr_number])
        """
        data = []
        dev_id_counter = 1
        login_to_dev_id = {}

        # Collect contributors
        try:
            for contributor in repo.get_contributors():
                dev_id = dev_id_counter
                login_to_dev_id[contributor.login] = dev_id
                data.append({
                    'dev_id': dev_id,
                    'dev_name': contributor.login,
                    'dev_email': getattr(contributor, 'email', '') or '',
                    'dev_full': contributor.name if contributor.name else contributor.login
                })
                dev_id_counter += 1
        except Exception as e:
            print(f"Failed to fetch contributors: {e}")

        developers_df = pd.DataFrame(data) if data else pd.DataFrame(columns=['dev_id', 'dev_name', 'dev_email', 'dev_full'])

        # If PR + function mapping provided, create edges: PR author -> function(s)
        dev_edges_df = pd.DataFrame(columns=['dev_id', 'func_id', 'pr_number'])
        if pr_nodes is not None and pr_function_edges is not None and not pr_nodes.empty and not pr_function_edges.empty:
            # Build PR -> author map
            pr_author = {}
            try:
                for pr_id in pr_nodes['ID'].dropna().astype(int).unique().tolist():
                    try:
                        pr = repo.get_pull(int(pr_id))
                        login = pr.user.login if pr.user else None
                        if login:
                            if login not in login_to_dev_id:
                                login_to_dev_id[login] = dev_id_counter
                                developers_df = pd.concat([developers_df, pd.DataFrame([{
                                    'dev_id': dev_id_counter,
                                    'dev_name': login,
                                    'dev_email': getattr(pr.user, 'email', '') or '',
                                    'dev_full': getattr(pr.user, 'name', login) or login
                                }])], ignore_index=True)
                                dev_id_counter += 1
                            pr_author[int(pr_id)] = login_to_dev_id[login]
                    except Exception as pe:
                        print(f"Warning: could not fetch PR#{pr_id}: {pe}")
            except Exception as e:
                print(f"Failed to map PR authors: {e}")

            # Join PR->func with PR->dev
            tmp = pr_function_edges[['source', 'target']].dropna().copy()
            if not tmp.empty:
                tmp['dev_id'] = tmp['source'].astype(int).map(pr_author)
                tmp = tmp[tmp['dev_id'].notna()]
                tmp = tmp.rename(columns={'target': 'func_id', 'source': 'pr_number'})
                dev_edges_df = tmp[['dev_id', 'func_id', 'pr_number']].drop_duplicates().reset_index(drop=True)

        return developers_df, dev_edges_df

    def __get_developers_from_commits(self, repo, cg_nodes, max_commits=None):
        """Collect developers and function edges from individual commit authors.
        :param repo: GitHub repo
        :param cg_nodes: DataFrame of function nodes (must include file_id, func_id, function_location)
        :param max_commits: Limit number of commits scanned (None = all)
        :return: developers_df (dev_id,...), dev_edges_df (dev_id, func_id, commit_sha)
        """
        # Map filename -> file_ids
        temp_nodes = cg_nodes.copy()
        temp_nodes['filename_only'] = temp_nodes['function_location'].apply(lambda p: os.path.basename(str(p)) if isinstance(p, str) else None)
        filename_map = temp_nodes.groupby('filename_only')['file_id'].apply(lambda x: list(set(x))).to_dict()
        file_to_funcs = temp_nodes.groupby('file_id')['func_id'].apply(list).to_dict()

        # Iterate commits
        commit_list = []
        try:
            commits = repo.get_commits()
            count = 0
            for c in commits:
                if max_commits is not None and count >= max_commits:
                    break
                commit_list.append(c)
                count += 1
        except Exception as e:
            print(f"Error fetching commits: {e}")

        dev_records = {}
        edge_rows = []
        dev_id_counter = 1

        for commit in commit_list:
            author_login = commit.author.login if commit.author else None
            if not author_login:
                continue
            if author_login not in dev_records:
                dev_records[author_login] = {
                    'dev_id': dev_id_counter,
                    'dev_name': author_login,
                    'dev_email': getattr(commit.author, 'email', '') if commit.author else '',
                    'dev_full': getattr(commit.author, 'name', author_login) if commit.author else author_login
                }
                dev_id_counter += 1
            # fetch detailed commit to get files
            try:
                detailed = repo.get_commit(commit.sha)
                for f in detailed.files:
                    filename_only = os.path.basename(f.filename)
                    file_ids = filename_map.get(filename_only, [])
                    for file_id in file_ids:
                        func_ids = file_to_funcs.get(file_id, [])
                        for func_id in func_ids:
                            edge_rows.append({
                                'dev_id': dev_records[author_login]['dev_id'],
                                'func_id': func_id,
                                'commit_sha': commit.sha
                            })
            except Exception as e:
                print(f"Failed to process commit {commit.sha}: {e}")
                continue

        developers_df = pd.DataFrame(list(dev_records.values())) if dev_records else pd.DataFrame(columns=['dev_id', 'dev_name', 'dev_email', 'dev_full'])
        dev_edges_df = pd.DataFrame(edge_rows).drop_duplicates().reset_index(drop=True) if edge_rows else pd.DataFrame(columns=['dev_id', 'func_id', 'commit_sha'])
        return developers_df, dev_edges_df
    

    def __format_dfs(self, cg_nodes, cg_edges, sg_nodes, sg_edges, imports, imp_edges, hier_1, hier_2):
        """
        Formats the DataFrames for the knowledge graph.

        :param cg_nodes: DataFrame containing call graph nodes.
        :param sg_nodes: DataFrame containing semantic graph nodes.
        :param imports: DataFrame containing import edges.
        :return: Formatted DataFrames.
        """
        cg_nodes = cg_nodes.rename(columns={'func_id': 'ID'})
        cg_edges = cg_edges.rename(columns={'source_id': 'source', 'target_id': 'target'})
        sg_nodes = sg_nodes.rename(columns={'node_id': 'ID'})
        sg_edges = sg_edges.rename(columns={'source_id': 'source', 'target_id': 'target'})
        imports = imports.rename(columns={'import_id': 'ID'})
        imp_edges = imp_edges.rename(columns={'import_id': 'source', 'func_id': 'target'})

        hier_1 = hier_1.rename(columns={'source_id': 'source', 'target_id': 'target'})
        hier_2 = hier_2.rename(columns={'source_id': 'source', 'target_id': 'target'})

        cg_nodes['class_name'] = cg_nodes['combinedName'].apply(lambda x: x.split(".")[0] if "." in x else "")
        cg_nodes['function_name'] = cg_nodes['combinedName'].apply(lambda x: x.split(".")[1] if "." in x else x)

        cg_nodes['docstring'] = cg_nodes['docstring'].fillna("")

        return cg_nodes, cg_edges, sg_nodes, sg_edges, imports, imp_edges, hier_1, hier_2
    


    def __create_question_nodes(self, cluster_nodes):

        questions = pd.DataFrame([
            [0, "general"],
            [1, " bug report/issue/PR"],
            [2, "performance"],
            [3, "feature request"],
        ], columns=["ID", "type"])

        question_edges = []

        for _, row in cluster_nodes.iterrows():
            question_edges.extend([
                {"source": 0, "target": row["ID"]},
                {"source": 1, "target": row["ID"]},
                {"source": 2, "target": row["ID"]}
            ])

        question_edges = pd.DataFrame(question_edges, columns=["source", "target"])

        return questions, question_edges

