# Knowledge graph creator

This package contains a workflow for creating knowledge graphs from python repositories. In order to use it, first run:

`pip install requirements.txt`


## 1. Build knowledge graph

First, import the package with the following code:

```
import sys, os
sys.path.append(os.path.abspath('package'))
from package import KnowledgeGraphBuilder
```


Next, initialize the `KnowledgeGraphBuilder` object. In order to build the knowledge graph for large repositories, it is highly recommended to add your github access token to the `KnowledgeGraphBuilder` class constructor (more github API requests can be done with a token). If you have no token, leave it empty: `KnowledgeGraphBuilder()`.

```
kgb = KnowledgeGraphBuilder('your_github_token_here')
repograph = kgb.build_knowledge_graph(repo_name='scikit-learn/scikit-learn')
```

`KnowledgeGraphBuilder.build_knowledge_graph()` function parameters:
- *repo_name*: Name of the repository. Must match the format "owner/repo_name", as it is used for github API calls.
- *graph_type* (optional): Type of subgraph to build from the functions. Can be "CFG" (Control Flow Graph) or "AST" (Abstract Syntax Tree). Default is "CFG".
- *num_of_PRs* (optional): Number of pull requests to retrieve in detail. Defaults to 0 (all).
- *create_embedding* (optional): Whether to create embeddings for the nodes. Defaults to False.
- *repo_path_modifier* (optional): Path modifier for the repository for cases when only a subfolder is meant to be parsed.
- *URI* (optional): URI for the Neo4J data saving.
- *user* (optional): Username for the Neo4J data saving.
- *password* (optional): Password for the Neo4J data saving.
- *project_language* (optional): Programming language of the project (e.g., "python", "cpp", "java"). If not specified, the tool will attempt to auto-detect the language. Specifying the language ensures more accurate parsing and graph generation

Returns:
- *object*: Returns a collection of dataframes.

<div style="height: 35px;"></div>

The knowledge graph object has the following keys:
- *function_nodes*: Call graph nodes, each representing a function
- *function_edges*: Call graph edges (function calls)
- *subgraph_nodes*: Subgraph nodes - Either the AST or CFG of a selected function's code.
- *subgraph_edges*: Subgraph edges (AST or CFG edges)
- *subgraph_function_edges*: Subgraph-node to Callgraph-node edges
- *function_subgraph_edges*: Callgraph-node to Subgraph-node edges
- *import_nodes*: Imported packages used in the repository (nodes in the graph)
- *import_function_edges*: Import nodes connected to functions that use them
- *class_nodes*: Class definitions found in the codebase
- *class_function_edges*: Connects class nodes to their member functions/methods
- *file_nodes*: Files and directories in the repository structure
- *file_file_edges*: Directory hierarchy - connects parent directories to their contained files/subdirectories
- *file_function_edges*: Connects file nodes to functions defined within them
- *file_class_edges*: Connects file nodes to classes defined within them
- *file_import_edges*: Connects file nodes to import statements they contain
- *issues*: Open issues about the repositoy
- *pr_nodes*: Pull requests. The summary text is stored
- *pr_function_edges*: Connects PR nodes to functions of files that were modified in that PR. Modification status and added/deleted rows are stored.
- *issue_nodes*: Issues collected from the repository, stored as nodes.
- *issue_pr_edges*: Issue nodes connected to the PRs solving them.
- *artifacts*: Artifacts of the repo collected into a dataframe.
- *artifacts*: Actions of the repo collected into a dataframe.


Check the object keys:

```
repograph.keys()
```

Check dataframes:

```
repograph['function_nodeds'].head()
```

## 2. Visualization

Create a HTML visualizaiton of the graph with the `visualize_graph` function. NOTE: for large graphs, it is advised to only plot a fraction of the nodes, othervise the visualization might not render properly. Parameters:
- *repograph*: The dictionary containing the created repository graph.
- *show_subgraph_nodes* (optional): Whether to plot the subgraph (CFG or AST) nodes. Defaults to *False*.
- *save_path* (optional): The file path to save the visualization. Defaults to "./graph.html".

```
kgb.visualize_graph(repograph)
```

## 3. Saving the graph

Saving the graph in different formats.

### 3.1 Save it as a dictionary

Saving and loading the resulting graph dictionary as a pickle.
```
import pickle

with open('graph.pkl', 'wb') as f:
    pickle.dump(repograph, f)
```

```
import pickle

with open('graph.pkl', 'rb') as f:
    repograph = pickle.load(f)
```

### 3.2 Saving it to Neo4j database

The result can be saved to a Neo4j database by calling the `store_knowledge_graph_in_neo4j` method. Parameters:
- *URI*: URI for the Neo4J data saving.
- *user*: Username for the Neo4J data saving.
- *password*: Password for the Neo4J data saving.
- *knowledge_graph*: The knowledge graph to save.

If the *URI*, *username* and *password* parameters are provided at the `build_knowledge_graph` method, this function will automatically be called and the graph will be saved to neo4j.

```
kgb.store_knowledge_graph_in_neo4j(
    URI="neo4j://127.0.0.1:7687",
    user="neo4j",
    password="password",
    knowledge_graph=repograph
)
```


## NOTE

1. Running the code to full repositories might take some time to process.
2. If the repository has lots of PRs, it's recommended to use a github token during the initialization of `KnowledgeGraphBuilder`. Even with the token it might take a long time to query everything using the API (for test purposes, it's recommended to limit the maximum number of PRs to pull in detail).