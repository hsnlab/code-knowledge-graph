# Knowledge graph creator

This package contains a workflow for creating knowledge graphs from python repositories. In order to use it, first run:

`pip install requirements.txt`

Usage example:

1. Import

```
import sys, os
sys.path.append(os.path.abspath('package'))
from package.knowledge_graph import KnowledgeGraphBuilder
```

2. Build knowledge graph

In order to build the knowledge graph for large repositories, it is recommended to add your github access token to the `KnowledgeGraphBuilder` class constructor (more github API requests can be done with a token). If you have no token, leave it empty: `KnowledgeGraphBuilder()`.

```
kgb = KnowledgeGraphBuilder('your_github_token_here')
repograph = kgb.build_knowledge_graph(
    repo_path='./sklearn/sklearn/feature_extraction/', 
    repo_name='scikit-learn/scikit-learn',
    graph_type='CFG',
    num_of_PRs=5
)
```

Parameters:
- *repo_path*: Path to the repository.
- *repo_name*: Name of the repository. Must match the format "owner/repo_name", as it is used for github API calls.
- *graph_type*: Type of subgraph to build. Can be "CFG" (Control Flow Graph) or "AST" (Abstract Syntax Tree). Default is "CFG".
- *num_of_PRs*: Number of pull requests to retrieve in detail. Defaults to 5. 0 means all PRs.

Return:
- *object*: Returns a collection of dataframes.



3. Knowledge graph info

The knowledge graph object has the following keys:
- *cg_nodes*: Call graph nodes, each representing a function
- *cg_edges*: Call graph edges (function calls)
- *sg_nodes*: Subgraph nodes - Either the AST or CFG of a selected function's code.
- *sg_edges*: Subgraph edges (AST or CFG edges)
- *hier_1*: Subgraph-node to Callgraph-node edges
- *hier_2*: Callgraph-node to Subgraph-node edges
- *imports*: Imported packages used in the repository (nodes in the graph)
- *imp_edges*: Import nodes connected to functions that use them
- *issues*: Open issues about the repositoy
- *prs*: Pull requests. The summary text is stored
- *pr_edges*: Connects PR nodes to functions of files that were modified in that PR. Modification status and added/deleted rows are stored.
- *artifacts*: Artifacts of the repo collected into a dataframe.

Check the object keys:

```
repograph.keys()
```

Check dataframes:

```
repograph['cg_nodeds'].head()
```

4. Visalization

A visualizazion of the graph can be created using the following command:

```
kgb.visualize_graph(repograph)
```

## NOTE

1. Running the code to full repositories might take some time to process.
2. If the repository has lots of PRs, it's recommended to use a github toke during the initialization of `KnowledgeGraphBuilder`. Even with the token it might take a long time to query everything using the API (for test purposes, it's recommended to limit the number of PRs to pull in detail).