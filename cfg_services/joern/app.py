"""
Fast Joern CFG Service with Simple Queue
Just process one request at a time - no fancy classes needed
"""

from flask import Flask, request, jsonify
import subprocess
import threading
import time
import tempfile
import shutil
import os
import json
import logging
import pydot


class PersistentJoernManager:
    """Manages a long-running Joern workspace for efficient CFG extraction"""
    
    def __init__(self, joern_install_path=None):
        # Auto-detect Joern installation
        if joern_install_path is None:
            joern_install_path = os.environ.get('JOERN_INSTALL', '/opt/joern')
        
        self.joern_install_path = joern_install_path
        
        # Try to find joern-parse
        possible_paths = [
            os.path.join(joern_install_path, "joern-parse"),
            os.path.join(joern_install_path, "joern-cli", "joern-parse"),
            "/usr/local/bin/joern-parse",
            shutil.which("joern-parse")
        ]
        
        self.joern_parse = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.joern_parse = path
                break
        
        if not self.joern_parse:
            raise Exception(f"joern-parse not found! Tried: {possible_paths}")
        
        # Find joern-export
        possible_export_paths = [
            os.path.join(os.path.dirname(self.joern_parse), "joern-export"),
            "/usr/local/bin/joern-export",
            shutil.which("joern-export")
        ]
        
        self.joern_export = None
        for path in possible_export_paths:
            if path and os.path.exists(path):
                self.joern_export = path
                break
        
        if not self.joern_export:
            raise Exception(f"joern-export not found! Tried: {possible_export_paths}")
        
        logging.info(f"Found joern-parse: {self.joern_parse}")
        logging.info(f"Found joern-export: {self.joern_export}")
        
        # Single lock - only one request processes at a time
        self.lock = threading.Lock()
    
    def is_statement_level_node(self, node):
        """
        Determine if a node represents a complete statement (not a sub-expression).
        
        Statement-level nodes are:
        - Assignments (operator.assignment)
        - Function calls (NOT operators)
        - Control structures (if/while/for)
        - Returns
        """
        node_type = node.get('type', '')
        node_type_raw = node.get('type_raw', '')
        code = node.get('code', '').strip()
        
        # 1. Always skip metadata
        if node_type in ['METHOD', 'METHOD_RETURN', 'BLOCK', 'METHOD_PARAMETER_IN', 
                        'METHOD_PARAMETER_OUT', 'TYPE_REF']:
            return False
        
        # 2. Always skip primitives
        if node_type in ['IDENTIFIER', 'LITERAL', 'FIELD_IDENTIFIER'] or node_type_raw == 'FIELD_IDENTIFIER':
            return False
        
        # 3. Always skip empty/generic
        if not code or code in ['ANY', 'RET', 'void', 'int', 'bool', 'UNKNOWN']:
            return False
        
        # 4. For CALL nodes, determine if it's a statement
        if node_type == 'CALL':
            is_operator = '&lt;operator&gt;' in node_type_raw or '<operator>' in node_type_raw
            
            if is_operator:
                # Only ASSIGNMENTS are statement-level operators
                # Everything else (cast, fieldAccess, indirectIndexAccess, shiftLeft, etc.) is sub-expression
                if 'assignment' in node_type_raw.lower():
                    return True
                return False
            else:
                # Regular function calls are statements
                return True
        
        # 5. Control structures are statements
        if node_type == 'CONTROL_STRUCTURE':
            return True
        
        # 6. Returns are statements
        if node_type == 'RETURN':
            return True
        
        return False
        
    def extract_cfg(self, code: str, language: str = "c") -> tuple:
        """
        Extract CFG from code - ONLY ONE AT A TIME
        Returns: (nodes, edges, dot_content) as lists of dicts
        """
        # Wait in line - only one request can run this entire function
        with self.lock:
            request_id = f"{time.time_ns()}"
            logging.info(f"Request {request_id}: STARTING (lock acquired)")    
            temp_dir = None
            
            try:
                # Create temp directory
                temp_dir = tempfile.mkdtemp(prefix=f"joern_{request_id}_")
                
                temp_code_file = os.path.join(temp_dir, "code.cpp")
                temp_cpg_dir = os.path.join(temp_dir, "cpg")
                cfg_output_dir = os.path.join(temp_dir, "cfg_out")
                
                # Write code to file
                with open(temp_code_file, 'w') as f:
                    f.write(code)         
                logging.info(f"Request {request_id}: Written {len(code)} chars to {temp_code_file}")
                
                # Step 1: Parse code to CPG
                parse_cmd = [
                    self.joern_parse,
                    temp_code_file,
                    "--output", temp_cpg_dir
                ]
                
                logging.info(f"Request {request_id}: Running joern-parse...")
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise Exception(f"joern-parse failed: {result.stderr}")
                
                logging.info(f"Request {request_id}: Parse complete")
                
                # Step 2: Export CFG
                export_cmd = [
                    self.joern_export,
                    "--repr", "cfg",
                    "--out", cfg_output_dir,
                    temp_cpg_dir
                ]
                
                logging.info(f"Request {request_id}: Running joern-export...")
                result = subprocess.run(
                    export_cmd,
                    capture_output=True,
                    text=True,
                    timeout=500
                )
                
                if result.returncode != 0:
                    raise Exception(f"joern-export failed: {result.stderr}")
                
                logging.info(f"Request {request_id}: Export complete")
                dot_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.dot')]
                logging.info(f"Request {request_id}: Found DOT files: {dot_files}")
                
                # Step 3: Only parse 0-cfg.dot (the actual method CFG)
                main_cfg_file = os.path.join(cfg_output_dir, '0-cfg.dot')

                if not os.path.exists(main_cfg_file):
                    raise Exception(f"No 0-cfg.dot file found in output: {os.listdir(cfg_output_dir)}")

                logging.info(f"Request {request_id}: Processing main CFG: 0-cfg.dot")

                nodes = []
                edges = []

                # Read dot content
                with open(main_cfg_file, 'r') as f:
                    dot_content = f.read()

                logging.info(f"Request {request_id}: DOT preview: {dot_content[:150]}...")

                # Parse DOT file with pydot
                graphs = pydot.graph_from_dot_file(main_cfg_file)

                if not graphs:
                    raise Exception("No graphs found in 0-cfg.dot")

                # Process the graph (should only be one)
                for graph in graphs:
                    graph_name = graph.get_name().strip('"')
                    
                    logging.info(f"Request {request_id}: Parsing CFG graph: {graph_name}")
                    
                    # Extract nodes
                    for node in graph.get_nodes():
                        node_id = node.get_name().strip('"')
                        
                        # Skip default node definitions
                        if node_id in ['node', 'edge', 'graph']:
                            continue
                        
                        # Get the label
                        label = node.get_label()
                        
                        if not label:
                            continue
                        
                        # Remove outer < > wrapper
                        label = label.strip('<>').strip()
                        
                        # Parse label: "TYPE, LINE<BR/>CODE" or "TYPE<BR/>CODE"
                        parts = label.split('<BR/>')
                        
                        if len(parts) >= 2:
                            metadata = parts[0]
                            code_text = '<BR/>'.join(parts[1:])
                        else:
                            metadata = label
                            code_text = label
                        
                        # Clean HTML entities
                        code_text = code_text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').replace('&quot;', '"')
                        
                        # Extract type and line number from metadata
                        # The format is "TYPE, LINE" or just "TYPE"
                        if ',' in metadata:
                            node_type_raw = metadata.split(',')[0].strip()
                            line_info = metadata.split(',')[1].strip()
                        else:
                            node_type_raw = metadata.strip()
                            line_info = None
                        
                        # INFER the actual CPG node type from the label
                        # If the type looks like a method name, it's a CALL node
                        if node_type_raw in ['METHOD', 'METHOD_RETURN', 'BLOCK', 
                                            'CONTROL_STRUCTURE', 'RETURN', 'IDENTIFIER', 
                                            'LITERAL', 'TYPE_REF']:
                            # It's an actual CPG node type
                            node_type = node_type_raw
                        elif node_type_raw.startswith('<operator>'):
                            # It's an operator call
                            node_type = 'CALL'
                        else:
                            # It's a function/method name, so it's a CALL node
                            node_type = 'CALL'
                        
                        nodes.append({
                            'node_id': node_id,
                            'code': code_text.strip(),
                            'type': node_type,
                            'type_raw': node_type_raw,
                            'line_number': line_info
                        })
                    
                    # Extract edges
                    for edge in graph.get_edges():
                        source = edge.get_source().strip('"')
                        target = edge.get_destination().strip('"')
                        
                        if source and target:
                            edges.append({
                                'source_id': source,
                                'target_id': target,
                                'label': 'CFG'
                            })

                # APPLY STATEMENT-LEVEL FILTERING
                original_node_count = len(nodes)

                # Filter to statement-level nodes only
                filtered_nodes = [n for n in nodes if self.is_statement_level_node(n)]

                # Get remaining node IDs
                filtered_node_ids = {n['node_id'] for n in filtered_nodes}

                # Filter edges to only connect remaining nodes
                filtered_edges = [
                    e for e in edges 
                    if e['source_id'] in filtered_node_ids and e['target_id'] in filtered_node_ids
                ]

                # Deduplicate edges
                seen_edges = set()
                unique_edges = []
                for edge in filtered_edges:
                    edge_key = (edge['source_id'], edge['target_id'])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        unique_edges.append(edge)

                nodes = filtered_nodes
                edges = unique_edges

                logging.info(f"Request {request_id}: Filtered {original_node_count} -> {len(nodes)} nodes")
                logging.info(f"Request {request_id}: COMPLETE - {len(nodes)} nodes, {len(edges)} edges")
                
                return nodes, edges, dot_content
                
            except Exception as e:
                logging.error(f"Request {request_id}: ERROR - {e}", exc_info=True)
                return [], [], None
                
            finally:

                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logging.info(f"Request {request_id}: Cleanup complete")
                    except Exception as e:
                        logging.warning(f"Request {request_id}: Cleanup error: {e}")
                
                logging.info(f"Request {request_id}: FINISHED (lock released)")



# Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize persistent Joern manager on startup
joern_manager = PersistentJoernManager()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'joern-simple-queue-service'
    })

@app.route('/get_cfg', methods=['POST'])
def get_cfg():
    """Extract CFG from code - processes one at a time"""
    try:
        data = request.json
        code = data.get('code_snippet', '')
        language = data.get('language', 'cpp')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        logging.info(f"API: Received request for {language} code ({len(code)} chars)")
        
        # This will wait if another request is processing
        nodes, edges, dot = joern_manager.extract_cfg(code, language)
        
        response = {
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'language': language,
            'dot': dot
        }
        
        logging.info(f"API: Returning {len(nodes)} nodes, {len(edges)} edges")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)