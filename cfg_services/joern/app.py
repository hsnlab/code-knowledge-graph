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
                
                temp_code_file = os.path.join(temp_dir, "code.c")
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
                
                # Step 3: Parse .dot output files using pydot
                dot_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.dot')]

                logging.info(f"Request {request_id}: Found DOT files: {dot_files}")

                if not dot_files:
                    raise Exception(f"No .dot files in output: {os.listdir(cfg_output_dir)}")

                nodes = []
                edges = []
                all_dot_content = []

                # Parse DOT files using pydot
                for dot_file in dot_files:
                    dot_path = os.path.join(cfg_output_dir, dot_file)
                    
                    # Read dot content for debugging
                    with open(dot_path, 'r') as f:
                        dot_content = f.read()
                        all_dot_content.append(dot_content)
                    
                    logging.info(f"Request {request_id}: Processing {dot_file} - preview: {dot_content[:150]}...")
                    
                    # Parse DOT file with pydot
                    graphs = pydot.graph_from_dot_file(dot_path)
                    
                    if not graphs:
                        logging.warning(f"No graphs found in {dot_file}")
                        continue
                    
                    # Process each graph (there may be multiple in one file)
                    for graph in graphs:
                        graph_name = graph.get_name().strip('"')
                        
                        # Skip operator and global metadata graphs
                        if graph_name.startswith('<operator>') or graph_name == '<global>':
                            logging.info(f"Request {request_id}: Skipping metadata graph: {graph_name}")
                            continue
                        
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
                            if ',' in metadata:
                                node_type = metadata.split(',')[0].strip()
                                line_info = metadata.split(',')[1].strip()
                            else:
                                node_type = metadata.strip()
                                line_info = None
                            
                            nodes.append({
                                'node_id': node_id,
                                'code': code_text.strip(),
                                'type': node_type,
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

                # Deduplicate edges
                seen_edges = set()
                unique_edges = []
                for edge in edges:
                    edge_key = (edge['source_id'], edge['target_id'])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        unique_edges.append(edge)
                edges = unique_edges

                # Concatenate all dot content
                dot_content = '\n'.join(all_dot_content)

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
        'service': 'joern-simple-queue-service',
        'version': '4.0.0'
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
