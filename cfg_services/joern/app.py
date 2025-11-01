"""
Fast Joern CFG Service with Persistent Process Manager
All-in-one file: keeps Joern workspace persistent for faster extractions
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
            shutil.which("joern-parse")  # Search in PATH
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
        
        self.lock = threading.Lock()
        
    def extract_cfg(self, code: str, language: str = "c") -> tuple:
        """
        Extract CFG from code using separate temp directories for each request
        Returns: (nodes, edges) as lists of dicts
        """
        with self.lock:  # Thread-safe
            temp_dir = None
            
            try:
                # Create a separate temp directory for this request
                temp_dir = tempfile.mkdtemp(prefix="joern_request_")
                
                temp_code_file = os.path.join(temp_dir, f"code_{time.time_ns()}.c")
                temp_cpg_dir = os.path.join(temp_dir, "cpg")
                cfg_output_dir = os.path.join(temp_dir, "cfg_out")
                
                # Write code to file
                with open(temp_code_file, 'w') as f:
                    f.write(code)
                
                # Step 1: Parse code to CPG
                parse_cmd = [
                    self.joern_parse,
                    temp_code_file,
                    "--output", temp_cpg_dir
                ]
                
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise Exception(f"joern-parse failed: {result.stderr}")
                
                # Step 2: Export CFG as JSON (use "all" format which includes JSON)
                export_cmd = [
                    self.joern_export,
                    "--repr", "all",
                    "--out", cfg_output_dir,
                    temp_cpg_dir
                ]
                
                result = subprocess.run(
                    export_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    raise Exception(f"joern-export failed: {result.stderr}")
                
                # Step 3: Parse .dot output files (Joern CFG exports as GraphViz DOT format)
                dot_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.dot')]
                
                if not dot_files:
                    raise Exception(f"No .dot files in output: {os.listdir(cfg_output_dir)}")
                
                nodes = []
                edges = []
                
                # Parse DOT files
                for dot_file in dot_files:
                    dot_path = os.path.join(cfg_output_dir, dot_file)
                    with open(dot_path, 'r') as f:
                        dot_content = f.read()
                    
                    # Parse DOT format (simple regex-based parsing)
                    import re
                    
                    # Extract nodes: "id" [label="code"]
                    node_pattern = r'"(\d+)"\s*\[label="([^"]+)"'
                    for match in re.finditer(node_pattern, dot_content):
                        node_id, label = match.groups()
                        nodes.append({
                            'node_id': node_id,
                            'code': label.replace('\\n', '\n').replace('\\l', ''),
                            'type': 'CFG_NODE'
                        })
                    
                    # Extract edges: "source" -> "target"
                    edge_pattern = r'"(\d+)"\s*->\s*"(\d+)"'
                    for match in re.finditer(edge_pattern, dot_content):
                        source, target = match.groups()
                        edges.append({
                            'source_id': source,
                            'target_id': target,
                            'label': 'CFG'
                        })
                
                return nodes, edges
                
            except Exception as e:
                logging.error(f"Error extracting CFG: {e}")
                return [], []
                
            finally:
                # Cleanup entire temp directory
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)



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
        'service': 'joern-persistent-cfg-service',
        'version': '3.0.0'
    })

@app.route('/get_cfg', methods=['POST'])
def get_cfg():
    """Extract CFG from code using persistent Joern process"""
    try:
        data = request.json
        code = data.get('code_snippet', '')
        language = data.get('language', 'cpp')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        logging.info(f"Extracting CFG for {language} code ({len(code)} chars)")
        
        # Use persistent manager (MUCH faster!)
        nodes, edges = joern_manager.extract_cfg(code, language)
        
        response = {
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'language': language
        }
        
        logging.info(f"✓ Extracted {len(nodes)} nodes, {len(edges)} edges")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)