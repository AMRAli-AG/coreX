# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import webbrowser
import http.server
import socketserver
import threading
import time

PORT = 8000

def run_export():
    print("==================================================")
    print("Step 1: Extracting Intermediate Model States...")
    print("==================================================")
    
    # Run the exporter using the active python executable (which is in the conda env)
    python_exe = sys.executable
    cmd = [python_exe, "export_dashboard_data.py"]
    
    try:
        subprocess.run(cmd, check=True)
        print("Data successfully extracted to dashboard/data.json!")
    except subprocess.CalledProcessError as e:
        print("Error running export: {}".format(e))
        sys.exit(1)

def start_server():
    # Serve from the root folder where dashboard/index.html is accessible
    handler = http.server.SimpleHTTPRequestHandler
    
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print("Local server running at http://localhost:{}".format(PORT))
        print("Press Ctrl+C to terminate.")
        httpd.serve_forever()

def main():
    # 1. Export the data first
    run_export()
    
    # 2. Start Python local server in a separate background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Give the server a second to start
    time.sleep(1.5)
    
    # 3. Open browser
    url = "http://localhost:{}/dashboard/index.html".format(PORT)
    print("Opening browser to: {}".format(url))
    webbrowser.open(url)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server. Goodbye!")

if __name__ == '__main__':
    main()
