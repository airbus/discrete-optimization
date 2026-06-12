#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Launch the interactive transformation graph viewer.

This script:
1. Exports the transformation graph to JSON
2. Starts a local web server
3. Opens the viewer in your default browser
"""

import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path
from threading import Timer

# Import the exporter
from interactive_graph_exporter import export_transformation_graph_to_json


def launch_viewer(port: int = 8000):
    """Launch the interactive viewer.

    Args:
        port: Port for the web server (default: 8000)

    """
    print("=" * 80)
    print("Interactive Transformation Graph Viewer Launcher")
    print("=" * 80)

    # Step 1: Export graph data
    print("\n[1/3] Exporting transformation graph data...")
    try:
        export_transformation_graph_to_json("transformation_graph.json")
    except Exception as e:
        print(f"\n✗ Error exporting graph: {e}")
        sys.exit(1)

    # Step 2: Start web server
    print(f"\n[2/3] Starting web server on port {port}...")

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)

    # Open browser after a short delay
    url = f"http://localhost:{port}/interactive_graph_viewer.html"

    def open_browser():
        print(f"\n[3/3] Opening viewer in browser...")
        print(f"URL: {url}")
        webbrowser.open(url)

    Timer(1.5, open_browser).start()

    # Start server
    print(f"\n✓ Server started successfully!")
    print(f"✓ Viewer available at: {url}")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 80)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        print("✓ Server stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the interactive transformation graph viewer"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export the graph data without starting the server",
    )

    args = parser.parse_args()

    if args.export_only:
        print("Exporting transformation graph data...")
        export_transformation_graph_to_json("transformation_graph.json")
    else:
        launch_viewer(port=args.port)
