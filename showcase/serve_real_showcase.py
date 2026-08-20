"""
Simple HTTP server for the Real Dataset Preprocessing Showcase.
Opens the interactive dashboard in the default browser.
"""

import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8081
SHOWCASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(SHOWCASE_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({".js": "application/javascript"})

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/real_data_showcase.html"
        print(f"\\n{'='*60}")
        print(f"  Parkinson's Real Data Preprocessing Showcase")
        print(f"  Open in browser: {url}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\\n")

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
