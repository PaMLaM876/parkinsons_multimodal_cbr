"""
Launcher script for the Interactive Preprocessing Showcase Dashboard.
Opens the browser automatically and serves the clinical demonstration studio.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
SHOWCASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SHOWCASE_DIR, **kwargs)


def main():
    os.chdir(SHOWCASE_DIR)
    url = f"http://localhost:{PORT}/index.html"
    print(f"[*] Starting Multimodal Preprocessing Showcase on {url}")
    print("[*] Opening interactive presentation dashboard in your browser...")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"[+] Server running at {url}. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Showcase server stopped.")


if __name__ == "__main__":
    main()
