"""
Orchestrator script to start the Flask Backend API and launch the browser.
"""
import os
import sys
import threading
import webbrowser
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def open_browser():
    """Wait a second for the server to start, then open the browser."""
    time.sleep(1.5)
    url = "http://127.0.0.1:8081/"
    print(f"\n[*] Opening Showcase UI in browser: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    from showcase.app import app
    
    print("=" * 60)
    print("  PARKINSON'S MULTIMODAL CBR — LIVE SHOWCASE SERVER")
    print("=" * 60)
    
    # Start browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Flask server (blocking)
    # Turn off reloader so we don't open multiple browser tabs
    app.run(host="0.0.0.0", port=8081, debug=False)
