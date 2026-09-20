"""
Climate Intelligence - Local Development & Production Preview Server
====================================================================
Serves the Climate Intelligence web application with proper MIME types,
CORS headers, local IP network detection for mobile testing, and automatic
data export if JSON artifacts are not yet built.
"""

import os
import sys
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def get_local_ip():
    """Attempts to retrieve the local LAN IP address for mobile device testing."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class ClimateHttpHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS for local network and asset testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Cache control for fresh local development
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        super().end_headers()

    def guess_type(self, path):
        # Guarantee accurate MIME types for modern web features
        if path.endswith(".json"):
            return "application/json"
        elif path.endswith(".js"):
            return "application/javascript"
        elif path.endswith(".css"):
            return "text/css"
        elif path.endswith(".svg"):
            return "image/svg+xml"
        elif path.endswith(".png"):
            return "image/png"
        return super().guess_type(path)

def ensure_data_exported():
    json_summary = os.path.join(DIRECTORY, "data", "processed", "climate_analytics_summary.json")
    json_records = os.path.join(DIRECTORY, "data", "processed", "climate_records_sample.json")
    if not os.path.exists(json_summary) or not os.path.exists(json_records):
        print("\n[Init] Precomputed web data JSON not found. Running export_web_data pipeline...")
        from src.export_web_data import export_web_analytics_data
        export_web_analytics_data()
        print("[Init] Data export complete.\n")

def run_server(port=PORT, auto_open=False):
    ensure_data_exported()
    local_ip = get_local_ip()
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, ClimateHttpHandler)
    
    print("=" * 75)
    print("       CLIMATE INTELLIGENCE - INTERACTIVE ANALYTICS WEB PLATFORM")
    print("=" * 75)
    print(f" Localhost Access  : http://localhost:{port}/")
    print(f" Mobile / LAN URL  : http://{local_ip}:{port}/")
    print(" Tagline           : Understand climate patterns through data.")
    print(" Platform Features : 14 Interactive Views | PWA Ready | Mobile QR Support")
    print("=" * 75)
    print(" Press Ctrl+C to terminate the server.\n")
    
    if auto_open:
        webbrowser.open(f"http://localhost:{port}/")
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Climate Intelligence server. Goodbye!")
        httpd.server_close()

if __name__ == "__main__":
    auto = "--open" in sys.argv
    run_server(PORT, auto_open=auto)
