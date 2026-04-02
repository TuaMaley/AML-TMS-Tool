#!/usr/bin/env python3
"""
AML-TMS Platform — Start Script
================================
Starts the backend API server which also serves the frontend.

Local usage:
    python start.py
    Open: http://localhost:8787

Cloud (Railway / Render):
    PORT is set automatically by the platform via environment variable.

Requirements: Python 3.8+
Install:      pip install -r requirements.txt
"""
import sys, os

# Ensure backend is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from api_server import run

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8787))
    run(port)
