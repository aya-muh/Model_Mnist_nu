#!/usr/bin/env python3
"""
Standalone Runner Script for MNIST Digit Recognition Service
Author: Aya / Antigravity MLOps Specialist
Execution: python run.py
"""

import sys
import os

# Ensure the current directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import app, model

def main():
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))

    print("=" * 65)
    print("  MNIST OPTICAL DIGIT RECOGNITION SERVICE (XGBoost Engine)")
    print("=" * 65)
    if model is not None:
        print("  [STATUS] Trained XGBoost Model: LOADED AND READY")
    else:
        print("  [WARNING] Model file could not be loaded!")
    print(f"  [SERVICE] Local Web Interface: http://{host}:{port}")
    print(f"  [SERVICE] API Endpoint:        http://{host}:{port}/predict")
    print("=" * 65)
    print("  Press Ctrl+C to stop the service.\n")

    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    main()
