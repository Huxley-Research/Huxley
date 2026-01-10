"""
Huxley Docker Onboarding Entry Point.

Run with: python -m huxley.docker.onboarding
"""

import sys

if __name__ == "__main__":
    from huxley.docker.onboarding import run_server
    run_server()
