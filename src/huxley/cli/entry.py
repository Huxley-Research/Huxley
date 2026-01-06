#!/usr/bin/env python3
"""
Huxley CLI Entry Point.

This module sets up the environment before any huxley imports
to suppress debug logging in CLI mode.
"""

import os
import sys
import logging


def main():
    """Entry point that suppresses import-time logging."""
    # Set log level to WARNING BEFORE any huxley imports
    os.environ["HUXLEY_LOG_LEVEL"] = "WARNING"
    os.environ["HUXLEY_LOG__LEVEL"] = "WARNING"
    
    # Pre-configure structlog before huxley imports
    # This prevents debug messages from tool registration
    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.WARNING
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )
    
    # Now import and run the CLI
    from huxley.cli.main import cli
    cli()


if __name__ == "__main__":
    main()
