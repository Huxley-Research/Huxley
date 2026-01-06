"""
Built-in tools for common operations.

These tools provide basic functionality that most agents need.
Domain-specific tools (e.g., biology tools) should be in separate modules.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from huxley.tools.registry import tool


@tool(tags={"system", "utility"})
def echo(message: str) -> str:
    """
    Echo a message back. Useful for testing tool execution.

    :param message: The message to echo
    """
    return message


@tool(tags={"system", "utility"})
def json_parse(json_string: str) -> dict[str, Any]:
    """
    Parse a JSON string into a Python dictionary.

    :param json_string: Valid JSON string to parse
    """
    return json.loads(json_string)


@tool(tags={"system", "utility"})
def json_format(data: dict[str, Any], indent: int = 2) -> str:
    """
    Format a dictionary as a JSON string.

    :param data: Dictionary to format
    :param indent: Indentation level (default 2)
    """
    return json.dumps(data, indent=indent, default=str)


@tool(tags={"system", "io"})
def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read the contents of a file.

    :param path: Path to the file
    :param encoding: File encoding (default utf-8)
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


@tool(tags={"system", "io"})
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.

    :param path: Path to the file
    :param content: Content to write
    :param encoding: File encoding (default utf-8)
    """
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


@tool(tags={"system", "utility"})
def shell_command(
    command: str,
    timeout: int = 30,
    cwd: str | None = None,
) -> dict[str, Any]:
    """
    Execute a shell command and return the result.

    WARNING: This tool executes arbitrary shell commands.
    Only enable in trusted environments.

    :param command: Shell command to execute
    :param timeout: Timeout in seconds (default 30)
    :param cwd: Working directory for the command
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1,
            "success": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "success": False,
        }


@tool(tags={"math", "utility"})
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Supports basic arithmetic: +, -, *, /, **, ()

    :param expression: Mathematical expression to evaluate
    """
    # Restricted eval for safety
    allowed_chars = set("0123456789+-*/().eE ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Invalid characters in expression: {expression}")

    return float(eval(expression, {"__builtins__": {}}, {}))


@tool(tags={"time", "utility"})
def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.

    :param format: strftime format string
    """
    from datetime import datetime

    return datetime.now().strftime(format)
