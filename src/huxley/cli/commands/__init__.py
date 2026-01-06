"""CLI command modules."""

from huxley.cli.commands.setup import run_setup
from huxley.cli.commands.generate import run_generate
from huxley.cli.commands.check import run_check
from huxley.cli.commands.chat import run_chat
from huxley.cli.commands.automate import run_automate

__all__ = ["run_setup", "run_generate", "run_check", "run_chat", "run_automate"]
