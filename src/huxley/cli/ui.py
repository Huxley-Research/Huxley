"""
Huxley CLI - Academic-grade UI components.

Design intent: Auditable, readable output suitable for publication or logs.
No emojis. No decorative elements. Structure over color.
"""

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

# =============================================================================
# COLOR PALETTE (Canonical)
# =============================================================================
# These colors are muted, academic, and non-semantic.
# Output must remain meaningful in grayscale.

COLORS = {
    "primary": "#5CC8FF",      # Institutional Cyan - headers, tool names
    "secondary": "#7AA2F7",    # Muted Indigo - arrows, IDs
    "text": "#C7D0D9",         # Soft Gray - default text
    "muted": "#6B7280",        # Slate - secondary info
    "rule": "#3A3F4B",         # Dim Slate - dividers
    "success": "#9ECE6A",      # Soft Green
    "warning": "#E0AF68",      # Academic Amber
    "error": "#F7768E",        # Dim Red
}

# Rich style strings
S_PRIMARY = f"bold {COLORS['primary']}"
S_SECONDARY = COLORS["secondary"]
S_MUTED = f"dim {COLORS['muted']}"
S_SUCCESS = COLORS["success"]
S_WARNING = COLORS["warning"]
S_ERROR = COLORS["error"]

# Global console instance
console = Console()

# Standard rule width
RULE_WIDTH = 44


def rule():
    """Print a horizontal rule using Unicode box drawing."""
    console.print("─" * RULE_WIDTH, style=S_MUTED)


def print_header(title: str, subtitle: str | None = None):
    """
    Print a section header.
    
    Headers are uppercase, minimal, and informational only.
    """
    console.print()
    console.print(title.upper(), style=S_PRIMARY)
    if subtitle:
        console.print(subtitle, style=S_MUTED)
    rule()


def print_banner():
    """
    Print the Huxley identity header.
    
    Rule: Headers are uppercase, minimal, and informational only.
    Max 2 lines of metadata. No emojis. No slogans.
    """
    console.print()
    console.print("HUXLEY", style=S_PRIMARY)
    console.print("Biological Computational Engine", style=S_MUTED)
    rule()
    console.print(f"{'Version:':<9}0.6.0", style=S_MUTED)
    console.print(f"{'Mode:':<9}CLI", style=S_MUTED)
    console.print()


def print_mini_banner():
    """Print minimal header for subcommands."""
    console.print()
    console.print("HUXLEY", style=S_PRIMARY)
    rule()
    console.print()


# =============================================================================
# STATUS MESSAGES
# =============================================================================
# Status text MUST read clearly without color.
# Color + uppercase text for PASS/FAIL.

def print_success(message: str):
    """Print a success message. No emoji, calm tone."""
    console.print(f"  {message}", style=S_SUCCESS)


def print_error(message: str):
    """Print an error message. No emoji, calm tone."""
    console.print(f"  {message}", style=S_ERROR)


def print_warning(message: str):
    """Print a warning message. No emoji, calm tone."""
    console.print(f"  {message}", style=S_WARNING)


def print_info(message: str):
    """Print an info message. Muted, secondary."""
    console.print(f"  {message}", style=S_MUTED)


def print_step(step: int, total: int, message: str):
    """
    Print a numbered step indicator.
    
    Rule: Numbered lists only (no bullets). Each step <= 1 sentence.
    """
    console.print(f"{step}. {message}")


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================
# Rule: Every run gets a unique, stable identifier.

def print_execution_header(
    execution_id: str,
    agent: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    """
    Print execution context.
    
    IDs are short, lowercase hex.
    Keys left-aligned, values right-aligned.
    """
    console.print()
    console.print(f"Execution {execution_id}", style=S_SECONDARY)
    rule()
    if agent:
        console.print(f"{'Agent:':<10}{agent}")
    if model:
        console.print(f"{'Model:':<10}{model}")
    if provider:
        console.print(f"{'Provider:':<10}{provider}")
    console.print()


# =============================================================================
# TOOL INVOCATION
# =============================================================================
# Rule: Tool calls must show inputs before execution.

def print_tool_call(tool_name: str, params: dict | None = None):
    """
    Print tool invocation.
    
    Arrow indicates invocation.
    Indent parameters by two spaces.
    Parameter names are Title Case.
    """
    console.print(f"[{S_SECONDARY}]->[/] Tool: [{S_PRIMARY}]{tool_name}[/]")
    if params:
        for key, value in params.items():
            # Title case the key
            display_key = key.replace("_", " ").title()
            console.print(f"  {display_key}: {value}", style=S_MUTED)


def print_tool_result(summary: str, details: dict | None = None):
    """
    Print tool result.
    
    Arrow indicates return.
    Never print large payloads by default.
    Reference IDs explicitly.
    """
    console.print(f"[{S_SECONDARY}]<-[/] Result")
    console.print(f"  {summary}")
    if details:
        for key, value in details.items():
            display_key = key.replace("_", " ").title()
            console.print(f"  {display_key}: {value}", style=S_MUTED)


# =============================================================================
# ITERATION / AGENT LOOP
# =============================================================================
# Rule: Each iteration is bounded and labeled.

def print_iteration(current: int, total: int, action: str | None = None):
    """
    Print iteration header.
    
    Always show iteration bounds.
    Metrics use units or percentages.
    """
    console.print()
    console.print(f"Iteration {current} / {total}")
    rule()
    if action:
        console.print(f"{'Action:':<10}{action}")


# =============================================================================
# VERIFICATION
# =============================================================================
# Rule: Verification is binary and explicit.

def print_verification(constraints: list[tuple[str, bool]]):
    """
    Print verification results.
    
    PASS / FAIL in uppercase.
    Constraints quoted verbatim.
    """
    console.print()
    console.print("Verification")
    rule()
    for constraint, passed in constraints:
        status = f"[{S_SUCCESS}]PASS[/]" if passed else f"[{S_ERROR}]FAIL[/]"
        # Right-align status
        padding = RULE_WIDTH - len(constraint) - 8
        console.print(f'"{constraint}"{" " * max(1, padding)}{status}')


# =============================================================================
# ERROR OUTPUT
# =============================================================================
# Rule: Errors are calm, named, and actionable.

def print_error_block(error_name: str, message: str, resolution: str | None = None):
    """
    Print a structured error block.
    
    Error names are PascalCase.
    No stack traces unless --debug.
    Always include a resolution hint.
    """
    console.print()
    console.print(f"[{S_ERROR}]Error:[/] {error_name}")
    console.print()
    console.print(message)
    if resolution:
        console.print()
        console.print("Resolution:")
        console.print(f"  {resolution}")
    console.print()


# =============================================================================
# PLAN OUTPUT
# =============================================================================
# Rule: Agent planning is explicit and enumerable.

def print_plan(steps: list[str]):
    """
    Print execution plan.
    
    Section titles are Title Case.
    Numbered lists only (no bullets).
    Each step <= 1 sentence.
    """
    console.print()
    console.print("Plan")
    rule()
    for i, step in enumerate(steps, 1):
        console.print(f"{i}. {step}")
    console.print()


# =============================================================================
# FINAL OUTPUT
# =============================================================================
# Rule: Final output is clearly separated and structured.

def print_final_response(content: str):
    """
    Print final response.
    
    JSON is pretty-printed.
    No commentary after final output.
    """
    console.print()
    console.print("Final Response")
    rule()
    console.print(content)
    console.print()


# =============================================================================
# DATA DISPLAY
# =============================================================================

def create_table(title: str | None = None) -> Table:
    """
    Create a minimal table.
    
    No decorative boxes. Clean alignment.
    """
    return Table(
        title=title,
        box=None,
        show_header=True,
        header_style=S_MUTED,
        padding=(0, 2),
        collapse_padding=True,
    )


def create_kv_table() -> Table:
    """Create a key-value table (no header)."""
    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        collapse_padding=True,
    )
    table.add_column("Key", style=S_MUTED, width=20)
    table.add_column("Value")
    return table


def create_status_table(title: str = "Status") -> Table:
    """Create a status table (backward compat)."""
    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Item", style=S_MUTED)
    table.add_column("Status")
    return table


def print_protein_card(
    structure_id: str,
    length: int,
    sequence: str,
    confidence: float,
    metrics: dict,
):
    """
    Print protein structure summary.
    
    Academic format: metrics, not decoration.
    """
    console.print()
    console.print(f"Structure {structure_id}", style=S_SECONDARY)
    rule()
    
    # Core metrics
    console.print(f"{'Length:':<16}{length} residues")
    console.print(f"{'Confidence:':<16}{confidence:.1%}")
    
    # Sequence (truncated)
    seq_display = sequence[:40] + "..." if len(sequence) > 40 else sequence
    console.print(f"{'Sequence:':<16}{seq_display}", style=S_MUTED)
    
    console.print()
    console.print("Metrics")
    console.print(f"  {'Clash Score:':<20}{metrics.get('clash_score', 0):.2f}")
    console.print(f"  {'Ramachandran:':<20}{metrics.get('ramachandran_favored', 0):.1f}%")
    console.print(f"  {'Radius of Gyration:':<20}{metrics.get('radius_of_gyration', 0):.1f} A")
    console.print()


def print_status_table(title: str, items: list[tuple[str, str, str | None]]):
    """
    Print a status table.
    
    items: list of (name, status_text, optional_detail)
    status_text should be "OK", "FAIL", or descriptive text
    """
    console.print()
    console.print(title)
    rule()
    
    for name, status, detail in items:
        # Color-code known statuses
        if status.upper() in ("PASS", "OK", "YES", "CONFIGURED", "READY"):
            status_str = f"[{S_SUCCESS}]{status}[/]"
        elif status.upper() in ("FAIL", "NO", "ERROR", "MISSING"):
            status_str = f"[{S_ERROR}]{status}[/]"
        elif status.upper() in ("WARN", "WARNING"):
            status_str = f"[{S_WARNING}]{status}[/]"
        else:
            status_str = status
        
        detail_str = f"  {detail}" if detail else ""
        console.print(f"  {name:<26}{status_str}{detail_str}")
    
    console.print()


# =============================================================================
# PROGRESS
# =============================================================================

def create_progress() -> Progress:
    """Create a minimal progress indicator."""
    return Progress(
        SpinnerColumn(style=S_SECONDARY),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


# =============================================================================
# USER INPUT
# =============================================================================

def ask(prompt: str, default: str | None = None, password: bool = False) -> str:
    """Ask for user input. Clean prompt style."""
    return Prompt.ask(
        f"  {prompt}",
        default=default,
        password=password,
        console=console,
    )


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask for confirmation. Y/N style."""
    return Confirm.ask(f"  {prompt}", default=default, console=console)


# =============================================================================
# COMPLETION MESSAGES
# =============================================================================

def print_setup_complete():
    """Print setup completion message."""
    console.print()
    console.print("Setup Complete")
    rule()
    console.print()
    console.print("Huxley is ready. Available commands:")
    console.print()
    console.print("  huxley generate    Generate a protein structure")
    console.print("  huxley chat        Interactive AI assistant")
    console.print("  huxley check       Check system status")
    console.print("  huxley binder      Design protein binders")
    console.print()


def print_markdown(content: str):
    """Render markdown content."""
    md = Markdown(content)
    console.print(md)
