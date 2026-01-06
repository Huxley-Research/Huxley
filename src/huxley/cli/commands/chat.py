"""
Huxley CLI - Chat command.

Interactive AI chat for biology questions with integrated tools.
"""

import asyncio
import uuid
from huxley.cli.ui import (
    console, print_mini_banner, print_success, print_error,
    print_info, print_warning, ask, rule, print_markdown,
    S_PRIMARY, S_SECONDARY, S_MUTED
)
from huxley.cli.config import ConfigManager
from huxley.llm.auto_selector import AutoModelSelector, CostTier


# Available slash commands
SLASH_COMMANDS = {
    "/search-pdb": "Search the Protein Data Bank",
    "/pdb": "Get details about a PDB structure",
    "/construct": "Construct/design a molecule",
    "/properties": "Calculate molecular properties",
    "/validate": "Validate a SMILES string",
    "/druglike": "Check drug-likeness rules",
    "/generate": "Generate a protein structure",
    "/literature": "Search scientific literature",
    "/help": "Show available commands",
}


async def run_chat(model: str | None = None, cost_tier: str = "balanced"):
    """Run interactive chat mode."""
    print_mini_banner()
    
    manager = ConfigManager()
    manager.load_api_keys_to_env()
    
    # Check if auto mode
    auto_mode = model and model.lower() == "auto"
    
    # Determine which model to use
    if model and not auto_mode:
        model_name = model
        provider = None
    else:
        provider, model_name = manager.get_default_model()
    
    if not model_name and not auto_mode:
        # Check if any provider is configured
        configured = manager.get_configured_providers()
        if not configured:
            from huxley.cli.ui import print_error_block
            print_error_block(
                "NoProviderConfigured",
                "No AI provider has been configured.",
                "Run 'huxley setup' to configure an API key."
            )
            return
        
        # Use first available
        provider = configured[0]
        model_name = manager.PROVIDERS[provider]["models"][0]
        print_info(f"Using {model_name}")
    
    # Generate session ID for this chat
    session_id = uuid.uuid4().hex[:8]
    
    console.print("Interactive Chat")
    rule()
    if auto_mode:
        console.print(f"{'Model:':<10}auto ({cost_tier})")
    else:
        console.print(f"{'Model:':<10}{model_name}")
    console.print(f"{'Session:':<10}{session_id}")
    console.print()
    console.print("Commands: 'quit' to exit, 'clear' to reset, '/help' for tools")
    console.print()
    
    # Create auto selector if needed
    auto_selector = None
    if auto_mode:
        from huxley.llm.auto_selector import AutoModelSelector, CostTier
        configured = manager.get_configured_providers()
        auto_selector = AutoModelSelector(configured)
        tier = CostTier(cost_tier)
    
    # Chat loop
    history = []
    
    while True:
        try:
            # Get user input
            user_input = console.input(f"[{S_SECONDARY}]>[/] ")
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print()
                # Save conversation on exit if database configured
                await save_conversation(session_id, history)
                console.print("Session ended")
                break
            
            if user_input.lower() == 'help':
                show_help()
                continue
            
            if user_input.lower() == 'clear':
                history = []
                console.clear()
                print_mini_banner()
                console.print("Chat history cleared")
                console.print()
                continue
            
            if user_input.lower() == 'save':
                await save_conversation(session_id, history)
                print_success(f"Conversation saved (session: {session_id})")
                continue
            
            # Handle slash commands (tools)
            if user_input.startswith('/'):
                result = await handle_slash_command(user_input)
                if result:
                    console.print()
                    console.print(f"[{S_PRIMARY}]Tool Result[/]")
                    console.print()
                    print_markdown(result)
                    console.print()
                    # Add to history
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": result})
                continue
            
            # Auto-select model if in auto mode
            current_model = model_name
            if auto_selector:
                selected_provider, selected_model = auto_selector.select_model(
                    prompt=user_input,
                    cost_tier=tier,
                    has_tools=True,
                )
                current_model = selected_model
                console.print(f"[dim]→ {selected_model}[/dim]")
            
            # Send to model
            console.print()
            with console.status(""):
                response = await get_response(user_input, history, current_model)
            
            # Display response
            console.print(f"[{S_PRIMARY}]Huxley[/]")
            console.print()
            # Try to render as markdown
            try:
                print_markdown(response)
            except:
                console.print(response)
            console.print()
            
            # Add to history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            console.print()
            console.print("Session ended")
            break
        except EOFError:
            break


def show_help():
    """Show chat help."""
    console.print()
    console.print("Commands")
    rule()
    console.print("  quit, exit, q     Exit chat")
    console.print("  clear             Clear chat history")
    console.print("  save              Save conversation to database")
    console.print("  help              Show this help")
    console.print()
    console.print("Tool Commands (use /help for details)")
    rule()
    for cmd, desc in SLASH_COMMANDS.items():
        console.print(f"  {cmd:<18} {desc}")
    console.print()
    console.print("Examples")
    rule()
    console.print("  /search-pdb insulin")
    console.print("  /pdb 1TUP")
    console.print("  /construct a kinase inhibitor")
    console.print("  /properties CCO")
    console.print("  /literature CRISPR gene editing")
    console.print()


async def handle_slash_command(user_input: str) -> str | None:
    """
    Handle slash commands for tool execution.
    
    Returns:
        Formatted result string or None if command not recognized
    """
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    try:
        # /help - Show tool commands
        if command == "/help":
            lines = ["**Available Tool Commands**\n"]
            for cmd, desc in SLASH_COMMANDS.items():
                lines.append(f"- `{cmd}` - {desc}")
            lines.append("\n**Usage Examples:**")
            lines.append("- `/search-pdb insulin` - Search PDB for insulin structures")
            lines.append("- `/pdb 1TUP` - Get details about PDB entry 1TUP")
            lines.append("- `/construct a selective EGFR inhibitor` - Design a molecule")
            lines.append("- `/properties CCO` - Calculate properties of ethanol")
            lines.append("- `/validate c1ccccc1` - Validate benzene SMILES")
            return "\n".join(lines)
        
        # /search-pdb <query> - Search PDB
        if command == "/search-pdb":
            if not args:
                return "**Usage:** `/search-pdb <query>`\n\nExample: `/search-pdb insulin receptor`"
            
            from huxley.tools.biology.rcsb import pdb_search
            result = await pdb_search(args, max_results=10)
            
            if result.get("total_count", 0) == 0:
                return f"No results found for: **{args}**"
            
            lines = [f"**PDB Search Results for '{args}'**\n"]
            lines.append(f"Found {result['total_count']} structures\n")
            for hit in result.get("results", [])[:10]:
                pdb_id = hit.get("id", "?")
                score = hit.get("score", 0)
                lines.append(f"- **{pdb_id}** (score: {score:.1f})")
            lines.append(f"\nUse `/pdb <ID>` to get details")
            return "\n".join(lines)
        
        # /pdb <id> - Get PDB details
        if command == "/pdb":
            if not args:
                return "**Usage:** `/pdb <PDB_ID>`\n\nExample: `/pdb 1TUP`"
            
            from huxley.tools.biology.rcsb import pdb_get_entry
            pdb_id = args.strip().upper()[:4]
            result = await pdb_get_entry(pdb_id)
            
            if "error" in result:
                return f"**Error:** {result['error']}"
            
            lines = [f"**PDB Entry: {pdb_id}**\n"]
            if result.get("title"):
                lines.append(f"**Title:** {result['title']}\n")
            if result.get("experimental_method"):
                lines.append(f"**Method:** {result['experimental_method']}")
            if result.get("resolution"):
                res = result["resolution"]
                if isinstance(res, list):
                    res = res[0] if res else "N/A"
                lines.append(f"**Resolution:** {res} Å")
            if result.get("deposit_date"):
                lines.append(f"**Deposited:** {result['deposit_date']}")
            if result.get("molecular_weight_kda"):
                lines.append(f"**MW:** {result['molecular_weight_kda']:.1f} kDa")
            
            # Entity counts
            counts = result.get("entity_counts", {})
            entities = []
            if counts.get("protein"):
                entities.append(f"{counts['protein']} protein")
            if counts.get("dna"):
                entities.append(f"{counts['dna']} DNA")
            if counts.get("rna"):
                entities.append(f"{counts['rna']} RNA")
            if counts.get("ligands"):
                entities.append(f"{counts['ligands']} ligands")
            if entities:
                lines.append(f"**Contains:** {', '.join(entities)}")
            
            # Organisms
            organisms = set()
            for ent in result.get("entities", []):
                if ent.get("organism"):
                    organisms.add(ent["organism"])
            if organisms:
                lines.append(f"**Organisms:** {', '.join(list(organisms)[:3])}")
            
            # Authors
            if result.get("authors"):
                lines.append(f"**Authors:** {', '.join(result['authors'][:3])}")
            
            # Citation
            if result.get("citation") and result["citation"].get("title"):
                lines.append(f"\n**Citation:** {result['citation']['title'][:100]}...")
                if result["citation"].get("doi"):
                    lines.append(f"**DOI:** {result['citation']['doi']}")
            
            lines.append(f"\n**URL:** {result.get('rcsb_url', f'https://www.rcsb.org/structure/{pdb_id}')}")
            return "\n".join(lines)
        
        # /construct <description> - Design a molecule
        if command == "/construct":
            if not args:
                return "**Usage:** `/construct <description>`\n\nExample: `/construct a kinase inhibitor with good oral bioavailability`"
            
            with console.status("Designing molecule..."):
                result = await design_molecule_with_ai(args)
            return result
        
        # /properties <smiles> - Calculate properties
        if command == "/properties":
            if not args:
                return "**Usage:** `/properties <SMILES>`\n\nExample: `/properties CCO` (ethanol)"
            
            try:
                from huxley.tools.chemistry.molecules import calculate_properties
                result = calculate_properties(args.strip())
                
                if "error" in result:
                    return f"**Error:** {result['error']}"
                
                lines = [f"**Molecular Properties**\n"]
                lines.append(f"**SMILES:** `{result.get('smiles', args)}`")
                lines.append(f"**Formula:** {result.get('formula', 'N/A')}")
                lines.append(f"**MW:** {result.get('molecular_weight', 'N/A')} g/mol")
                lines.append(f"**LogP:** {result.get('logP', 'N/A')}")
                lines.append(f"**TPSA:** {result.get('TPSA', 'N/A')} Å²")
                lines.append(f"**H-Bond Donors:** {result.get('num_H_donors', 'N/A')}")
                lines.append(f"**H-Bond Acceptors:** {result.get('num_H_acceptors', 'N/A')}")
                lines.append(f"**Rotatable Bonds:** {result.get('num_rotatable_bonds', 'N/A')}")
                return "\n".join(lines)
            except ImportError:
                return "**Error:** RDKit not installed. Run `pip install rdkit`"
        
        # /validate <smiles> - Validate SMILES
        if command == "/validate":
            if not args:
                return "**Usage:** `/validate <SMILES>`\n\nExample: `/validate c1ccccc1`"
            
            try:
                from huxley.tools.chemistry.molecules import validate_smiles
                result = validate_smiles(args.strip())
                
                if result.get("valid"):
                    lines = ["**✓ Valid SMILES**\n"]
                    lines.append(f"**Canonical:** `{result.get('canonical_smiles')}`")
                    lines.append(f"**Formula:** {result.get('formula')}")
                    lines.append(f"**Atoms:** {result.get('num_atoms')}")
                    lines.append(f"**Bonds:** {result.get('num_bonds')}")
                    lines.append(f"**Rings:** {result.get('num_rings')}")
                    return "\n".join(lines)
                else:
                    return f"**✗ Invalid SMILES**\n\n{result.get('error', 'Could not parse structure')}"
            except ImportError:
                return "**Error:** RDKit not installed. Run `pip install rdkit`"
        
        # /druglike <smiles> - Check drug-likeness
        if command == "/druglike":
            if not args:
                return "**Usage:** `/druglike <SMILES>`\n\nExample: `/druglike CC(=O)Oc1ccccc1C(=O)O` (aspirin)"
            
            try:
                from huxley.tools.chemistry.molecules import check_drug_likeness
                result = check_drug_likeness(args.strip())
                
                if "error" in result:
                    return f"**Error:** {result['error']}"
                
                lines = ["**Drug-Likeness Analysis**\n"]
                
                # Lipinski
                lip = result.get("lipinski", {})
                lip_pass = lip.get("passes", False)
                lines.append(f"**Lipinski's Rule of 5:** {'✓ PASS' if lip_pass else '✗ FAIL'}")
                if lip.get("violations"):
                    for v in lip["violations"]:
                        lines.append(f"  - {v}")
                
                # Veber
                veb = result.get("veber", {})
                veb_pass = veb.get("passes", False)
                lines.append(f"\n**Veber's Rules:** {'✓ PASS' if veb_pass else '✗ FAIL'}")
                
                # PAINS
                pains = result.get("pains_alerts", [])
                if pains:
                    lines.append(f"\n**⚠️ PAINS Alerts:** {len(pains)}")
                else:
                    lines.append(f"\n**PAINS Alerts:** None detected ✓")
                
                return "\n".join(lines)
            except ImportError:
                return "**Error:** RDKit not installed. Run `pip install rdkit`"
        
        # /generate <description> - Generate protein
        if command == "/generate":
            if not args:
                return "**Usage:** `/generate <description>`\n\nExample: `/generate a 100 residue alpha helical protein`"
            
            console.print()
            await quick_generate(args)
            return None  # quick_generate handles its own output
        
        # /literature <query> - Search literature
        if command == "/literature":
            if not args:
                return "**Usage:** `/literature <query>`\n\nExample: `/literature CRISPR cancer therapy`"
            
            try:
                from huxley.tools.chemistry.literature import search_arxiv
                result = await search_arxiv(args, max_results=5)
                
                if "error" in result:
                    return f"**Error:** {result['error']}"
                
                papers = result.get("papers", [])
                if not papers:
                    return f"No papers found for: **{args}**"
                
                lines = [f"**Literature Search: '{args}'**\n"]
                lines.append(f"Found {result.get('total_results', len(papers))} results\n")
                
                for paper in papers[:5]:
                    lines.append(f"**{paper.get('title', 'Untitled')}**")
                    lines.append(f"  Authors: {', '.join(paper.get('authors', [])[:3])}")
                    lines.append(f"  Published: {paper.get('published', 'N/A')}")
                    if paper.get('arxiv_id'):
                        lines.append(f"  arXiv: {paper['arxiv_id']}")
                    lines.append("")
                
                return "\n".join(lines)
            except Exception as e:
                return f"**Error:** {str(e)}"
        
        # Unknown command
        return f"Unknown command: `{command}`\n\nUse `/help` to see available commands."
        
    except Exception as e:
        return f"**Error executing command:** {str(e)}"


async def design_molecule_with_ai(description: str) -> str:
    """Design a molecule using AI + chemistry tools."""
    try:
        from huxley.tools.chemistry.molecules import design_molecule_for_target, calculate_properties
        
        # Extract target info from description
        desc_lower = description.lower()
        
        # Try to design based on description
        result = design_molecule_for_target(
            target_name=description,
            properties=description,
        )
        
        if "error" in result:
            return f"**Error:** {result['error']}"
        
        smiles = result.get("smiles")
        if not smiles:
            return "**Error:** Could not generate molecule structure"
        
        # Get properties
        props = calculate_properties(smiles)
        
        lines = ["**Designed Molecule**\n"]
        lines.append(f"**Target:** {description}\n")
        lines.append(f"**SMILES:** `{smiles}`")
        lines.append(f"**ID:** {result.get('molecule_id', 'N/A')}")
        lines.append(f"**Strategy:** {result.get('strategy', 'N/A')}")
        lines.append("")
        lines.append("**Properties:**")
        lines.append(f"- MW: {props.get('molecular_weight', 'N/A')} g/mol")
        lines.append(f"- LogP: {props.get('logP', 'N/A')}")
        lines.append(f"- TPSA: {props.get('TPSA', 'N/A')} Å²")
        lines.append(f"- H-Donors: {props.get('num_H_donors', 'N/A')}")
        lines.append(f"- H-Acceptors: {props.get('num_H_acceptors', 'N/A')}")
        
        # Check drug-likeness
        try:
            from huxley.tools.chemistry.molecules import check_drug_likeness
            dl = check_drug_likeness(smiles)
            lip_pass = dl.get("lipinski", {}).get("passes", False)
            lines.append(f"\n**Drug-like:** {'✓ Yes' if lip_pass else '✗ No'}")
        except:
            pass
        
        lines.append(f"\nUse `/properties {smiles}` for full analysis")
        
        return "\n".join(lines)
        
    except ImportError:
        return "**Error:** RDKit not installed. Run `pip install rdkit`"
    except Exception as e:
        return f"**Error:** {str(e)}"


async def save_conversation(session_id: str, history: list[dict]):
    """Save conversation to database if configured."""
    if not history:
        return
    
    try:
        from huxley.memory.factory import get_database_connection
        import json
        
        conn = await get_database_connection()
        if conn is None:
            return
        
        try:
            if hasattr(conn, 'execute') and hasattr(conn, 'fetchrow'):
                # asyncpg (PostgreSQL)
                # Create or update conversation
                await conn.execute("""
                    INSERT INTO huxley_conversations (session_id, metadata)
                    VALUES ($1, $2)
                    ON CONFLICT (session_id) DO UPDATE SET updated_at = NOW()
                """, session_id, json.dumps({}))
                
                # Get conversation ID
                row = await conn.fetchrow(
                    "SELECT id FROM huxley_conversations WHERE session_id = $1",
                    session_id
                )
                if row:
                    conv_id = row['id']
                    
                    # Save messages
                    for msg in history:
                        await conn.execute("""
                            INSERT INTO huxley_messages (conversation_id, role, content)
                            VALUES ($1, $2, $3)
                        """, conv_id, msg['role'], msg['content'])
            else:
                # aiosqlite (SQLite)
                import uuid as uuid_mod
                conv_id = str(uuid_mod.uuid4())
                
                await conn.execute("""
                    INSERT OR REPLACE INTO huxley_conversations (id, session_id, metadata)
                    VALUES (?, ?, ?)
                """, (conv_id, session_id, json.dumps({})))
                
                for msg in history:
                    await conn.execute("""
                        INSERT INTO huxley_messages (id, conversation_id, role, content)
                        VALUES (?, ?, ?, ?)
                    """, (str(uuid_mod.uuid4()), conv_id, msg['role'], msg['content']))
                
                await conn.commit()
                
        finally:
            await conn.close()
            
    except ImportError:
        pass  # Database dependencies not installed
    except Exception:
        pass  # Database not configured or error


async def get_response(
    message: str,
    history: list[dict],
    model: str,
    provider: str | None = None,
) -> str:
    """Get a response from the AI model."""
    from huxley.cli.config import ConfigManager
    manager = ConfigManager()
    
    # Get the default provider if not specified
    if not provider:
        default_provider, default_model = manager.get_default_model()
        if default_provider:
            provider = default_provider
    
    # If provider specified, use its config
    if provider:
        config = manager.get_provider_config(provider)
        if config.get("compatible") or config.get("no_key"):
            return await call_openai_compatible(
                message, history, model,
                base_url=config.get("base_url"),
                api_key=config.get("api_key"),
            )
        elif provider == "anthropic":
            return await call_anthropic(message, history, model)
        elif provider == "google":
            return await call_google(message, history, model)
        elif provider == "openai":
            return await call_openai(message, history, model)
    
    # Fallback: Determine provider from model name
    if "gpt" in model.lower():
        return await call_openai(message, history, model)
    elif "claude" in model.lower():
        return await call_anthropic(message, history, model)
    elif "gemini" in model.lower():
        return await call_google(message, history, model)
    elif "grok" in model.lower():
        return await call_openai_compatible(
            message, history, model,
            base_url="https://api.x.ai/v1",
            api_key=manager.get_api_key("xai"),
        )
    elif "llama" in model.lower() or "mistral" in model.lower() or "mixtral" in model.lower():
        # Try local providers first
        ollama_config = manager.get_provider_config("ollama")
        return await call_openai_compatible(
            message, history, model,
            base_url=ollama_config.get("base_url"),
            api_key=None,
        )
    else:
        # Default: try to use custom provider if configured
        custom_config = manager.get_provider_config("custom")
        if custom_config.get("base_url"):
            return await call_openai_compatible(
                message, history, model,
                base_url=custom_config.get("base_url"),
                api_key=custom_config.get("api_key"),
            )
        # Last resort: OpenAI
        return await call_openai(message, history, model)


async def call_openai_compatible(
    message: str,
    history: list,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call any OpenAI-compatible API."""
    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
        )
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            *history,
            {"role": "user", "content": message},
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content
    except ImportError:
        return "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        return f"Error: {str(e)}"


async def call_openai(message: str, history: list, model: str) -> str:
    """Call OpenAI API."""
    try:
        import openai
        client = openai.AsyncOpenAI()
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            *history,
            {"role": "user", "content": message},
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content
    except ImportError:
        return "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        return f"Error: {str(e)}"


async def call_anthropic(message: str, history: list, model: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        
        messages = [
            *history,
            {"role": "user", "content": message},
        ]
        
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=get_system_prompt(),
            messages=messages,
        )
        
        return response.content[0].text
    except ImportError:
        return "Anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Error: {str(e)}"


async def call_google(message: str, history: list, model: str) -> str:
    """Call Google AI API."""
    try:
        import google.generativeai as genai
        
        gen_model = genai.GenerativeModel(model)
        
        chat_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        
        chat = gen_model.start_chat(history=chat_history)
        response = await asyncio.to_thread(chat.send_message, message)
        
        return response.text
    except ImportError:
        return "Google AI package not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Error: {str(e)}"


async def call_xai(message: str, history: list, model: str) -> str:
    """Call xAI/Grok API."""
    import os
    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            *history,
            {"role": "user", "content": message},
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def get_system_prompt() -> str:
    """Get the system prompt for Huxley chat."""
    return """You are Huxley, a friendly AI assistant specialized in biology, proteins, and structural biology.

You help users:
- Understand proteins and their structures
- Design new proteins using diffusion models
- Analyze biological data
- Use the Huxley framework for computational biology

Be helpful, concise, and accurate. When discussing proteins, use proper terminology but explain complex concepts clearly.

You have access to tools for:
- Protein structure generation (FrameDiff)
- PDB database queries
- Structure validation

When users ask about generating proteins, you can guide them to use 'huxley generate' command or the Python API."""


async def quick_generate(description: str | None = None):
    """Quick protein generation from chat."""
    from huxley import generate_protein_structure
    from huxley.cli.ui import print_protein_card, print_tool_call
    
    console.print()
    print_tool_call("generate_protein_structure", {
        "target_length": 80,
        "description": description or "none",
    })
    
    with console.status("Running SE(3) diffusion..."):
        result = await generate_protein_structure(
            target_length=80,
            conditioning_text=description,
        )
    
    if result["success"]:
        struct = result["structures"][0]
        print_protein_card(
            structure_id=struct["id"],
            length=struct["length"],
            sequence=struct["sequence"],
            confidence=struct["confidence_score"],
            metrics=struct["metrics"],
        )
    else:
        print_error(f"Generation failed: {result.get('error')}")
    
    console.print()
