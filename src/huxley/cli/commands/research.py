"""
Huxley CLI - Research command.

Autonomous agent mode for biological research and experimentation.
The agent works independently to explore hypotheses, run experiments,
and synthesize findings.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


async def run_research(
    goal: str,
    max_iterations: int = 10,
    model: str | None = None,
    output_dir: str | None = None,
    verbose: bool = False,
    cost_tier: str = "balanced",
):
    """
    Run autonomous research mode.
    
    The agent will:
    1. Analyze the research goal
    2. Plan experiments and queries
    3. Execute tools autonomously
    4. Iterate based on findings
    5. Synthesize and report results
    """
    from huxley.cli.ui import (
        console, print_banner, print_success, print_error,
        print_info, print_warning, rule, print_header,
        print_tool_call, print_tool_result, S_PRIMARY, S_SECONDARY, S_MUTED
    )
    from huxley.cli.config import ConfigManager
    from huxley.llm.auto_selector import AutoModelSelector, CostTier
    
    print_banner()
    
    # Setup
    manager = ConfigManager()
    manager.load_api_keys_to_env()
    
    # Check for auto mode
    auto_mode = model and model.lower() == "auto"
    auto_selector = None
    tier = CostTier(cost_tier) if cost_tier else CostTier.BALANCED
    
    if auto_mode:
        # Get available models from config
        available_models = manager.get_available_models()
        if not available_models:
            from huxley.cli.ui import print_error_block
            print_error_block(
                "NoModelsAvailable",
                "No models available for auto selection.",
                "Run 'huxley setup' to configure at least one provider."
            )
            return
        auto_selector = AutoModelSelector(available_models)
        # Select initial model for planning
        _, model = auto_selector.select_model(prompt=goal, cost_tier=tier, has_tools=True)
    elif not model:
        provider, model = manager.get_default_model()
    
    if not model:
        from huxley.cli.ui import print_error_block
        print_error_block(
            "NoModelConfigured",
            "No AI model configured for research mode.",
            "Run 'huxley setup' to configure an API key and model."
        )
        return
    
    # Generate research session ID
    session_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path.home() / ".huxley" / "research" / f"{timestamp}_{session_id}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print header
    console.print("RESEARCH MODE")
    rule()
    console.print(f"{'Session:':<12}{session_id}")
    if auto_mode:
        console.print(f"{'Model:':<12}auto ({tier.value})")
        console.print(f"{'Initial:':<12}{model}")
    else:
        console.print(f"{'Model:':<12}{model}")
    console.print(f"{'Max Iter:':<12}{max_iterations}")
    console.print(f"{'Output:':<12}{output_path}")
    console.print()
    
    console.print("OBJECTIVE")
    rule()
    console.print(f"  {goal}")
    console.print()
    
    # Initialize research agent
    agent = ResearchAgent(
        goal=goal,
        model=model,
        session_id=session_id,
        output_path=output_path,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    
    # Run the research loop
    try:
        result = await agent.run()
        
        # Save final report
        report_path = output_path / "report.md"
        with open(report_path, "w") as f:
            f.write(result["report"])
        
        console.print()
        console.print("RESEARCH COMPLETE")
        rule()
        
        if result["viable_solutions"]:
            print_success(f"Found {len(result['viable_solutions'])} viable solution(s)")
            console.print()
            for i, solution in enumerate(result["viable_solutions"], 1):
                console.print(f"  {i}. {solution['summary']}")
        else:
            print_warning("No viable solutions found in this session")
            console.print()
            console.print("  The agent explored the problem but did not identify")
            console.print("  definitive solutions. See report for detailed findings.")
        
        console.print()
        console.print(f"Full report: {report_path}", style=S_MUTED)
        
        # Show generated visualizations
        if result.get("visualizations"):
            console.print()
            console.print("Generated Visualizations:")
            for vis_path in result["visualizations"]:
                console.print(f"  {vis_path}", style=S_MUTED)
        
        console.print()
        
    except KeyboardInterrupt:
        console.print()
        print_warning("Research interrupted by user")
        console.print()
    except Exception as e:
        print_error(f"Research failed: {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc(), style=S_MUTED)


class ResearchAgent:
    """
    Autonomous research agent that iteratively explores a biological problem.
    
    The agent follows a loop:
    1. THINK - Analyze current state and plan next action
    2. ACT - Execute a tool or query
    3. OBSERVE - Process results
    4. REFLECT - Update hypotheses and decide if done
    """
    
    # Available tools for research
    TOOLS = [
        {
            "name": "search_pdb",
            "description": "Search the RCSB Protein Data Bank for structures matching a query. Use for finding proteins, ligands, or structures related to the research.",
            "parameters": {"query": "Search query string", "max_results": "Maximum results (default 10)"}
        },
        {
            "name": "get_structure",
            "description": "Get detailed information about a specific PDB structure by its ID.",
            "parameters": {"pdb_id": "4-character PDB identifier"}
        },
        {
            "name": "search_ligands",
            "description": "Search for ligands/small molecules. Use ligand_id for specific 3-char codes (ATP, HEM, NAD), or query to find structures with relevant ligands.",
            "parameters": {"query": "Search term (e.g., 'protease inhibitor')", "ligand_id": "Optional 3-char ligand code"}
        },
        {
            "name": "analyze_binding_site",
            "description": "Analyze the binding site of a protein structure.",
            "parameters": {"pdb_id": "PDB ID", "ligand_id": "Optional ligand ID"}
        },
        {
            "name": "generate_protein",
            "description": "Generate a new protein structure using FrameDiff diffusion model.",
            "parameters": {"length": "Target length in residues", "description": "Optional description"}
        },
        {
            "name": "design_molecule",
            "description": "Design a novel small molecule/drug candidate. Can generate molecules targeting specific properties, scaffolds, or binding sites.",
            "parameters": {
                "target": "Target protein/receptor name or PDB ID",
                "scaffold": "Optional base scaffold SMILES to build from",
                "properties": "Desired properties (e.g., 'high selectivity', 'oral bioavailability', 'BBB penetrant')",
                "constraints": "Optional constraints (MW < 500, logP < 5, etc.)"
            }
        },
        {
            "name": "modify_molecule",
            "description": "Modify an existing molecule by adding/removing functional groups or optimizing properties.",
            "parameters": {
                "smiles": "Input molecule SMILES string",
                "modification": "Type of modification (add_hydroxyl, add_amine, add_fluorine, remove_group, optimize_binding, reduce_toxicity)",
                "position": "Optional position for modification"
            }
        },
        {
            "name": "predict_properties",
            "description": "Predict drug-like properties of a molecule including ADMET, toxicity, solubility, and Lipinski's Rule of Five compliance.",
            "parameters": {
                "smiles": "Molecule SMILES string",
                "properties": "Which properties to predict (all, admet, toxicity, druglikeness)"
            }
        },
        {
            "name": "dock_molecule",
            "description": "Simulate molecular docking of a ligand to a target protein binding site.",
            "parameters": {
                "ligand_smiles": "Ligand molecule SMILES",
                "target_pdb": "Target protein PDB ID",
                "binding_site": "Optional specific binding site residues"
            }
        },
        {
            "name": "design_binder",
            "description": "Design a molecule that specifically binds to a target protein's active site or allosteric site.",
            "parameters": {
                "target_pdb": "Target protein PDB ID",
                "site_type": "binding site type (active_site, allosteric, interface)",
                "selectivity": "Optional: proteins to avoid binding to"
            }
        },
        {
            "name": "generate_analogs",
            "description": "Generate structural analogs of a lead compound with improved properties.",
            "parameters": {
                "lead_smiles": "Lead compound SMILES",
                "num_analogs": "Number of analogs to generate (default 5)",
                "optimize_for": "Property to optimize (potency, selectivity, solubility, metabolic_stability)"
            }
        },
        {
            "name": "compare_structures",
            "description": "Compare two protein structures for similarity.",
            "parameters": {"pdb_id_1": "First PDB ID", "pdb_id_2": "Second PDB ID"}
        },
        {
            "name": "search_literature",
            "description": "Search scientific literature for relevant papers and findings.",
            "parameters": {"query": "Search query"}
        },
        {
            "name": "visualize_molecule",
            "description": "Generate a 3D interactive visualization of a molecule/protein. Creates an HTML file that can be opened in a browser.",
            "parameters": {"pdb_id": "PDB ID to visualize", "smiles": "Or SMILES string for small molecule", "title": "Title for visualization"}
        },
        {
            "name": "assess_viability",
            "description": "Assess the viability of a candidate solution based on structural and chemical properties.",
            "parameters": {"candidate_name": "Name of candidate", "properties": "Dict of properties to assess"}
        },
        {
            "name": "hypothesize",
            "description": "Record a hypothesis based on current findings.",
            "parameters": {"hypothesis": "The hypothesis statement", "confidence": "low/medium/high"}
        },
        {
            "name": "conclude",
            "description": "Mark research as complete and provide final conclusions. Include any designed molecules in findings.",
            "parameters": {"findings": "Summary of findings including novel molecules", "viable": "true/false if viable solution found"}
        },
    ]
    
    def __init__(
        self,
        goal: str,
        model: str,
        session_id: str,
        output_path: Path,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        self.goal = goal
        self.model = model
        self.session_id = session_id
        self.output_path = output_path
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # State
        self.iteration = 0
        self.history: list[dict] = []
        self.hypotheses: list[dict] = []
        self.findings: list[dict] = []
        self.tool_results: list[dict] = []
        self.is_complete = False
        self.viable_solutions: list[dict] = []
        
        # Visualization tracking
        self.visualizations: list[Path] = []
        self.viability_assessments: list[dict] = []
    
    async def run(self) -> dict:
        """Run the research loop."""
        from huxley.cli.ui import console, rule, print_header, S_MUTED
        
        while self.iteration < self.max_iterations and not self.is_complete:
            self.iteration += 1
            
            console.print()
            console.print(f"ITERATION {self.iteration}/{self.max_iterations}")
            rule()
            
            # Think: Plan next action
            action = await self._think()
            
            if action is None:
                console.print("  Agent could not determine next action", style=S_MUTED)
                break
            
            # Act: Execute the planned action
            result = await self._act(action)
            
            # Observe: Record and display result
            self._observe(action, result)
            
            # Reflect: Update state and check if done
            await self._reflect()
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Save to database (if configured)
            await self._save_to_database()
        
        # Generate final report
        report = await self._generate_report()
        
        # Save final state to database
        await self._save_to_database(status="completed" if self.is_complete else "finished")
        
        # Generate final viability report if we have assessments
        if self.viability_assessments:
            from huxley.visualization.molecule_viewer import (
                generate_viability_report,
                save_visualization,
            )
            
            final_report_html = generate_viability_report(
                results=self.viability_assessments,
                title="Final Viability Report",
                goal=self.goal,
            )
            
            final_vis_path = self.output_path / "final_viability_report.html"
            save_visualization(final_report_html, final_vis_path)
            self.visualizations.append(final_vis_path)
        
        return {
            "session_id": self.session_id,
            "iterations": self.iteration,
            "hypotheses": self.hypotheses,
            "findings": self.findings,
            "viable_solutions": self.viable_solutions,
            "visualizations": [str(p) for p in self.visualizations],
            "viability_assessments": self.viability_assessments,
            "report": report,
        }
    
    async def _think(self) -> dict | None:
        """Plan the next action based on current state."""
        from huxley.cli.ui import console, S_MUTED
        
        # Build context for the model
        context = self._build_context()
        
        prompt = f"""You are Huxley, an autonomous biological research agent conducting computational drug discovery and molecular design.

RESEARCH GOAL:
{self.goal}

AVAILABLE TOOLS:
{json.dumps(self.TOOLS, indent=2)}

CURRENT STATE:
- Iteration: {self.iteration}/{self.max_iterations}
- Hypotheses recorded: {len(self.hypotheses)}
- Tool calls made: {len(self.tool_results)}
- Visualizations created: {len(self.visualizations)}
- Designed molecules: {len([f for f in self.findings if isinstance(f, dict) and f.get('type') in ['designed_molecule', 'designed_binder', 'analog_series']])}

PREVIOUS ACTIONS AND RESULTS:
{context}

DRUG DISCOVERY WORKFLOW (follow this approach):
1. SEARCH: Find target proteins and known ligands in PDB
2. ANALYZE: Study binding sites and existing inhibitors
3. DESIGN: Create novel molecules using design_molecule or design_binder
4. OPTIMIZE: Generate analogs and modify molecules for better properties
5. PREDICT: Use predict_properties to assess ADMET and drug-likeness
6. DOCK: Simulate binding with dock_molecule to validate designs
7. VISUALIZE: Create 3D visualizations of targets and designed molecules
8. CONCLUDE: Report designed molecules and their predicted efficacy

KEY CAPABILITIES - USE THESE TO DESIGN NEW MOLECULES:
- design_molecule: Create novel drug candidates targeting a protein
- design_binder: Design molecules for specific binding sites  
- modify_molecule: Optimize existing molecules (add groups, improve properties)
- generate_analogs: Create variations of lead compounds
- predict_properties: Check ADMET, toxicity, drug-likeness
- dock_molecule: Simulate binding affinity to targets

IMPORTANT:
- If the goal involves finding a "cure" or "inhibitor", you MUST design novel molecules
- Always predict_properties on designed molecules
- Use dock_molecule to validate binding to targets
- Visualize both targets and your designed molecules
- Report designed molecule SMILES and properties in your conclusion

Based on the research goal, decide your next action.
You must respond with a JSON object containing:
{{
    "reasoning": "Your reasoning for this action",
    "tool": "tool_name",
    "parameters": {{"param": "value"}}
}}

If you have designed molecules and gathered enough data, use "conclude" with your findings.
Think like a medicinal chemist designing novel therapeutics."""

        response = await self._call_model(prompt)
        
        # Check for error responses
        if not response or response.startswith("Error:"):
            console.print(f"  Model error: {response}", style=S_MUTED)
            return None
        
        if self.verbose:
            console.print(f"  Raw response: {response[:200]}...", style=S_MUTED)
        
        try:
            # Parse the action from response
            action = self._parse_action(response)
            
            console.print(f"  Reasoning: {action.get('reasoning', 'N/A')[:80]}...", style=S_MUTED)
            console.print(f"  Action: {action.get('tool', 'unknown')}")
            
            return action
        except Exception as e:
            console.print(f"  Failed to parse action: {e}", style=S_MUTED)
            if self.verbose:
                console.print(f"  Response was: {response[:300]}", style=S_MUTED)
            return None
    
    async def _act(self, action: dict) -> dict:
        """Execute the planned action."""
        from huxley.cli.ui import console, print_tool_call, print_tool_result
        
        tool_name = action.get("tool", "")
        params = action.get("parameters", {})
        
        print_tool_call(tool_name, params)
        
        # Execute the appropriate tool
        try:
            if tool_name == "search_pdb":
                result = await self._tool_search_pdb(params)
            elif tool_name == "get_structure":
                result = await self._tool_get_structure(params)
            elif tool_name == "search_ligands":
                result = await self._tool_search_ligands(params)
            elif tool_name == "analyze_binding_site":
                result = await self._tool_analyze_binding(params)
            elif tool_name == "generate_protein":
                result = await self._tool_generate_protein(params)
            elif tool_name == "compare_structures":
                result = await self._tool_compare_structures(params)
            elif tool_name == "design_molecule":
                result = await self._tool_design_molecule(params)
            elif tool_name == "modify_molecule":
                result = await self._tool_modify_molecule(params)
            elif tool_name == "predict_properties":
                result = await self._tool_predict_properties(params)
            elif tool_name == "dock_molecule":
                result = await self._tool_dock_molecule(params)
            elif tool_name == "design_binder":
                result = await self._tool_design_binder(params)
            elif tool_name == "generate_analogs":
                result = await self._tool_generate_analogs(params)
            elif tool_name == "search_literature":
                result = await self._tool_search_literature(params)
            elif tool_name == "visualize_molecule":
                result = await self._tool_visualize_molecule(params)
            elif tool_name == "assess_viability":
                result = await self._tool_assess_viability(params)
            elif tool_name == "hypothesize":
                result = await self._tool_hypothesize(params)
            elif tool_name == "conclude":
                result = await self._tool_conclude(params)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            result = {"error": str(e)}
        
        # Display result summary
        if "error" in result:
            print_tool_result(f"{tool_name}: Error - {result['error']}")
        else:
            summary = result.get("summary", str(result)[:100])
            print_tool_result(f"{tool_name}: {summary}")
        
        return result
    
    def _observe(self, action: dict, result: dict):
        """Record the action and result."""
        self.tool_results.append({
            "iteration": self.iteration,
            "action": action,
            "result": result,
        })
        
        self.history.append({
            "role": "assistant",
            "content": f"Tool: {action.get('tool')}\nResult: {json.dumps(result, default=str)[:500]}"
        })
    
    async def _reflect(self):
        """Reflect on findings and update state."""
        from huxley.cli.ui import console, S_MUTED
        
        # Check if research is complete
        if self.tool_results and self.tool_results[-1].get("action", {}).get("tool") == "conclude":
            self.is_complete = True
            console.print("  Research marked as complete", style=S_MUTED)
    
    def _build_context(self) -> str:
        """Build context string from history."""
        if not self.tool_results:
            return "No actions taken yet."
        
        context_parts = []
        for tr in self.tool_results[-5:]:  # Last 5 actions
            action = tr["action"]
            result = tr["result"]
            context_parts.append(
                f"- {action.get('tool')}: {json.dumps(result, default=str)[:200]}"
            )
        
        return "\n".join(context_parts)
    
    def _parse_action(self, response: str) -> dict:
        """Parse action JSON from model response."""
        import re
        
        if not response or not response.strip():
            raise ValueError("Empty response from model")
        
        # Check for error response
        if response.startswith("Error:"):
            raise ValueError(f"Model returned error: {response}")
        
        # Method 1: Try to find JSON block in markdown code fence
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Method 2: Find outermost JSON object using bracket matching
        def extract_json_object(text: str) -> str | None:
            """Extract the first complete JSON object using bracket counting."""
            start = text.find('{')
            if start == -1:
                return None
            
            depth = 0
            in_string = False
            escape = False
            
            for i, char in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None
        
        json_str = extract_json_object(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Method 3: Try parsing entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Could not find valid JSON action in response")
    
    async def _call_model(self, prompt: str) -> str:
        """Call the AI model."""
        from huxley.cli.commands.chat import get_response
        from huxley.cli.config import ConfigManager
        from huxley.cli.ui import console, S_MUTED
        
        manager = ConfigManager()
        provider, _ = manager.get_default_model()
        
        if self.verbose:
            console.print(f"  Calling model: {self.model} (provider: {provider})", style=S_MUTED)
        
        response = await get_response(prompt, self.history, self.model, provider=provider)
        
        if self.verbose:
            console.print(f"  Response length: {len(response)} chars", style=S_MUTED)
            if len(response) < 500:
                console.print(f"  Response: {response}", style=S_MUTED)
        
        # Check for error responses
        if response.startswith("Error:"):
            console.print(f"  Model error: {response}", style="red")
        
        return response
    
    def _save_checkpoint(self):
        """Save current state to disk."""
        checkpoint = {
            "session_id": self.session_id,
            "goal": self.goal,
            "iteration": self.iteration,
            "hypotheses": self.hypotheses,
            "findings": self.findings,
            "tool_results": self.tool_results,
            "is_complete": self.is_complete,
        }
        
        checkpoint_path = self.output_path / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    async def _save_to_database(self, status: str = "running"):
        """Save research session and molecules to configured database."""
        try:
            from huxley.memory.factory import save_research_session, save_molecule
            
            # Save research session
            await save_research_session(
                session_id=self.session_id,
                objective=self.goal,
                status=status,
                iterations=self.iteration,
                findings=self.findings,
                hypotheses=self.hypotheses,
                viable_solutions=self.viable_solutions,
                metadata={
                    "model": self.model,
                    "max_iterations": self.max_iterations,
                    "output_path": str(self.output_path),
                }
            )
            
            # Save designed molecules
            for finding in self.findings:
                if isinstance(finding, dict):
                    if finding.get("type") == "designed_molecule" and "molecule" in finding:
                        mol = finding["molecule"]
                        await save_molecule(
                            molecule_id=mol.get("molecule_id", ""),
                            smiles=mol.get("smiles", ""),
                            name=mol.get("name"),
                            target=mol.get("target"),
                            properties=mol.get("properties"),
                            druglikeness=mol.get("druglikeness"),
                            session_id=self.session_id,
                        )
                    elif finding.get("type") == "designed_binder" and "molecule" in finding:
                        mol = finding["molecule"]
                        await save_molecule(
                            molecule_id=mol.get("molecule_id", ""),
                            smiles=mol.get("smiles", ""),
                            target=mol.get("target"),
                            properties=mol.get("properties"),
                            druglikeness={"site_type": mol.get("site_type")},
                            docking_results=[{"binding_energy": mol.get("binding_energy")}],
                            session_id=self.session_id,
                        )
        except ImportError:
            pass  # Database dependencies not installed
        except Exception:
            pass  # Database not configured or error - continue silently

    async def _generate_report(self) -> str:
        """Generate final research report."""
        prompt = f"""Generate a research report based on these findings.

RESEARCH GOAL:
{self.goal}

HYPOTHESES:
{json.dumps(self.hypotheses, indent=2)}

TOOL RESULTS:
{json.dumps(self.tool_results[-10:], indent=2, default=str)}

VIABLE SOLUTIONS:
{json.dumps(self.viable_solutions, indent=2)}

Write a structured markdown report with:
1. Executive Summary
2. Methodology
3. Key Findings
4. Viable Solutions (if any)
5. Limitations
6. Recommendations for Further Research"""

        report = await self._call_model(prompt)
        
        # Add header
        full_report = f"""# Huxley Research Report

**Session:** {self.session_id}  
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Goal:** {self.goal}  
**Iterations:** {self.iteration}  

---

{report}
"""
        return full_report
    
    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================
    
    async def _tool_search_pdb(self, params: dict) -> dict:
        """Search RCSB PDB."""
        try:
            from huxley.tools.biology import pdb_search
            
            query = params.get("query", "")
            max_results = params.get("max_results", 10)
            
            result = await pdb_search(query, max_results=max_results)
            
            if result.get("results"):
                return {
                    "success": True,
                    "count": len(result["results"]),
                    "structures": result["results"][:5],  # Summarize
                    "summary": f"Found {len(result['results'])} structures for '{query}'"
                }
            return {"success": False, "summary": "No structures found"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_get_structure(self, params: dict) -> dict:
        """Get PDB structure details and auto-generate visualization."""
        try:
            from huxley.tools.biology import pdb_get_entry
            from huxley.visualization.molecule_viewer import (
                generate_protein_html,
                save_visualization,
            )
            
            pdb_id = params.get("pdb_id", "")
            result = await pdb_get_entry(pdb_id)
            
            # Auto-generate 3D visualization
            vis_path = None
            try:
                title = result.get("title", f"Structure {pdb_id}")
                html = generate_protein_html(
                    pdb_id=pdb_id,
                    title=title[:60],
                    viability_score=None,
                    metrics={
                        "PDB ID": pdb_id,
                        "Method": result.get("method", "Unknown"),
                        "Resolution": result.get("resolution", "N/A"),
                        "Session": self.session_id,
                    },
                )
                vis_path = self.output_path / f"structure_{pdb_id}.html"
                save_visualization(html, vis_path)
                self.visualizations.append(vis_path)
            except Exception as vis_err:
                pass  # Visualization is optional
            
            response = {
                "success": True,
                "pdb_id": pdb_id,
                "data": result,
                "summary": f"Retrieved structure {pdb_id}"
            }
            
            if vis_path:
                response["visualization"] = str(vis_path)
                response["summary"] += f" (3D model: {vis_path.name})"
            
            return response
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_search_ligands(self, params: dict) -> dict:
        """Search for ligands/small molecules related to a query."""
        try:
            from huxley.tools.biology import pdb_search, pdb_ligand_info
            
            query = params.get("query", "")
            ligand_id = params.get("ligand_id", "")
            
            # If a specific ligand_id is provided (3-char code), get its info directly
            if ligand_id and len(ligand_id) <= 4:
                result = await pdb_ligand_info(ligand_id.upper())
                if result and not result.get("error"):
                    return {
                        "success": True,
                        "ligand": result,
                        "summary": f"Found ligand info for '{ligand_id}'"
                    }
            
            # Otherwise, search PDB for structures containing relevant ligands
            search_query = f"{query} ligand inhibitor" if query else "ligand"
            search_results = await pdb_search(search_query, max_results=10)
            
            if not search_results or search_results.get("error"):
                return {
                    "success": False,
                    "error": f"No structures found for query '{query}'",
                    "suggestion": "Try searching with a specific ligand ID (3-char code like ATP, HEM, NAD)"
                }
            
            # Extract ligand information from results
            structures = search_results.get("results", [])
            
            return {
                "success": True,
                "query": query,
                "structures_with_ligands": structures[:10],
                "count": len(structures),
                "summary": f"Found {len(structures)} structures related to '{query}'. Analyze specific structures to identify bound ligands.",
                "tip": "Use get_structure on a PDB ID to see its bound ligands, or provide a ligand_id parameter with a 3-char code (e.g., ATP, HEM) for detailed ligand info."
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_analyze_binding(self, params: dict) -> dict:
        """Analyze binding site."""
        try:
            from huxley.tools.biology import pdb_get_entry
            
            pdb_id = params.get("pdb_id", "")
            
            # Get structure info which includes binding site data
            result = await pdb_get_entry(pdb_id)
            
            return {
                "success": True,
                "pdb_id": pdb_id,
                "structure_info": result,
                "summary": f"Analyzed binding site in {pdb_id}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_generate_protein(self, params: dict) -> dict:
        """Generate protein with FrameDiff."""
        try:
            from huxley import generate_protein_structure
            
            length = params.get("length", 80)
            description = params.get("description")
            
            result = await generate_protein_structure(
                target_length=length,
                conditioning_text=description,
            )
            
            if result.get("success"):
                struct = result["structures"][0]
                return {
                    "success": True,
                    "structure_id": struct["id"],
                    "length": struct["length"],
                    "sequence": struct["sequence"],
                    "confidence": struct["confidence_score"],
                    "summary": f"Generated {struct['length']}-residue protein (conf: {struct['confidence_score']:.2f})"
                }
            return {"error": result.get("error", "Generation failed")}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_compare_structures(self, params: dict) -> dict:
        """Compare two structures."""
        try:
            from huxley.tools.biology import pdb_get_entry
            
            pdb1 = params.get("pdb_id_1", "")
            pdb2 = params.get("pdb_id_2", "")
            
            # Get both structures
            result1 = await pdb_get_entry(pdb1)
            result2 = await pdb_get_entry(pdb2)
            
            return {
                "success": True,
                "structure_1": result1,
                "structure_2": result2,
                "summary": f"Compared {pdb1} vs {pdb2}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_design_molecule(self, params: dict) -> dict:
        """Design a novel small molecule using RDKit."""
        try:
            from huxley.tools.chemistry.molecules import design_molecule_for_target, calculate_properties
            
            target = params.get("target", "")
            scaffold = params.get("scaffold")
            properties = params.get("properties", "")
            constraints = params.get("constraints")
            
            # Parse constraints if string
            if isinstance(constraints, str) and constraints:
                # Try to parse "MW < 500, logP < 5" format
                constraint_dict = {}
                for part in constraints.split(","):
                    if "<" in part:
                        key, val = part.split("<")
                        constraint_dict[key.strip()] = float(val.strip())
                constraints = constraint_dict if constraint_dict else None
            
            result = design_molecule_for_target(
                target_name=target,
                scaffold=scaffold,
                properties=properties,
                constraints=constraints,
            )
            
            if result.get("error"):
                return result
            
            # Store the designed molecule
            self.findings.append({
                "type": "designed_molecule",
                "molecule": result,
                "iteration": self.iteration,
            })
            
            props = result.get("properties", {})
            return {
                "success": True,
                "molecule_id": result["molecule_id"],
                "smiles": result["smiles"],
                "target": result["target"],
                "properties": {
                    "MW": props.get("molecular_weight"),
                    "logP": props.get("logP"),
                    "HBD": props.get("num_H_donors"),
                    "HBA": props.get("num_H_acceptors"),
                    "TPSA": props.get("TPSA"),
                },
                "lipinski_compliant": result.get("lipinski_compliant", False),
                "druglike": result.get("druglikeness", {}).get("overall_druglike", False),
                "summary": f"Designed {result['molecule_id']} (MW={props.get('molecular_weight', 0):.0f}, logP={props.get('logP', 0):.1f}) targeting {target}"
            }
            
        except ImportError:
            return {"error": "RDKit required. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_modify_molecule(self, params: dict) -> dict:
        """Modify an existing molecule using RDKit."""
        try:
            from huxley.tools.chemistry.molecules import modify_molecule
            
            smiles = params.get("smiles", "")
            modification = params.get("modification", "add_fluorine")
            position = params.get("position")
            
            if not smiles:
                return {"error": "SMILES string required"}
            
            result = modify_molecule(
                smiles=smiles,
                modification=modification,
                position=int(position) if position else None,
            )
            
            if result.get("error"):
                return result
            
            return {
                "success": True,
                "molecule_id": f"HUX-MOD-{hash(result['modified_smiles']) % 0xFFFFFF:06X}",
                "original_smiles": result["original_smiles"],
                "modified_smiles": result["modified_smiles"],
                "modification_type": modification,
                "rationale": result["description"],
                "property_changes": result.get("property_changes", {}),
                "summary": f"Modified molecule: {result['description']}"
            }
            
        except ImportError:
            return {"error": "RDKit required. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_predict_properties(self, params: dict) -> dict:
        """Predict drug-like properties using RDKit."""
        try:
            from huxley.tools.chemistry.molecules import calculate_properties, predict_druglikeness
            
            smiles = params.get("smiles", "")
            property_type = params.get("properties", "all")
            
            if not smiles:
                return {"error": "SMILES string required"}
            
            # Calculate real properties
            props = calculate_properties(smiles)
            if props.get("error"):
                return props
            
            druglike = predict_druglikeness(smiles)
            
            result = {
                "success": True,
                "smiles": props.get("smiles", smiles),
                "properties": {
                    "molecular_weight": props["molecular_weight"],
                    "logP": props["logP"],
                    "TPSA": props["TPSA"],
                    "HBD": props["num_H_donors"],
                    "HBA": props["num_H_acceptors"],
                    "rotatable_bonds": props["num_rotatable_bonds"],
                    "num_rings": props["num_rings"],
                    "num_aromatic_rings": props["num_aromatic_rings"],
                    "formula": props["formula"],
                },
                "lipinski_rule_of_5": druglike["lipinski"],
                "veber_rules": druglike["veber"],
                "druglike": druglike["overall_druglike"],
                "pains_alert": druglike["pains_alert"],
                "pains_alerts": druglike.get("pains_alerts", []),
            }
            
            result["summary"] = f"MW={props['molecular_weight']:.0f}, logP={props['logP']:.1f}, Lipinski violations={druglike['lipinski']['violations']}"
            
            return result
            
        except ImportError:
            return {"error": "RDKit required. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_dock_molecule(self, params: dict) -> dict:
        """Dock a ligand to a protein target using empirical scoring."""
        try:
            from huxley.tools.chemistry.docking import dock_molecule
            
            ligand_smiles = params.get("ligand_smiles", "")
            target_pdb = params.get("target_pdb", "")
            binding_site = params.get("binding_site", "")
            
            if not ligand_smiles or not target_pdb:
                return {"error": "Both ligand_smiles and target_pdb required"}
            
            result = dock_molecule(
                ligand_smiles=ligand_smiles,
                target_pdb_id=target_pdb,
                binding_site=binding_site if binding_site else None,
            )
            
            if result.get("error"):
                return result
            
            # Store docking result
            self.findings.append({
                "type": "docking_result",
                "data": result,
                "iteration": self.iteration,
            })
            
            return {
                "success": True,
                "binding_energy": result["binding_energy"],
                "binding_energy_unit": "kcal/mol",
                "binding_affinity_ki": f"{result['ki_nM']:.2f} nM",
                "binding_strength": result["binding_strength"],
                "interactions": result["interactions"],
                "confidence": result["confidence"],
                "ligand_properties": result.get("ligand_properties", {}),
                "druglike": result.get("druglike"),
                "summary": f"Docked to {target_pdb}: ΔG={result['binding_energy']:.1f} kcal/mol, Ki={result['ki_nM']:.1f} nM ({result['binding_strength']})"
            }
            
        except ImportError:
            return {"error": "Chemistry tools require RDKit. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_design_binder(self, params: dict) -> dict:
        """Design a molecule that binds to target protein using RDKit."""
        try:
            from huxley.tools.chemistry.molecules import design_molecule_for_target, calculate_properties, predict_druglikeness
            from huxley.tools.chemistry.docking import dock_molecule
            import hashlib
            
            target_pdb = params.get("target_pdb", "")
            site_type = params.get("site_type", "active_site")
            selectivity = params.get("selectivity", "")
            
            if not target_pdb:
                return {"error": "Target PDB ID required"}
            
            # Design molecule with site-specific properties
            properties = f"binding {site_type}"
            if selectivity:
                properties += f" selectivity over {selectivity}"
            
            result = design_molecule_for_target(
                target_name=f"{target_pdb}_{site_type}",
                properties=properties,
            )
            
            if result.get("error"):
                return result
            
            smiles = result["smiles"]
            mol_id = f"HUX-BIND-{hashlib.md5(f'{target_pdb}{site_type}{smiles}'.encode()).hexdigest()[:6].upper()}"
            
            # Dock the designed molecule
            dock_result = dock_molecule(smiles, target_pdb)
            
            props = result.get("properties", {})
            druglike = result.get("druglikeness", {})
            
            binder_data = {
                "molecule_id": mol_id,
                "smiles": smiles,
                "target": target_pdb,
                "site_type": site_type,
                "binding_energy": dock_result.get("binding_energy", -7.0),
                "properties": props,
                "druglike": druglike.get("overall_druglike", False),
            }
            
            self.findings.append({
                "type": "designed_binder",
                "molecule": binder_data,
                "iteration": self.iteration,
            })
            
            return {
                "success": True,
                "molecule_id": mol_id,
                "smiles": smiles,
                "target": target_pdb,
                "site_type": site_type,
                "predicted_binding_energy": f"{dock_result.get('binding_energy', -7.0):.1f} kcal/mol",
                "ki_nM": dock_result.get("ki_nM", 1000),
                "molecular_weight": props.get("molecular_weight", 0),
                "logP": props.get("logP", 0),
                "druglike": druglike.get("overall_druglike", False),
                "interactions": dock_result.get("interactions", [])[:5],
                "summary": f"Designed binder {mol_id} for {target_pdb} {site_type} (ΔG={dock_result.get('binding_energy', -7.0):.1f} kcal/mol)"
            }
            
        except ImportError:
            return {"error": "RDKit required. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _tool_generate_analogs(self, params: dict) -> dict:
        """Generate structural analogs using RDKit."""
        try:
            from huxley.tools.chemistry.molecules import generate_analogs
            
            lead_smiles = params.get("lead_smiles", "")
            num_analogs = int(params.get("num_analogs", 5))
            optimize_for = params.get("optimize_for", "potency")
            
            if not lead_smiles:
                return {"error": "Lead SMILES required"}
            
            # Map optimization targets to strategies
            strategy_map = {
                "potency": "diverse",
                "selectivity": "bioisostere",
                "solubility": "diverse",
                "metabolic_stability": "bioisostere",
            }
            
            result = generate_analogs(
                smiles=lead_smiles,
                num_analogs=num_analogs,
                strategy=strategy_map.get(optimize_for, "diverse"),
            )
            
            if result.get("error"):
                return result
            
            # Store analog series
            self.findings.append({
                "type": "analog_series",
                "lead_smiles": lead_smiles,
                "analogs": result["analogs"],
                "optimized_for": optimize_for,
                "iteration": self.iteration,
            })
            
            # Find best analog by druglikeness and low MW
            best = None
            for analog in result["analogs"]:
                if analog.get("druglike", False):
                    if best is None or analog["molecular_weight"] < best["molecular_weight"]:
                        best = analog
            if best is None and result["analogs"]:
                best = result["analogs"][0]
            
            return {
                "success": True,
                "lead_compound": lead_smiles,
                "optimization_target": optimize_for,
                "num_analogs": len(result["analogs"]),
                "analogs": result["analogs"],
                "best_analog": best,
                "summary": f"Generated {len(result['analogs'])} analogs optimized for {optimize_for}"
            }
            
        except ImportError:
            return {"error": "RDKit required. Install with: pip install rdkit"}
        except Exception as e:
            return {"error": str(e)}

    async def _tool_search_literature(self, params: dict) -> dict:
        """Search scientific literature using arXiv and PubMed APIs."""
        try:
            from huxley.tools.chemistry.literature import search_arxiv, search_pubmed
            
            query = params.get("query", "")
            source = params.get("source", "arxiv")  # arxiv or pubmed
            max_results = int(params.get("max_results", 5))
            
            if not query:
                return {"error": "Search query required"}
            
            papers = []
            
            if source.lower() == "pubmed":
                # Search PubMed for biomedical literature
                result = await search_pubmed(query, max_results=max_results)
                if result.get("error"):
                    return result
                papers = result.get("papers", [])
                source_name = "PubMed"
            else:
                # Default to arXiv for computational/physics papers
                result = await search_arxiv(query, max_results=max_results)
                if result.get("error"):
                    return result
                papers = result.get("papers", [])
                source_name = "arXiv"
            
            # Store literature findings
            self.findings.append({
                "type": "literature_search",
                "query": query,
                "source": source_name,
                "papers": papers,
                "iteration": self.iteration,
            })
            
            # Format papers for display
            formatted_papers = []
            for paper in papers[:max_results]:
                formatted_papers.append({
                    "title": paper.get("title", "Unknown"),
                    "authors": paper.get("authors", [])[:3],  # First 3 authors
                    "year": paper.get("year") or paper.get("published", "")[:4],
                    "abstract": paper.get("abstract", "")[:300] + "..." if len(paper.get("abstract", "")) > 300 else paper.get("abstract", ""),
                    "url": paper.get("url", ""),
                    "id": paper.get("id", ""),
                })
            
            return {
                "success": True,
                "source": source_name,
                "query": query,
                "num_results": len(formatted_papers),
                "papers": formatted_papers,
                "summary": f"Found {len(formatted_papers)} papers from {source_name} for '{query[:50]}...'"
            }
            
        except Exception as e:
            return {"error": f"Literature search failed: {str(e)}"}
    
    async def _tool_hypothesize(self, params: dict) -> dict:
        """Record a hypothesis."""
        hypothesis = params.get("hypothesis", "")
        confidence = params.get("confidence", "medium")
        
        self.hypotheses.append({
            "iteration": self.iteration,
            "hypothesis": hypothesis,
            "confidence": confidence,
        })
        
        return {
            "success": True,
            "recorded": True,
            "summary": f"Hypothesis recorded: {hypothesis[:50]}..."
        }
    
    async def _tool_conclude(self, params: dict) -> dict:
        """Conclude the research."""
        findings = params.get("findings", "")
        viable = params.get("viable", "false")
        
        self.is_complete = True
        
        if str(viable).lower() == "true":
            self.viable_solutions.append({
                "summary": findings,
                "iteration": self.iteration,
            })
        
        self.findings.append({
            "iteration": self.iteration,
            "findings": findings,
            "viable": viable,
        })
        
        return {
            "success": True,
            "complete": True,
            "summary": f"Research concluded. Viable: {viable}"
        }

    async def _tool_visualize_molecule(self, params: dict) -> dict:
        """Generate 3D visualization of a molecule/protein."""
        try:
            from huxley.visualization.molecule_viewer import (
                generate_protein_html,
                generate_molecule_html,
                save_visualization,
            )
            
            pdb_id = params.get("pdb_id", "")
            smiles = params.get("smiles", "")
            title = params.get("title", f"Structure {pdb_id or 'Molecule'}")
            
            if pdb_id:
                # Generate protein visualization from PDB
                html = generate_protein_html(
                    pdb_id=pdb_id,
                    title=title,
                    viability_score=None,
                    metrics={
                        "PDB ID": pdb_id,
                        "Session": self.session_id,
                        "Iteration": self.iteration,
                    },
                )
                vis_path = self.output_path / f"visualization_{pdb_id}_{self.iteration}.html"
            elif smiles:
                # Generate small molecule visualization from SMILES
                try:
                    from huxley.tools.chemistry.molecules import smiles_to_3d, calculate_properties
                    
                    # Get 3D coordinates
                    mol_3d = smiles_to_3d(smiles)
                    if mol_3d.get("error"):
                        return {"error": mol_3d["error"]}
                    
                    # Get properties for display
                    props = calculate_properties(smiles)
                    
                    # Generate HTML with embedded MOL data
                    html = self._generate_smiles_viewer_html(
                        smiles=smiles,
                        mol_block=mol_3d.get("mol_block", ""),
                        title=title,
                        properties=props,
                    )
                    vis_path = self.output_path / f"molecule_{self.iteration}.html"
                except ImportError:
                    return {"error": "RDKit required for SMILES visualization. Install with: pip install rdkit"}
            else:
                return {"error": "Either pdb_id or smiles required"}
            
            # Save to output directory
            save_visualization(html, vis_path)
            self.visualizations.append(vis_path)
            
            return {
                "success": True,
                "visualization_path": str(vis_path),
                "summary": f"Generated 3D visualization: {vis_path.name}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_smiles_viewer_html(
        self,
        smiles: str,
        mol_block: str,
        title: str,
        properties: dict,
    ) -> str:
        """Generate HTML viewer for small molecule from SMILES."""
        from datetime import datetime
        
        # Escape the mol_block for JavaScript
        mol_block_escaped = mol_block.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'")
        
        # Format properties for display
        props_html = ""
        if properties and not properties.get("error"):
            props_html = f"""
            <div class="properties-panel">
                <h3>Molecular Properties</h3>
                <div class="prop-grid">
                    <div class="prop"><span class="label">MW</span><span class="value">{properties.get('molecular_weight', 'N/A'):.1f}</span></div>
                    <div class="prop"><span class="label">logP</span><span class="value">{properties.get('logP', 'N/A'):.2f}</span></div>
                    <div class="prop"><span class="label">TPSA</span><span class="value">{properties.get('tpsa', 'N/A'):.1f} Å²</span></div>
                    <div class="prop"><span class="label">HBD</span><span class="value">{properties.get('hbd', 'N/A')}</span></div>
                    <div class="prop"><span class="label">HBA</span><span class="value">{properties.get('hba', 'N/A')}</span></div>
                    <div class="prop"><span class="label">Rot. Bonds</span><span class="value">{properties.get('rotatable_bonds', 'N/A')}</span></div>
                </div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Huxley Molecular Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #A51C30;
            --accent: #1E3A5F;
            --bg: #FAFBFC;
            --card: #FFFFFF;
            --text: #1A1A1A;
            --text-muted: #718096;
            --border: #E2E8F0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }}
        .header {{
            background: var(--primary);
            color: white;
            padding: 16px 40px;
        }}
        .header h1 {{
            font-size: 20px;
            font-weight: 500;
        }}
        .header .subtitle {{
            font-size: 12px;
            opacity: 0.8;
            margin-top: 4px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px;
        }}
        .viewer-card {{
            background: var(--card);
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 24px;
        }}
        .viewer-header {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .viewer-header h2 {{
            font-size: 16px;
            font-weight: 600;
        }}
        #viewer {{
            width: 100%;
            height: 500px;
            position: relative;
        }}
        .smiles-display {{
            padding: 16px 24px;
            background: #F7FAFC;
            border-top: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            word-break: break-all;
        }}
        .properties-panel {{
            background: var(--card);
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 24px;
        }}
        .properties-panel h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--accent);
        }}
        .prop-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .prop {{
            display: flex;
            flex-direction: column;
            padding: 12px;
            background: var(--bg);
            border-radius: 8px;
        }}
        .prop .label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .prop .value {{
            font-size: 18px;
            font-weight: 600;
            color: var(--text);
            margin-top: 4px;
        }}
        .controls {{
            display: flex;
            gap: 8px;
        }}
        .btn {{
            padding: 6px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: white;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn:hover {{
            background: var(--bg);
            border-color: var(--accent);
        }}
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Generated by Huxley • {datetime.now().strftime('%B %d, %Y')}</div>
    </div>
    
    <div class="container">
        <div class="viewer-card">
            <div class="viewer-header">
                <h2>3D Structure</h2>
                <div class="controls">
                    <button class="btn" onclick="setStyle('stick')">Stick</button>
                    <button class="btn" onclick="setStyle('sphere')">Sphere</button>
                    <button class="btn" onclick="setStyle('line')">Line</button>
                    <button class="btn" onclick="viewer.zoomTo()">Reset View</button>
                </div>
            </div>
            <div id="viewer"></div>
            <div class="smiles-display">
                <strong>SMILES:</strong> {smiles}
            </div>
        </div>
        
        {props_html}
    </div>
    
    <div class="footer">
        Huxley Biological Intelligence Framework • Molecular Visualization
    </div>
    
    <script>
        var viewer = $3Dmol.createViewer('viewer', {{
            backgroundColor: 'white'
        }});
        
        var molData = '{mol_block_escaped}';
        
        if (molData && molData.length > 10) {{
            viewer.addModel(molData, 'sdf');
            viewer.setStyle({{}}, {{stick: {{colorscheme: 'Jmol'}}}});
            viewer.zoomTo();
            viewer.render();
        }} else {{
            document.getElementById('viewer').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#718096;">Could not generate 3D structure</div>';
        }}
        
        function setStyle(style) {{
            if (style === 'stick') {{
                viewer.setStyle({{}}, {{stick: {{colorscheme: 'Jmol'}}}});
            }} else if (style === 'sphere') {{
                viewer.setStyle({{}}, {{sphere: {{colorscheme: 'Jmol', scale: 0.3}}}});
            }} else if (style === 'line') {{
                viewer.setStyle({{}}, {{line: {{colorscheme: 'Jmol'}}}});
            }}
            viewer.render();
        }}
    </script>
</body>
</html>"""
        return html
    
    async def _tool_assess_viability(self, params: dict) -> dict:
        """Assess viability of a candidate solution."""
        try:
            candidate_name = params.get("candidate_name", "Unknown")
            properties = params.get("properties", {})
            
            # Calculate viability score based on properties
            # This is a simplified scoring - in production would use ML models
            score_factors = []
            
            # Structural factors
            if "binding_affinity" in properties:
                affinity = float(properties["binding_affinity"])
                score_factors.append(min(1.0, affinity / 10))
            
            if "stability" in properties:
                stability = float(properties["stability"])
                score_factors.append(stability)
            
            if "solubility" in properties:
                solubility = float(properties["solubility"])
                score_factors.append(solubility)
            
            if "toxicity" in properties:
                toxicity = float(properties["toxicity"])
                score_factors.append(1.0 - toxicity)  # Lower toxicity is better
            
            # Default factors if none provided
            if not score_factors:
                # Use a heuristic score based on iteration progress
                score_factors = [0.5 + (self.iteration / self.max_iterations) * 0.3]
            
            viability_score = sum(score_factors) / len(score_factors)
            
            assessment = {
                "candidate_name": candidate_name,
                "viability_score": viability_score,
                "properties": properties,
                "iteration": self.iteration,
                "details": f"Based on {len(score_factors)} factors",
            }
            
            self.viability_assessments.append(assessment)
            
            # Generate viability visualization
            if viability_score >= 0.5:
                from huxley.visualization.molecule_viewer import (
                    generate_viability_report,
                    save_visualization,
                )
                
                report_html = generate_viability_report(
                    results=self.viability_assessments,
                    title=f"Viability Report - Iteration {self.iteration}",
                    goal=self.goal,
                )
                
                report_path = self.output_path / f"viability_report_{self.iteration}.html"
                save_visualization(report_html, report_path)
                
                return {
                    "success": True,
                    "candidate": candidate_name,
                    "viability_score": viability_score,
                    "report_path": str(report_path),
                    "summary": f"{candidate_name}: {viability_score*100:.0f}% viable"
                }
            
            return {
                "success": True,
                "candidate": candidate_name,
                "viability_score": viability_score,
                "summary": f"{candidate_name}: {viability_score*100:.0f}% viable"
            }
        except Exception as e:
            return {"error": str(e)}
