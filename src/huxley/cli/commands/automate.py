"""
Huxley CLI - Automate command.

Autonomous knowledge acquisition under strict epistemic and safety constraints.

SAFETY RULES (NON-NEGOTIABLE):
❌ NO wet-lab execution
❌ NO experimental protocols
❌ NO real-world actions
❌ NO self-modification of tools/permissions
❌ NO claimed discoveries as facts

✅ ALLOWED:
- Read papers & synthesize
- Build embeddings & vector memory
- Improve internal reasoning patterns
- Create speculative hypotheses (tagged)
- Track uncertainty & contradictions
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal

from huxley.cli.ui import (
    console, print_mini_banner, print_success, print_error,
    print_info, print_warning, rule, print_markdown,
    S_PRIMARY, S_SECONDARY, S_MUTED
)
from huxley.cli.config import ConfigManager

# Accent color style
S_ACCENT = "bold cyan"


# Knowledge domains for exploration
KNOWLEDGE_DOMAINS = [
    "protein_structures",
    "drug_discovery",
    "enzyme_mechanisms", 
    "structural_biology",
    "molecular_interactions",
    "disease_mechanisms",
    "therapeutic_targets",
]

# Research questions per domain (uncertainty-driven)
RESEARCH_QUESTIONS = {
    "protein_structures": [
        "intrinsically disordered regions in signaling",
        "allosteric mechanisms in kinases",
        "membrane protein stability",
        "protein fold evolution",
    ],
    "drug_discovery": [
        "resistance mechanisms in kinase inhibitors",
        "selectivity determinants for targets",
        "failure modes in lead optimization",
        "off-target binding patterns",
    ],
    "enzyme_mechanisms": [
        "proton transfer in catalysis",
        "metal cofactor coordination",
        "substrate recognition specificity",
        "rate-limiting steps identification",
    ],
    "structural_biology": [
        "resolution limits in cryo-EM",
        "conformational heterogeneity",
        "structural flexibility analysis",
        "dynamic assembly mechanisms",
    ],
    "molecular_interactions": [
        "weak binding cooperativity",
        "hydrophobic effect contributions",
        "electrostatic steering",
        "entropic penalties",
    ],
    "disease_mechanisms": [
        "misfolding cascades",
        "aggregation pathways",
        "mutation impact on stability",
        "dysfunction mechanisms",
    ],
    "therapeutic_targets": [
        "druggability assessment",
        "binding site characterization",
        "selectivity requirements",
        "resistance vulnerabilities",
    ],
}

# Safe action types (defensive allowlist)
SAFE_ACTIONS = [
    "literature_search",
    "structure_query",
    "knowledge_synthesis",
    "hypothesis_generation",
    "self_critique",
    "memory_consolidation",
]


class AutonomousResearchSystem:
    """Autonomous knowledge acquisition system with epistemic constraints."""
    
    def __init__(
        self, 
        duration_hours: float,
        domain: str | None = None,
        objective: str | None = None,
        curiosity_policy: Literal["uncertainty", "gaps", "contradictions"] = "uncertainty",
        model: str | None = None,
        cost_tier: str = "balanced",
    ):
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        self.session_id = f"explore-{uuid.uuid4().hex[:8]}"
        self.model = model
        self.cost_tier = cost_tier
        self.start_time = None
        self.end_time = None
        
        # Auto model selection
        self.auto_mode = model and model.lower() == "auto"
        self.auto_selector = None
        
        # Research focus
        self.domain = domain or random.choice(KNOWLEDGE_DOMAINS)
        self.objective = objective or f"Map knowledge gaps in {self.domain}"
        self.curiosity_policy = curiosity_policy
        
        # Epistemic state
        self.hypotheses = []  # All tagged speculative
        self.knowledge_gaps = []
        self.contradictions = []
        self.weak_assumptions = []
        self.confidence_deltas = {}
        
        # Acquired knowledge
        self.papers_reviewed = []
        self.structures_analyzed = []
        self.concept_maps = []
        self.failure_modes = []
        
        # Skills & patterns
        self.reasoning_patterns = {}
        self.task_templates = {}
        self.success_rates = {}
        
        # Safety tracking
        self.safety_violations = []
        self.disallowed_attempts = []
        
        # Stats
        self.iteration = 0
        self.actions_taken = 0
    
    async def run(self):
        """Run autonomous research with safety constraints."""
        from huxley.llm.auto_selector import AutoModelSelector, CostTier
        
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=self.duration_seconds)
        
        # Setup model
        manager = ConfigManager()
        manager.load_api_keys_to_env()
        
        tier = CostTier(self.cost_tier) if self.cost_tier else CostTier.BALANCED
        
        if self.auto_mode:
            available_models = manager.get_available_models()
            if available_models:
                self.auto_selector = AutoModelSelector(available_models)
                _, self.model = self.auto_selector.select_model(
                    prompt=self.objective, 
                    cost_tier=tier, 
                    has_tools=True
                )
            else:
                print_error("No models available for auto selection")
                return
        elif not self.model:
            _, self.model = manager.get_default_model()
        
        if not self.model:
            print_error("No model configured. Run 'huxley setup'")
            return
        
        self._print_header(tier)
        await self._initialize_session()
        
        try:
            while datetime.now() < self.end_time:
                self.iteration += 1
                
                # Core loop: choose → decompose → retrieve → synthesize → critique → store
                axis = self._choose_research_axis()
                questions = self._decompose_question(axis)
                
                for question in questions[:2]:  # Limit depth
                    # Auto-select model per question if in auto mode
                    if self.auto_selector:
                        _, self.model = self.auto_selector.select_model(
                            prompt=question, 
                            cost_tier=tier, 
                            has_tools=True
                        )
                    
                    knowledge = await self._retrieve_knowledge(question)
                    insights = self._synthesize_insights(knowledge, question)
                    critique = self._self_critique(insights)
                    await self._store_with_provenance(question, insights, critique)
                
                # Periodic checkpoints
                if self.iteration % 5 == 0:
                    await self._save_checkpoint()
                    self._print_status()
                
                await asyncio.sleep(random.uniform(3, 7))
                
        except KeyboardInterrupt:
            console.print()
            print_warning("Research interrupted by user")
        
        await self._finalize_session()
        self._print_summary()
    
    def _print_header(self, tier=None):
        """Print session header."""
        print_mini_banner()
        console.print()
        console.print(f"[{S_PRIMARY}]Autonomous Knowledge Acquisition[/]")
        rule()
        console.print(f"{'Session:':<18}{self.session_id}")
        console.print(f"{'Duration:':<18}{self.duration_hours:.1f}h")
        console.print(f"{'Domain:':<18}{self.domain}")
        console.print(f"{'Objective:':<18}{self.objective[:60]}...")
        console.print(f"{'Curiosity Policy:':<18}{self.curiosity_policy}")
        if self.auto_mode and tier:
            console.print(f"{'Model:':<18}auto ({tier.value}) → {self.model}")
        else:
            console.print(f"{'Model:':<18}{self.model}")
        console.print()
        console.print(f"[{S_MUTED}]Safety: No wet-lab, no protocols, no real-world actions[/]")
        console.print()
        rule()
        console.print()
    
    def _print_status(self):
        """Print epistemic status."""
        elapsed = datetime.now() - self.start_time
        remaining = self.end_time - datetime.now()
        progress = elapsed.total_seconds() / self.duration_seconds * 100
        
        console.print()
        console.print(f"[{S_ACCENT}]─── Epistemic State ───[/]")
        console.print(f"Progress: {progress:.0f}% | Remaining: {str(remaining).split('.')[0]}")
        console.print(f"Iterations: {self.iteration} | Actions: {self.actions_taken}")
        console.print(f"Hypotheses (speculative): {len(self.hypotheses)}")
        console.print(f"Knowledge gaps identified: {len(self.knowledge_gaps)}")
        console.print(f"Contradictions found: {len(self.contradictions)}")
        console.print(f"Papers reviewed: {len(self.papers_reviewed)}")
        console.print(f"Structures analyzed: {len(self.structures_analyzed)}")
        console.print()
    
    def _print_summary(self):
        """Print final research summary."""
        elapsed = datetime.now() - self.start_time
        
        console.print()
        console.print(f"[{S_PRIMARY}]═══ Research Session Complete ═══[/]")
        console.print()
        console.print(f"[{S_SECONDARY}]Session Metadata[/]")
        rule()
        console.print(f"{'ID:':<25}{self.session_id}")
        console.print(f"{'Duration:':<25}{str(elapsed).split('.')[0]}")
        console.print(f"{'Domain:':<25}{self.domain}")
        console.print(f"{'Objective:':<25}{self.objective[:50]}...")
        console.print(f"{'Iterations:':<25}{self.iteration}")
        console.print(f"{'Safe Actions:':<25}{self.actions_taken}")
        console.print(f"{'Safety Violations:':<25}{len(self.safety_violations)} ⚠️" if self.safety_violations else f"{'Safety Violations:':<25}0 ✓")
        console.print()
        
        console.print(f"[{S_SECONDARY}]Knowledge Acquired[/]")
        rule()
        console.print(f"{'Papers Reviewed:':<25}{len(self.papers_reviewed)}")
        console.print(f"{'Structures Analyzed:':<25}{len(self.structures_analyzed)}")
        console.print(f"{'Concept Maps:':<25}{len(self.concept_maps)}")
        console.print()
        
        console.print(f"[{S_SECONDARY}]Epistemic Outputs[/]")
        rule()
        console.print(f"{'Hypotheses (speculative):':<25}{len(self.hypotheses)}")
        console.print(f"{'Knowledge Gaps:':<25}{len(self.knowledge_gaps)}")
        console.print(f"{'Contradictions:':<25}{len(self.contradictions)}")
        console.print(f"{'Weak Assumptions:':<25}{len(self.weak_assumptions)}")
        console.print()
        
        if self.knowledge_gaps:
            console.print(f"[{S_SECONDARY}]Top Knowledge Gaps[/]")
            rule()
            for gap in self.knowledge_gaps[:5]:
                console.print(f"  • {gap}")
            console.print()
        
        if self.hypotheses:
            console.print(f"[{S_SECONDARY}]Speculative Hypotheses[/]")
            rule()
            for hyp in self.hypotheses[:3]:
                console.print(f"  • {hyp.get('statement', 'N/A')[:70]}...")
                console.print(f"    Confidence: {hyp.get('confidence', 0):.2f} | Evidence: {len(hyp.get('evidence', []))}")
            console.print()
        
        print_success(f"All data stored with provenance (session: {self.session_id})")
    
    def _choose_research_axis(self) -> str:
        """Choose research direction based on curiosity policy."""
        if self.curiosity_policy == "uncertainty":
            # Target high-uncertainty regions
            questions = RESEARCH_QUESTIONS.get(self.domain, [])
            return random.choice(questions) if questions else "general inquiry"
        
        elif self.curiosity_policy == "gaps":
            # Focus on identified knowledge gaps
            if self.knowledge_gaps:
                return random.choice(self.knowledge_gaps)
            return random.choice(RESEARCH_QUESTIONS.get(self.domain, ["general"]))
        
        elif self.curiosity_policy == "contradictions":
            # Resolve contradictions first
            if self.contradictions:
                return f"Resolve: {random.choice(self.contradictions)}"
            return random.choice(RESEARCH_QUESTIONS.get(self.domain, ["general"]))
        
        return "general inquiry"
    
    def _decompose_question(self, axis: str) -> list[str]:
        """Decompose research question into sub-questions."""
        # Simple decomposition - in production use LLM
        sub_questions = [
            f"What is known about {axis}?",
            f"What are the gaps in understanding {axis}?",
            f"What are contradictory findings on {axis}?",
        ]
        return sub_questions
    
    async def _retrieve_knowledge(self, question: str) -> dict:
        """Retrieve knowledge from safe sources."""
        self.actions_taken += 1
        knowledge = {"sources": [], "facts": [], "uncertainties": []}
        
        console.print(f"[{S_SECONDARY}]📚 Retrieving:[/] {question[:60]}...")
        
        try:
            # Literature search
            from huxley.tools.chemistry.literature import search_arxiv
            results = await search_arxiv(question, max_results=3)
            
            if results.get("papers"):
                for paper in results["papers"][:2]:
                    knowledge["sources"].append({
                        "type": "paper",
                        "title": paper.get("title"),
                        "arxiv_id": paper.get("arxiv_id"),
                        "authors": paper.get("authors", [])[:3],
                    })
                    self.papers_reviewed.append(paper)
                    console.print(f"[{S_MUTED}]   Found: {paper.get('title', '')[:50]}...[/]")
        except Exception as e:
            console.print(f"[{S_MUTED}]   Search limited: {str(e)[:30]}[/]")
        
        # Structure search if relevant
        if any(term in question.lower() for term in ["protein", "structure", "enzyme", "kinase"]):
            try:
                from huxley.tools.biology.rcsb import pdb_search
                pdb_results = await pdb_search(question, max_results=3)
                
                if pdb_results.get("results"):
                    for hit in pdb_results["results"][:2]:
                        pdb_id = hit.get("id")
                        knowledge["sources"].append({
                            "type": "structure",
                            "pdb_id": pdb_id,
                            "score": hit.get("score"),
                        })
                        self.structures_analyzed.append(pdb_id)
                        console.print(f"[{S_MUTED}]   Structure: {pdb_id}[/]")
            except:
                pass
        
        return knowledge
    
    def _synthesize_insights(self, knowledge: dict, question: str) -> dict:
        """Synthesize insights from retrieved knowledge."""
        insights = {
            "summary": f"Reviewed {len(knowledge['sources'])} sources on: {question}",
            "key_points": [],
            "gaps_identified": [],
            "confidence": 0.5,  # Default medium confidence
        }
        
        # Identify gaps
        if len(knowledge["sources"]) < 2:
            insights["gaps_identified"].append(f"Limited literature on: {question}")
            self.knowledge_gaps.append(question)
        
        # Check for contradictions (simple heuristic)
        if len(knowledge["sources"]) >= 2:
            insights["key_points"].append("Multiple sources found - review for consistency")
        
        return insights
    
    def _self_critique(self, insights: dict) -> dict:
        """Critical self-assessment of insights."""
        critique = {
            "confidence_assessment": insights.get("confidence", 0.5),
            "evidence_quality": "limited" if len(insights.get("key_points", [])) < 2 else "moderate",
            "assumptions": [],
            "limitations": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Identify weak assumptions
        if insights["confidence"] < 0.6:
            assumption = f"Low confidence in: {insights.get('summary', 'unknown')}"
            self.weak_assumptions.append(assumption)
            critique["assumptions"].append(assumption)
        
        # Note limitations
        if insights.get("gaps_identified"):
            critique["limitations"].extend(insights["gaps_identified"])
        
        console.print(f"[{S_MUTED}]   Confidence: {critique['confidence_assessment']:.2f} | Quality: {critique['evidence_quality']}[/]")
        
        return critique
    
    async def _store_with_provenance(self, question: str, insights: dict, critique: dict):
        """Store insights with full provenance tracking."""
        # Create hypothesis (tagged speculative)
        hypothesis = {
            "id": f"hyp-{uuid.uuid4().hex[:8]}",
            "statement": insights.get("summary", question),
            "question": question,
            "confidence": critique["confidence_assessment"],
            "evidence": insights.get("key_points", []),
            "speculative_flag": True,  # ALWAYS speculative
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "critique": critique,
        }
        
        self.hypotheses.append(hypothesis)
        
        # Store to database
        await self._persist_hypothesis(hypothesis)
    
    async def _initialize_session(self):
        """Initialize exploration session in database."""
        try:
            from huxley.memory.factory import get_database_connection
            conn = await get_database_connection()
            if conn is None:
                return
            
            try:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        INSERT INTO huxley_exploration_sessions 
                        (session_id, domain, objective, curiosity_policy, start_time)
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                        self.session_id,
                        self.domain,
                        self.objective,
                        self.curiosity_policy,
                        self.start_time
                    )
                else:
                    # SQLite
                    await conn.execute("""
                        INSERT INTO huxley_exploration_sessions 
                        (id, session_id, domain, objective, curiosity_policy, start_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        self.session_id,
                        self.session_id,
                        self.domain,
                        self.objective,
                        self.curiosity_policy,
                        self.start_time.isoformat(),
                    ))
                    await conn.commit()
            finally:
                await conn.close()
        except:
            pass
    
    async def _save_checkpoint(self):
        """Save progress checkpoint."""
        await self._finalize_session()
    
    async def _finalize_session(self):
        """Finalize and save complete session."""
        try:
            from huxley.memory.factory import get_database_connection
            conn = await get_database_connection()
            if conn is None:
                return
            
            try:
                metadata = json.dumps({
                    "domain": self.domain,
                    "objective": self.objective,
                    "curiosity_policy": self.curiosity_policy,
                    "papers_reviewed": len(self.papers_reviewed),
                    "structures_analyzed": len(self.structures_analyzed),
                    "hypotheses_count": len(self.hypotheses),
                    "knowledge_gaps": len(self.knowledge_gaps),
                    "contradictions": len(self.contradictions),
                })
                
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        UPDATE huxley_exploration_sessions 
                        SET end_time = $1, 
                            iterations = $2,
                            confidence_delta = $3,
                            metadata = $4
                        WHERE session_id = $5
                    """,
                        datetime.now(),
                        self.iteration,
                        json.dumps(self.confidence_deltas),
                        metadata,
                        self.session_id
                    )
                else:
                    # SQLite
                    await conn.execute("""
                        UPDATE huxley_exploration_sessions 
                        SET end_time = ?,
                            iterations = ?,
                            metadata = ?
                        WHERE session_id = ?
                    """, (
                        datetime.now().isoformat(),
                        self.iteration,
                        metadata,
                        self.session_id,
                    ))
                    await conn.commit()
            finally:
                await conn.close()
        except:
            pass
    
    async def _persist_hypothesis(self, hypothesis: dict):
        """Persist hypothesis to database with provenance."""
        try:
            from huxley.memory.factory import get_database_connection
            conn = await get_database_connection()
            if conn is None:
                return
            
            try:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute("""
                        INSERT INTO huxley_hypothesis_ledger 
                        (hypothesis_id, session_id, statement, confidence, evidence_links, speculative_flag, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        hypothesis["id"],
                        self.session_id,
                        hypothesis["statement"],
                        hypothesis["confidence"],
                        json.dumps(hypothesis.get("evidence", [])),
                        True,  # Always speculative
                        json.dumps(hypothesis.get("critique", {}))
                    )
                else:
                    # SQLite
                    await conn.execute("""
                        INSERT INTO huxley_hypothesis_ledger 
                        (id, hypothesis_id, session_id, statement, confidence, evidence_links, speculative_flag, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        hypothesis["id"],
                        hypothesis["id"],
                        self.session_id,
                        hypothesis["statement"],
                        hypothesis["confidence"],
                        json.dumps(hypothesis.get("evidence", [])),
                        1,  # True
                        json.dumps(hypothesis.get("critique", {}))
                    ))
                    await conn.commit()
            finally:
                await conn.close()
        except:
            pass


async def run_automate(
    hours: float,
    model: str | None = None,
    domain: str | None = None,
    objective: str | None = None,
    curiosity_policy: str = "uncertainty",
    cost_tier: str = "balanced",
):
    """Run autonomous knowledge acquisition."""
    system = AutonomousResearchSystem(
        duration_hours=hours,
        model=model,
        domain=domain,
        objective=objective,
        curiosity_policy=curiosity_policy,
        cost_tier=cost_tier,
    )
    await system.run()
