"""
Biology domain tools for Huxley.

This module provides tools for biological data retrieval and analysis,
including:
- RCSB PDB (Protein Data Bank) integration
- Diffusion-based protein structure generation (FrameDiff recommended)
"""

from huxley.tools.biology.rcsb import (
    pdb_search,
    pdb_get_entry,
    pdb_get_entity,
    pdb_get_assembly,
    pdb_sequence_search,
    pdb_structure_summary,
    pdb_ligand_info,
    pdb_advanced_search,
)

from huxley.tools.biology.diffusion import (
    generate_protein_structure,
    scaffold_protein_motif,
    design_protein_binder,
    generate_symmetric_assembly,
    validate_protein_structure,
    download_framediff_weights,
    check_framediff_setup,
    # Data types
    DiffusionBackend,
    DiffusionConfig,
    GeneratedStructure,
    DiffusionResult,
    ConditioningType,
    MotifConstraint,
    BindingConstraint,
    SymmetryConstraint,
    Residue,
    # Backend implementations
    BaseDiffusionBackend,
    FrameDiffBackend,
    MockDiffusionBackend,
    register_backend,
    get_backend,
)

__all__ = [
    # RCSB PDB tools
    "pdb_search",
    "pdb_get_entry",
    "pdb_get_entity",
    "pdb_get_assembly",
    "pdb_sequence_search",
    "pdb_structure_summary",
    "pdb_ligand_info",
    "pdb_advanced_search",
    # Diffusion tools
    "generate_protein_structure",
    "scaffold_protein_motif",
    "design_protein_binder",
    "generate_symmetric_assembly",
    "validate_protein_structure",
    "download_framediff_weights",
    "check_framediff_setup",
    # Diffusion types
    "DiffusionBackend",
    "DiffusionConfig",
    "GeneratedStructure",
    "DiffusionResult",
    "ConditioningType",
    "MotifConstraint",
    "BindingConstraint",
    "SymmetryConstraint",
    "Residue",
    # Diffusion backends
    "BaseDiffusionBackend",
    "FrameDiffBackend",
    "MockDiffusionBackend",
    "register_backend",
    "get_backend",
]
