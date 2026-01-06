"""
Chemistry tools for Huxley.

Provides real molecular chemistry capabilities using RDKit:
- Property calculation (MW, logP, TPSA, etc.)
- Molecule manipulation and modification
- Drug-likeness assessment
- SMILES validation and conversion
"""

from huxley.tools.chemistry.molecules import (
    validate_smiles,
    calculate_properties,
    predict_druglikeness,
    modify_molecule,
    generate_analogs,
    design_molecule_for_target,
    smiles_to_3d,
)

from huxley.tools.chemistry.docking import (
    prepare_ligand,
    dock_molecule,
    analyze_docking_results,
)

from huxley.tools.chemistry.literature import (
    search_arxiv,
    search_pubmed,
    get_paper_details,
)

__all__ = [
    # Molecule tools
    "validate_smiles",
    "calculate_properties", 
    "predict_druglikeness",
    "modify_molecule",
    "generate_analogs",
    "design_molecule_for_target",
    "smiles_to_3d",
    # Docking tools
    "prepare_ligand",
    "dock_molecule",
    "analyze_docking_results",
    # Literature tools
    "search_arxiv",
    "search_pubmed",
    "get_paper_details",
]
