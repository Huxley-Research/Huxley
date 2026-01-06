"""
Real molecular chemistry tools using RDKit.

Provides property calculation, molecule manipulation, and design capabilities.
"""

import hashlib
from typing import Any
from datetime import datetime

# Try to import RDKit - gracefully handle if not installed
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        Lipinski,
        Crippen,
        rdMolDescriptors,
        Draw,
    )
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


def _check_rdkit():
    """Check if RDKit is available."""
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for chemistry tools. "
            "Install with: pip install rdkit"
        )


def validate_smiles(smiles: str) -> dict[str, Any]:
    """
    Validate a SMILES string and return canonical form.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Dict with validation results
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "error": "Invalid SMILES string",
            "input": smiles,
        }
    
    canonical = Chem.MolToSmiles(mol, canonical=True)
    
    return {
        "valid": True,
        "input": smiles,
        "canonical_smiles": canonical,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def calculate_properties(smiles: str) -> dict[str, Any]:
    """
    Calculate molecular properties from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dict with calculated properties
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    # Calculate all properties
    properties = {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
        "logP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "num_H_donors": Lipinski.NumHDonors(mol),
        "num_H_acceptors": Lipinski.NumHAcceptors(mol),
        "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_heavy_atoms": Lipinski.HeavyAtomCount(mol),
        "fraction_sp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "molar_refractivity": round(Crippen.MolMR(mol), 2),
    }
    
    # Add ring information
    ring_info = mol.GetRingInfo()
    properties["ring_sizes"] = [len(r) for r in ring_info.AtomRings()]
    
    return properties


def predict_druglikeness(smiles: str) -> dict[str, Any]:
    """
    Predict drug-likeness using multiple rules.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dict with drug-likeness predictions
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    # Calculate properties
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)
    
    # Lipinski's Rule of Five
    lipinski_violations = 0
    lipinski_details = {}
    
    lipinski_details["MW_ok"] = mw <= 500
    if not lipinski_details["MW_ok"]: lipinski_violations += 1
    
    lipinski_details["logP_ok"] = logp <= 5
    if not lipinski_details["logP_ok"]: lipinski_violations += 1
    
    lipinski_details["HBD_ok"] = hbd <= 5
    if not lipinski_details["HBD_ok"]: lipinski_violations += 1
    
    lipinski_details["HBA_ok"] = hba <= 10
    if not lipinski_details["HBA_ok"]: lipinski_violations += 1
    
    lipinski = {
        "violations": lipinski_violations,
        "compliant": lipinski_violations <= 1,
        "details": lipinski_details,
    }
    
    # Veber rules (oral bioavailability)
    veber = {
        "rotatable_bonds_ok": rotatable <= 10,
        "TPSA_ok": tpsa <= 140,
        "compliant": rotatable <= 10 and tpsa <= 140,
    }
    
    # Ghose filter
    ghose = {
        "MW_ok": 160 <= mw <= 480,
        "logP_ok": -0.4 <= logp <= 5.6,
        "atoms_ok": 20 <= mol.GetNumAtoms() <= 70,
        "molar_refractivity_ok": 40 <= Crippen.MolMR(mol) <= 130,
    }
    ghose["compliant"] = all(ghose.values())
    
    # Lead-likeness (Teague)
    lead_like = {
        "MW_ok": mw <= 350,
        "logP_ok": logp <= 3.5,
        "rotatable_ok": rotatable <= 7,
        "compliant": mw <= 350 and logp <= 3.5 and rotatable <= 7,
    }
    
    # PAINS filter (pan-assay interference compounds)
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        pains_matches = catalog.GetMatches(mol)
        pains_alert = len(pains_matches) > 0
        pains_alerts = [match.GetDescription() for match in pains_matches]
    except:
        pains_alert = False
        pains_alerts = []
    
    # Synthetic accessibility score (1-10, lower is easier)
    try:
        from rdkit.Chem import rdMolDescriptors
        sa_score = round(rdMolDescriptors.CalcNumRotatableBonds(mol) * 0.5 + 
                        rdMolDescriptors.CalcNumRings(mol) * 0.3 + 
                        len(Chem.FindMolChiralCenters(mol)) * 0.8 + 1, 1)
        sa_score = min(10, sa_score)
    except:
        sa_score = None
    
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "lipinski": lipinski,
        "veber": veber,
        "ghose": ghose,
        "lead_likeness": lead_like,
        "pains_alert": pains_alert,
        "pains_alerts": pains_alerts,
        "synthetic_accessibility": sa_score,
        "overall_druglike": lipinski["compliant"] and veber["compliant"] and not pains_alert,
    }


def modify_molecule(
    smiles: str,
    modification: str,
    position: int | None = None,
) -> dict[str, Any]:
    """
    Modify a molecule by adding/removing functional groups.
    
    Args:
        smiles: Input SMILES
        modification: Type of modification
        position: Optional atom position
        
    Returns:
        Dict with modified molecule
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    mol = Chem.RWMol(mol)
    
    modification_map = {
        "add_hydroxyl": ("O", "Added hydroxyl (-OH) group"),
        "add_amine": ("N", "Added amine (-NH2) group"),
        "add_fluorine": ("F", "Added fluorine for metabolic stability"),
        "add_chlorine": ("Cl", "Added chlorine"),
        "add_methyl": ("C", "Added methyl group"),
        "add_carboxyl": ("C(=O)O", "Added carboxylic acid"),
        "add_amide": ("C(=O)N", "Added amide group"),
        "add_sulfonyl": ("S(=O)(=O)O", "Added sulfonyl group"),
    }
    
    if modification not in modification_map and modification not in ["remove_group", "reduce_size"]:
        return {
            "error": f"Unknown modification: {modification}",
            "available": list(modification_map.keys()) + ["remove_group", "reduce_size"]
        }
    
    original_smiles = Chem.MolToSmiles(mol, canonical=True)
    
    if modification == "remove_group":
        # Remove a terminal heavy atom
        atoms_to_remove = []
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 1 and atom.GetSymbol() not in ['C', 'H']:
                atoms_to_remove.append(atom.GetIdx())
                break
        
        if atoms_to_remove:
            mol.RemoveAtom(atoms_to_remove[0])
            description = "Removed terminal functional group"
        else:
            return {"error": "No removable groups found"}
    
    elif modification == "reduce_size":
        # Remove a carbon from the longest chain
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetDegree() == 1:
                mol.RemoveAtom(atom.GetIdx())
                break
        description = "Reduced molecule size"
    
    else:
        # Add functional group
        group_smiles, description = modification_map[modification]
        
        # Find a suitable attachment point (preferably aromatic carbon)
        attach_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetNumImplicitHs() > 0:
                attach_idx = atom.GetIdx()
                if atom.GetIsAromatic():
                    break
        
        if attach_idx is None:
            return {"error": "No suitable attachment point found"}
        
        # Add the functional group
        group_mol = Chem.MolFromSmiles(group_smiles)
        if group_mol:
            combined = Chem.CombineMols(mol, group_mol)
            combined = Chem.RWMol(combined)
            # Connect them
            combined.AddBond(attach_idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
            mol = combined
    
    try:
        Chem.SanitizeMol(mol)
        new_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # Calculate property changes
        old_props = calculate_properties(original_smiles)
        new_props = calculate_properties(new_smiles)
        
        return {
            "success": True,
            "original_smiles": original_smiles,
            "modified_smiles": new_smiles,
            "modification": modification,
            "description": description,
            "property_changes": {
                "MW_change": round(new_props["molecular_weight"] - old_props["molecular_weight"], 1),
                "logP_change": round(new_props["logP"] - old_props["logP"], 2),
                "HBD_change": new_props["num_H_donors"] - old_props["num_H_donors"],
                "HBA_change": new_props["num_H_acceptors"] - old_props["num_H_acceptors"],
            },
            "new_properties": new_props,
        }
    except Exception as e:
        return {"error": f"Failed to modify molecule: {str(e)}"}


def generate_analogs(
    smiles: str,
    num_analogs: int = 5,
    strategy: str = "diverse",
) -> dict[str, Any]:
    """
    Generate structural analogs of a molecule.
    
    Args:
        smiles: Lead compound SMILES
        num_analogs: Number of analogs to generate
        strategy: Generation strategy (diverse, bioisostere, scaffold_hop)
        
    Returns:
        Dict with generated analogs
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    analogs = []
    lead_props = calculate_properties(smiles)
    
    # Bioisosteric replacements
    bioisosteres = [
        ("c1ccccc1", "c1ccncc1", "phenyl → pyridyl"),
        ("C(=O)O", "C(=O)NS(=O)(=O)", "carboxyl → acyl sulfonamide"),
        ("O", "S", "oxygen → sulfur"),
        ("F", "Cl", "fluorine → chlorine"),
        ("N", "O", "nitrogen → oxygen"),
        ("C", "N", "carbon → nitrogen"),
    ]
    
    modifications = [
        ("add_fluorine", "Fluorine scan"),
        ("add_hydroxyl", "Hydroxyl analog"),
        ("add_methyl", "Methyl analog"),
        ("add_amine", "Amine analog"),
        ("add_chlorine", "Chlorine analog"),
    ]
    
    generated = set()
    generated.add(Chem.MolToSmiles(mol, canonical=True))
    
    # Generate via modifications
    for mod_type, mod_name in modifications[:num_analogs]:
        result = modify_molecule(smiles, mod_type)
        if result.get("success") and result["modified_smiles"] not in generated:
            generated.add(result["modified_smiles"])
            analog_props = calculate_properties(result["modified_smiles"])
            druglike = predict_druglikeness(result["modified_smiles"])
            
            analogs.append({
                "analog_id": f"HUX-A{len(analogs)+1}-{hashlib.md5(result['modified_smiles'].encode()).hexdigest()[:4].upper()}",
                "smiles": result["modified_smiles"],
                "modification": mod_name,
                "molecular_weight": analog_props["molecular_weight"],
                "logP": analog_props["logP"],
                "druglike": druglike["overall_druglike"],
                "lipinski_violations": druglike["lipinski"]["violations"],
            })
        
        if len(analogs) >= num_analogs:
            break
    
    # Try bioisosteric replacements if we need more
    if len(analogs) < num_analogs:
        for old, new, desc in bioisosteres:
            if old in smiles:
                new_smiles = smiles.replace(old, new, 1)
                test_mol = Chem.MolFromSmiles(new_smiles)
                if test_mol and new_smiles not in generated:
                    generated.add(new_smiles)
                    analog_props = calculate_properties(new_smiles)
                    druglike = predict_druglikeness(new_smiles)
                    
                    analogs.append({
                        "analog_id": f"HUX-A{len(analogs)+1}-{hashlib.md5(new_smiles.encode()).hexdigest()[:4].upper()}",
                        "smiles": new_smiles,
                        "modification": desc,
                        "molecular_weight": analog_props["molecular_weight"],
                        "logP": analog_props["logP"],
                        "druglike": druglike["overall_druglike"],
                        "lipinski_violations": druglike["lipinski"]["violations"],
                    })
                
                if len(analogs) >= num_analogs:
                    break
    
    return {
        "success": True,
        "lead_smiles": smiles,
        "lead_properties": lead_props,
        "num_analogs": len(analogs),
        "analogs": analogs,
        "strategy": strategy,
    }


def design_molecule_for_target(
    target_name: str,
    scaffold: str | None = None,
    properties: str | None = None,
    constraints: dict | None = None,
) -> dict[str, Any]:
    """
    Design a novel molecule targeting a specific protein.
    
    Uses rule-based generation with common pharmacophores.
    
    Args:
        target_name: Target protein/receptor name
        scaffold: Optional starting scaffold SMILES
        properties: Desired property keywords
        constraints: Property constraints (e.g., {"MW": 500, "logP": 5})
        
    Returns:
        Dict with designed molecule
    """
    _check_rdkit()
    
    # Target-specific scaffolds based on known drug classes
    target_scaffolds = {
        "kinase": [
            "c1ccc2[nH]c(-c3ccccn3)nc2c1",  # Imidazopyridine
            "c1ccc2nc(-c3cccnc3)oc2c1",      # Benzoxazole-pyridine
            "c1cnc2ccccc2n1",                 # Quinazoline
        ],
        "protease": [
            "NC(CC(=O)O)C(=O)NC(Cc1ccccc1)C(=O)O",  # Peptidomimetic
            "c1ccc(C(=O)Nc2ccccn2)cc1",              # Benzamide-pyridine
            "c1ccc2[nH]cc(CC(N)C(=O)O)c2c1",        # Tryptophan derivative
        ],
        "gpcr": [
            "c1ccc(CCN2CCCCC2)cc1",           # Phenethylpiperidine
            "c1ccc(CN2CCN(c3ccccc3)CC2)cc1",  # Diphenylpiperazine
        ],
        "ion_channel": [
            "c1ccc2c(c1)CC(N)C2",             # Aminoindane
            "c1ccc(CCNCC2CCCCC2)cc1",         # Phenylcyclohexylamine
        ],
        "nuclear_receptor": [
            "CC12CCC3C(CCC4CC(O)CCC34)C1CCC2O",  # Steroid scaffold
        ],
    }
    
    # Determine target type
    target_lower = target_name.lower()
    target_type = "protease"  # Default
    
    for key in target_scaffolds:
        if key in target_lower:
            target_type = key
            break
    
    if "covid" in target_lower or "sars" in target_lower or "mpro" in target_lower:
        target_type = "protease"
    
    # Select or use provided scaffold
    if scaffold:
        base_mol = Chem.MolFromSmiles(scaffold)
        if base_mol is None:
            return {"error": f"Invalid scaffold SMILES: {scaffold}"}
        base_smiles = scaffold
    else:
        import random
        scaffolds = target_scaffolds.get(target_type, target_scaffolds["protease"])
        base_smiles = random.choice(scaffolds)
        base_mol = Chem.MolFromSmiles(base_smiles)
    
    # Apply modifications based on desired properties
    current_smiles = base_smiles
    
    if properties:
        props_lower = properties.lower()
        
        if "selectivity" in props_lower or "metabolic" in props_lower:
            result = modify_molecule(current_smiles, "add_fluorine")
            if result.get("success"):
                current_smiles = result["modified_smiles"]
        
        if "solubility" in props_lower or "oral" in props_lower:
            result = modify_molecule(current_smiles, "add_hydroxyl")
            if result.get("success"):
                current_smiles = result["modified_smiles"]
        
        if "potency" in props_lower or "binding" in props_lower:
            result = modify_molecule(current_smiles, "add_amide")
            if result.get("success"):
                current_smiles = result["modified_smiles"]
    
    # Validate final molecule
    final_mol = Chem.MolFromSmiles(current_smiles)
    if final_mol is None:
        return {"error": "Failed to generate valid molecule"}
    
    # Calculate properties
    final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
    mol_props = calculate_properties(final_smiles)
    druglike = predict_druglikeness(final_smiles)
    
    # Check constraints
    constraint_violations = []
    if constraints:
        if constraints.get("MW") and mol_props["molecular_weight"] > constraints["MW"]:
            constraint_violations.append(f"MW ({mol_props['molecular_weight']}) > {constraints['MW']}")
        if constraints.get("logP") and mol_props["logP"] > constraints["logP"]:
            constraint_violations.append(f"logP ({mol_props['logP']}) > {constraints['logP']}")
    
    mol_id = f"HUX-{hashlib.md5(f'{target_name}{final_smiles}'.encode()).hexdigest()[:8].upper()}"
    
    return {
        "success": True,
        "molecule_id": mol_id,
        "smiles": final_smiles,
        "target": target_name,
        "target_class": target_type,
        "scaffold_used": base_smiles,
        "properties": mol_props,
        "druglikeness": druglike,
        "constraint_violations": constraint_violations,
        "lipinski_compliant": druglike["lipinski"]["compliant"],
    }


def smiles_to_3d(smiles: str) -> dict[str, Any]:
    """
    Generate 3D coordinates for a molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dict with 3D structure (MOL block or PDB)
    """
    _check_rdkit()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result == -1:
        return {"error": "Failed to generate 3D coordinates"}
    
    # Optimize geometry
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Get MOL block
    mol_block = Chem.MolToMolBlock(mol)
    
    # Get PDB block
    pdb_block = Chem.MolToPDBBlock(mol)
    
    return {
        "success": True,
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True),
        "mol_block": mol_block,
        "pdb_block": pdb_block,
        "num_atoms": mol.GetNumAtoms(),
        "num_conformers": mol.GetNumConformers(),
    }
