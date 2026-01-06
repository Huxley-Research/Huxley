"""
RCSB PDB (Protein Data Bank) integration tools.

These tools provide access to the RCSB PDB Search and Data APIs for
retrieving biological molecule structure information.

API Documentation:
- Search API: https://search.rcsb.org/index.html
- Data API: https://data.rcsb.org/index.html
"""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import quote

import httpx

from huxley.tools.registry import tool


# API Base URLs
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_API = "https://data.rcsb.org/rest/v1/core"
RCSB_GRAPHQL_API = "https://data.rcsb.org/graphql"


async def _make_request(
    url: str,
    method: str = "GET",
    json_data: dict | None = None,
    params: dict | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Make an async HTTP request to RCSB APIs."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method == "GET":
            response = await client.get(url, params=params)
        else:
            response = await client.post(url, json=json_data)
        
        if response.status_code == 204:
            return {"total_count": 0, "result_set": []}
        
        response.raise_for_status()
        return response.json()


@tool(tags={"biology", "pdb", "search"})
async def pdb_search(
    query: str,
    return_type: Literal["entry", "polymer_entity", "assembly", "non_polymer_entity"] = "entry",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search the RCSB Protein Data Bank for structures matching a text query.
    
    Performs a full-text search across all PDB annotations including
    molecule names, organism names, authors, and experimental details.
    
    :param query: Search query text (e.g., "insulin", "kinase human", "COVID spike protein")
    :param return_type: Type of results - "entry" for PDB IDs, "polymer_entity" for molecules,
                       "assembly" for biological assemblies, "non_polymer_entity" for ligands
    :param max_results: Maximum number of results to return (default 10, max 100)
    """
    max_results = min(max_results, 100)
    
    search_request = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": query
            }
        },
        "return_type": return_type,
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "results_verbosity": "minimal"
        }
    }
    
    result = await _make_request(RCSB_SEARCH_API, method="POST", json_data=search_request)
    
    return {
        "total_count": result.get("total_count", 0),
        "results": [
            {
                "id": hit.get("identifier"),
                "score": hit.get("score")
            }
            for hit in result.get("result_set", [])
        ],
        "query": query,
        "return_type": return_type
    }


@tool(tags={"biology", "pdb", "search"})
async def pdb_sequence_search(
    sequence: str,
    sequence_type: Literal["protein", "dna", "rna"] = "protein",
    identity_cutoff: float = 0.9,
    evalue_cutoff: float = 1.0,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search PDB for structures with similar sequences using BLAST-like search.
    
    Finds protein or nucleic acid structures that share sequence similarity
    with the provided query sequence.
    
    :param sequence: Query sequence in single-letter amino acid or nucleotide codes
                    (e.g., "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
    :param sequence_type: Type of sequence - "protein", "dna", or "rna"
    :param identity_cutoff: Minimum sequence identity (0.0-1.0, default 0.9 = 90%)
    :param evalue_cutoff: Maximum E-value threshold (default 1.0)
    :param max_results: Maximum number of results to return (default 10)
    """
    max_results = min(max_results, 100)
    
    search_request = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "value": sequence.upper().replace(" ", "").replace("\n", ""),
                "sequence_type": sequence_type,
                "identity_cutoff": identity_cutoff,
                "evalue_cutoff": evalue_cutoff
            }
        },
        "return_type": "polymer_entity",
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }
    
    result = await _make_request(RCSB_SEARCH_API, method="POST", json_data=search_request)
    
    return {
        "total_count": result.get("total_count", 0),
        "results": [
            {
                "entity_id": hit.get("identifier"),
                "score": hit.get("score")
            }
            for hit in result.get("result_set", [])
        ],
        "sequence_type": sequence_type,
        "identity_cutoff": identity_cutoff
    }


@tool(tags={"biology", "pdb", "data"})
async def pdb_get_entry(pdb_id: str) -> dict[str, Any]:
    """
    Get detailed information about a PDB entry (structure).
    
    Returns comprehensive metadata including title, authors, experimental
    method, resolution, release date, and organism information.
    
    :param pdb_id: 4-character PDB identifier (e.g., "4HHB", "1CRN", "6LU7")
    """
    pdb_id = pdb_id.upper().strip()
    
    # Use GraphQL for richer data
    graphql_query = """
    query getEntry($id: String!) {
        entry(entry_id: $id) {
            rcsb_id
            struct {
                title
                pdbx_descriptor
            }
            exptl {
                method
            }
            rcsb_entry_info {
                resolution_combined
                molecular_weight
                deposited_polymer_entity_instance_count
                polymer_entity_count_protein
                polymer_entity_count_DNA
                polymer_entity_count_RNA
                nonpolymer_entity_count
                experimental_method
            }
            rcsb_accession_info {
                deposit_date
                initial_release_date
            }
            audit_author {
                name
            }
            rcsb_primary_citation {
                title
                journal_abbrev
                year
                pdbx_database_id_DOI
                pdbx_database_id_PubMed
            }
            polymer_entities {
                rcsb_id
                rcsb_polymer_entity {
                    pdbx_description
                }
                rcsb_entity_source_organism {
                    ncbi_scientific_name
                    ncbi_taxonomy_id
                }
                entity_poly {
                    pdbx_seq_one_letter_code_can
                    type
                }
            }
        }
    }
    """
    
    result = await _make_request(
        RCSB_GRAPHQL_API,
        method="POST",
        json_data={
            "query": graphql_query,
            "variables": {"id": pdb_id}
        }
    )
    
    entry = result.get("data", {}).get("entry")
    if not entry:
        return {"error": f"Entry {pdb_id} not found", "pdb_id": pdb_id}
    
    # Format the response
    return {
        "pdb_id": entry.get("rcsb_id"),
        "title": entry.get("struct", {}).get("title"),
        "description": entry.get("struct", {}).get("pdbx_descriptor"),
        "experimental_method": entry.get("rcsb_entry_info", {}).get("experimental_method"),
        "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined"),
        "molecular_weight_kda": entry.get("rcsb_entry_info", {}).get("molecular_weight"),
        "deposit_date": entry.get("rcsb_accession_info", {}).get("deposit_date"),
        "release_date": entry.get("rcsb_accession_info", {}).get("initial_release_date"),
        "authors": [a.get("name") for a in entry.get("audit_author", [])],
        "citation": {
            "title": entry.get("rcsb_primary_citation", {}).get("title"),
            "journal": entry.get("rcsb_primary_citation", {}).get("journal_abbrev"),
            "year": entry.get("rcsb_primary_citation", {}).get("year"),
            "doi": entry.get("rcsb_primary_citation", {}).get("pdbx_database_id_DOI"),
            "pubmed_id": entry.get("rcsb_primary_citation", {}).get("pdbx_database_id_PubMed"),
        } if entry.get("rcsb_primary_citation") else None,
        "entity_counts": {
            "protein": entry.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
            "dna": entry.get("rcsb_entry_info", {}).get("polymer_entity_count_DNA", 0),
            "rna": entry.get("rcsb_entry_info", {}).get("polymer_entity_count_RNA", 0),
            "ligands": entry.get("rcsb_entry_info", {}).get("nonpolymer_entity_count", 0),
        },
        "entities": [
            {
                "entity_id": e.get("rcsb_id"),
                "description": e.get("rcsb_polymer_entity", {}).get("pdbx_description"),
                "type": e.get("entity_poly", {}).get("type"),
                "organism": e.get("rcsb_entity_source_organism", [{}])[0].get("ncbi_scientific_name")
                    if e.get("rcsb_entity_source_organism") else None,
                "sequence_length": len(e.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "") or "")
            }
            for e in entry.get("polymer_entities", [])
        ],
        "rcsb_url": f"https://www.rcsb.org/structure/{pdb_id}"
    }


@tool(tags={"biology", "pdb", "data"})
async def pdb_get_entity(
    pdb_id: str,
    entity_id: str = "1",
) -> dict[str, Any]:
    """
    Get detailed information about a specific polymer entity (molecule) in a PDB structure.
    
    Returns sequence, organism source, UniProt mappings, and structural annotations.
    
    :param pdb_id: 4-character PDB identifier (e.g., "4HHB")
    :param entity_id: Entity number within the structure (default "1", usually the main chain)
    """
    pdb_id = pdb_id.upper().strip()
    
    graphql_query = """
    query getEntity($pdb_id: String!, $entity_id: String!) {
        polymer_entity(entry_id: $pdb_id, entity_id: $entity_id) {
            rcsb_id
            rcsb_polymer_entity {
                pdbx_description
                formula_weight
                rcsb_enzyme_class_combined {
                    ec
                }
            }
            entity_poly {
                pdbx_seq_one_letter_code_can
                pdbx_strand_id
                type
                rcsb_sample_sequence_length
            }
            rcsb_entity_source_organism {
                ncbi_scientific_name
                ncbi_taxonomy_id
                scientific_name
            }
            rcsb_polymer_entity_container_identifiers {
                reference_sequence_identifiers {
                    database_name
                    database_accession
                }
            }
            rcsb_cluster_membership {
                cluster_id
                identity
            }
            uniprots {
                rcsb_id
                rcsb_uniprot_protein {
                    name {
                        value
                    }
                }
            }
        }
    }
    """
    
    result = await _make_request(
        RCSB_GRAPHQL_API,
        method="POST",
        json_data={
            "query": graphql_query,
            "variables": {"pdb_id": pdb_id, "entity_id": entity_id}
        }
    )
    
    entity = result.get("data", {}).get("polymer_entity")
    if not entity:
        return {"error": f"Entity {pdb_id}_{entity_id} not found", "pdb_id": pdb_id, "entity_id": entity_id}
    
    # Extract reference identifiers
    ref_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers", [])
    uniprot_ids = [r.get("database_accession") for r in ref_ids if r.get("database_name") == "UniProt"]
    genbank_ids = [r.get("database_accession") for r in ref_ids if r.get("database_name") == "GenBank"]
    
    return {
        "entity_id": entity.get("rcsb_id"),
        "description": entity.get("rcsb_polymer_entity", {}).get("pdbx_description"),
        "type": entity.get("entity_poly", {}).get("type"),
        "sequence": entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can"),
        "sequence_length": entity.get("entity_poly", {}).get("rcsb_sample_sequence_length"),
        "chain_ids": entity.get("entity_poly", {}).get("pdbx_strand_id"),
        "molecular_weight_da": entity.get("rcsb_polymer_entity", {}).get("formula_weight"),
        "organism": {
            "scientific_name": entity.get("rcsb_entity_source_organism", [{}])[0].get("ncbi_scientific_name")
                if entity.get("rcsb_entity_source_organism") else None,
            "taxonomy_id": entity.get("rcsb_entity_source_organism", [{}])[0].get("ncbi_taxonomy_id")
                if entity.get("rcsb_entity_source_organism") else None,
        },
        "enzyme_classification": [
            ec.get("ec") for ec in (entity.get("rcsb_polymer_entity", {}).get("rcsb_enzyme_class_combined") or [])
        ],
        "uniprot_ids": uniprot_ids,
        "genbank_ids": genbank_ids,
        "sequence_clusters": [
            {"identity": c.get("identity"), "cluster_id": c.get("cluster_id")}
            for c in (entity.get("rcsb_cluster_membership") or [])
        ],
        "uniprot_annotations": [
            {
                "uniprot_id": u.get("rcsb_id"),
                "protein_name": u.get("rcsb_uniprot_protein", {}).get("name", {}).get("value")
            }
            for u in (entity.get("uniprots") or [])
        ]
    }


@tool(tags={"biology", "pdb", "data"})
async def pdb_get_assembly(
    pdb_id: str,
    assembly_id: str = "1",
) -> dict[str, Any]:
    """
    Get information about a biological assembly in a PDB structure.
    
    Biological assemblies represent the functional form of the molecule,
    which may differ from the asymmetric unit in the crystal.
    
    :param pdb_id: 4-character PDB identifier (e.g., "4HHB")
    :param assembly_id: Assembly number (default "1" for the first/preferred assembly)
    """
    pdb_id = pdb_id.upper().strip()
    
    graphql_query = """
    query getAssembly($assembly_id: String!) {
        assembly(assembly_id: $assembly_id) {
            rcsb_id
            rcsb_assembly_info {
                entry_id
                assembly_id
                polymer_entity_instance_count
                polymer_composition
            }
            rcsb_struct_symmetry {
                kind
                symbol
                oligomeric_state
                type
            }
            rcsb_assembly_container_identifiers {
                entry_id
                assembly_id
            }
        }
    }
    """
    
    assembly_full_id = f"{pdb_id}-{assembly_id}"
    
    result = await _make_request(
        RCSB_GRAPHQL_API,
        method="POST",
        json_data={
            "query": graphql_query,
            "variables": {"assembly_id": assembly_full_id}
        }
    )
    
    assembly = result.get("data", {}).get("assembly")
    if not assembly:
        return {"error": f"Assembly {assembly_full_id} not found", "pdb_id": pdb_id, "assembly_id": assembly_id}
    
    symmetry = assembly.get("rcsb_struct_symmetry", [{}])[0] if assembly.get("rcsb_struct_symmetry") else {}
    
    return {
        "assembly_id": assembly.get("rcsb_id"),
        "pdb_id": assembly.get("rcsb_assembly_info", {}).get("entry_id"),
        "chain_count": assembly.get("rcsb_assembly_info", {}).get("polymer_entity_instance_count"),
        "composition": assembly.get("rcsb_assembly_info", {}).get("polymer_composition"),
        "symmetry": {
            "kind": symmetry.get("kind"),
            "symbol": symmetry.get("symbol"),
            "oligomeric_state": symmetry.get("oligomeric_state"),
            "type": symmetry.get("type"),
        } if symmetry else None,
    }


@tool(tags={"biology", "pdb", "data"})
async def pdb_structure_summary(
    pdb_id: str,
) -> dict[str, Any]:
    """
    Get a concise summary of a PDB structure for quick reference.
    
    Returns key facts about the structure including what molecule it is,
    where it's from, how it was determined, and its quality.
    
    :param pdb_id: 4-character PDB identifier (e.g., "4HHB", "1CRN", "6LU7")
    """
    # Get full entry data
    entry_data = await pdb_get_entry(pdb_id)
    
    if "error" in entry_data:
        return entry_data
    
    # Create a human-readable summary
    entities_desc = []
    for e in entry_data.get("entities", []):
        desc = e.get("description", "Unknown")
        org = e.get("organism")
        if org:
            entities_desc.append(f"{desc} from {org}")
        else:
            entities_desc.append(desc)
    
    summary_text = f"""
**{entry_data.get('pdb_id')}**: {entry_data.get('title')}

**Molecules**: {'; '.join(entities_desc[:3])}{'...' if len(entities_desc) > 3 else ''}

**Method**: {entry_data.get('experimental_method')}
**Resolution**: {entry_data.get('resolution', 'N/A')} Å
**Released**: {entry_data.get('release_date', 'Unknown')}

**Contents**: {entry_data.get('entity_counts', {}).get('protein', 0)} protein(s), \
{entry_data.get('entity_counts', {}).get('ligands', 0)} ligand(s)

**Link**: {entry_data.get('rcsb_url')}
""".strip()
    
    return {
        "pdb_id": entry_data.get("pdb_id"),
        "summary": summary_text,
        "title": entry_data.get("title"),
        "molecules": entities_desc,
        "method": entry_data.get("experimental_method"),
        "resolution": entry_data.get("resolution"),
        "release_date": entry_data.get("release_date"),
        "url": entry_data.get("rcsb_url")
    }


@tool(tags={"biology", "pdb", "chemistry"})
async def pdb_ligand_info(
    ligand_id: str,
) -> dict[str, Any]:
    """
    Get information about a chemical component (ligand/small molecule) in PDB.
    
    Returns chemical descriptors, formula, synonyms, and structural information
    for ligands, cofactors, ions, and modified residues.
    
    :param ligand_id: 3-character chemical component ID (e.g., "ATP", "HEM", "NAD", "ZN")
    """
    ligand_id = ligand_id.upper().strip()
    
    graphql_query = """
    query getLigand($comp_id: String!) {
        chem_comp(comp_id: $comp_id) {
            rcsb_id
            chem_comp {
                id
                name
                type
                formula
                formula_weight
                pdbx_formal_charge
            }
            rcsb_chem_comp_info {
                initial_release_date
            }
            rcsb_chem_comp_descriptor {
                SMILES
                SMILES_stereo
                InChI
                InChIKey
            }
            rcsb_chem_comp_synonyms {
                name
                provenance_source
            }
            drugbank {
                drugbank_id
                name
                description
                mechanism_of_action
                drug_categories
                drug_groups
            }
        }
    }
    """
    
    result = await _make_request(
        RCSB_GRAPHQL_API,
        method="POST",
        json_data={
            "query": graphql_query,
            "variables": {"comp_id": ligand_id}
        }
    )
    
    chem = result.get("data", {}).get("chem_comp")
    if not chem:
        return {"error": f"Ligand {ligand_id} not found", "ligand_id": ligand_id}
    
    chem_comp = chem.get("chem_comp", {})
    descriptors = chem.get("rcsb_chem_comp_descriptor", {}) or {}
    drugbank = chem.get("drugbank", {})
    
    return {
        "ligand_id": chem.get("rcsb_id"),
        "name": chem_comp.get("name"),
        "type": chem_comp.get("type"),
        "formula": chem_comp.get("formula"),
        "molecular_weight": chem_comp.get("formula_weight"),
        "formal_charge": chem_comp.get("pdbx_formal_charge"),
        "descriptors": {
            "smiles": descriptors.get("SMILES"),
            "smiles_stereo": descriptors.get("SMILES_stereo"),
            "inchi": descriptors.get("InChI"),
            "inchi_key": descriptors.get("InChIKey"),
        },
        "synonyms": [
            s.get("name") for s in (chem.get("rcsb_chem_comp_synonyms") or [])
        ][:10],  # Limit synonyms
        "drugbank": {
            "id": drugbank.get("drugbank_id"),
            "name": drugbank.get("name"),
            "description": drugbank.get("description"),
            "mechanism": drugbank.get("mechanism_of_action"),
            "categories": drugbank.get("drug_categories"),
            "groups": drugbank.get("drug_groups"),
        } if drugbank else None,
        "rcsb_url": f"https://www.rcsb.org/ligand/{ligand_id}"
    }


@tool(tags={"biology", "pdb", "search"})
async def pdb_advanced_search(
    organism: str | None = None,
    experimental_method: Literal["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR", "SOLID-STATE NMR"] | None = None,
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    has_ligand: str | None = None,
    released_after: str | None = None,
    text_query: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Perform an advanced search of PDB with multiple filters.
    
    Combines multiple search criteria to find structures matching
    specific requirements for organism, method, quality, and content.
    
    :param organism: Scientific name of source organism (e.g., "Homo sapiens", "Escherichia coli")
    :param experimental_method: Structure determination method
    :param min_resolution: Minimum resolution in Angstroms (lower is better)
    :param max_resolution: Maximum resolution in Angstroms
    :param has_ligand: 3-letter ligand code that must be present (e.g., "ATP", "HEM")
    :param released_after: Only structures released after this date (YYYY-MM-DD format)
    :param text_query: Additional full-text search terms
    :param max_results: Maximum number of results (default 10)
    """
    max_results = min(max_results, 100)
    nodes = []
    
    # Build query nodes
    if text_query:
        nodes.append({
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": text_query}
        })
    
    if organism:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                "operator": "exact_match",
                "value": organism
            }
        })
    
    if experimental_method:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": experimental_method
            }
        })
    
    if min_resolution is not None or max_resolution is not None:
        resolution_value = {}
        if min_resolution is not None:
            resolution_value["from"] = min_resolution
            resolution_value["include_lower"] = True
        if max_resolution is not None:
            resolution_value["to"] = max_resolution
            resolution_value["include_upper"] = True
        
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "range",
                "value": resolution_value
            }
        })
    
    if has_ligand:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                "operator": "exact_match",
                "value": has_ligand.upper()
            }
        })
    
    if released_after:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater",
                "value": released_after
            }
        })
    
    if not nodes:
        return {"error": "At least one search criterion must be provided"}
    
    # Build the query
    if len(nodes) == 1:
        query = nodes[0]
    else:
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes
        }
    
    search_request = {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "results_verbosity": "minimal"
        }
    }
    
    result = await _make_request(RCSB_SEARCH_API, method="POST", json_data=search_request)
    
    return {
        "total_count": result.get("total_count", 0),
        "results": [
            {
                "pdb_id": hit.get("identifier"),
                "score": hit.get("score")
            }
            for hit in result.get("result_set", [])
        ],
        "filters_applied": {
            "organism": organism,
            "method": experimental_method,
            "resolution_range": f"{min_resolution or '*'}-{max_resolution or '*'}" if min_resolution or max_resolution else None,
            "ligand": has_ligand,
            "released_after": released_after,
            "text": text_query
        }
    }
