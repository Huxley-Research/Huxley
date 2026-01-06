"""
Huxley CLI - Generate command.

Generate protein structures from the command line.
"""

import asyncio
from pathlib import Path


async def run_generate(
    length: int = 100,
    description: str | None = None,
    output: str | None = None,
    samples: int = 1,
    seed: int | None = None,
):
    """Generate protein structures."""
    # Lazy imports to avoid tool registration at startup
    from huxley.cli.ui import (
        console, print_mini_banner, print_success, print_error,
        print_info, print_warning, print_protein_card, print_tool_call, rule
    )
    
    print_mini_banner()
    
    # Import here to avoid slow startup - this triggers tool registration
    from huxley import generate_protein_structure
    
    # Show execution parameters
    console.print("Generate Protein Structure")
    rule()
    console.print(f"{'Length:':<16}{length} residues")
    if description:
        console.print(f"{'Description:':<16}{description}")
    console.print(f"{'Samples:':<16}{samples}")
    if seed:
        console.print(f"{'Seed:':<16}{seed}")
    console.print()
    
    # Generate
    print_tool_call("generate_protein_structure", {
        "target_length": length,
        "num_samples": samples,
    })
    console.print()
    
    with console.status("Running SE(3) diffusion..."):
        result = await generate_protein_structure(
            target_length=length,
            num_samples=samples,
            conditioning_text=description,
            seed=seed,
        )
    
    if not result["success"]:
        from huxley.cli.ui import print_error_block
        print_error_block(
            "GenerationFailed",
            result.get("error", "Unknown error"),
            "Check model weights with 'huxley check' or re-run 'huxley setup'"
        )
        return
    
    # Show results
    print_success(f"Generated {result['num_generated']} structure(s)")
    
    for i, struct in enumerate(result["structures"]):
        # Print the protein card
        print_protein_card(
            structure_id=struct["id"],
            length=struct["length"],
            sequence=struct["sequence"],
            confidence=struct["confidence_score"],
            metrics=struct["metrics"],
        )
        
        # Save to file if requested
        if output:
            if samples > 1:
                # Multiple samples: add number suffix
                out_path = Path(output)
                out_file = out_path.parent / f"{out_path.stem}_{i+1}{out_path.suffix}"
            else:
                out_file = Path(output)
            
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(struct["pdb"])
            print_success(f"Saved: {out_file}")
    
    # Helpful next steps
    if not output:
        console.print()
        console.print("Tip: Use -o to save the PDB file")
        console.print(f"  huxley generate -l {length} -o protein.pdb")
