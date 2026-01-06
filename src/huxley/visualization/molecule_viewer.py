"""
Huxley 3D Molecular Viewer.

Generates interactive HTML visualizations of molecules, proteins, and DNA
using 3Dmol.js for WebGL-based rendering.
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime


def generate_molecule_html(
    molecule_data: dict,
    title: str = "Molecular Structure",
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Generate an interactive HTML page for 3D molecule visualization.
    
    Args:
        molecule_data: Dictionary containing molecule information
            - pdb_id: PDB identifier (optional)
            - smiles: SMILES string (optional)
            - atoms: List of atom coordinates (optional)
            - name: Molecule name
        title: Page title
        width: Viewer width in pixels
        height: Viewer height in pixels
    
    Returns:
        HTML string with embedded 3Dmol.js viewer
    """
    pdb_id = molecule_data.get("pdb_id", "")
    name = molecule_data.get("name", "Unknown Molecule")
    smiles = molecule_data.get("smiles", "")
    
    # Build the viewer initialization script
    if pdb_id:
        load_script = f"""
            // Try PDB format first, then CIF format as fallback
            function loadStructure(format, url) {{
                document.getElementById('load-status').textContent = 'Fetching structure from RCSB PDB...';
                $.ajax({{
                    url: url,
                    success: function(data) {{
                        viewer.addModel(data, format);
                        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                        viewer.zoomTo();
                        viewer.render();
                        document.getElementById('load-status').textContent = 'Structure loaded successfully • Click and drag to rotate';
                        updateMetrics();
                    }},
                    error: function() {{
                        if (format === 'pdb') {{
                            // Fallback to mmCIF format
                            document.getElementById('load-status').textContent = 'Trying alternate format...';
                            loadStructure('cif', 'https://files.rcsb.org/download/{pdb_id}.cif');
                        }} else {{
                            document.getElementById('load-status').innerHTML = 
                                '<span style="color: #EF4444;">Structure unavailable. PDB ID may not exist or be released yet.</span>';
                            // Show placeholder
                            viewer.addSphere({{center: {{x:0, y:0, z:0}}, radius: 5, color: 'gray'}});
                            viewer.zoomTo();
                            viewer.render();
                        }}
                    }}
                }});
            }}
            loadStructure('pdb', 'https://files.rcsb.org/download/{pdb_id}.pdb');
        """
    elif smiles:
        load_script = f"""
            // Load from SMILES via translation service
            viewer.addModel();
            viewer.setStyle({{}}, {{stick: {{}}}});
            viewer.zoomTo();
            viewer.render();
            document.getElementById('info').innerHTML += '<br>SMILES: {smiles}';
        """
    else:
        load_script = """
            viewer.addModel();
            viewer.render();
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Huxley Structural Biology</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Crimson+Pro:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #A51C30;
            --primary-dark: #8A1829;
            --accent: #1E3A5F;
            --accent-light: #2B5278;
            --bg-primary: #FAFBFC;
            --bg-secondary: #FFFFFF;
            --bg-tertiary: #F4F5F7;
            --text-primary: #1A1A1A;
            --text-secondary: #4A5568;
            --text-muted: #718096;
            --border: #E2E8F0;
            --border-light: #EDF2F7;
            --success: #059669;
            --warning: #D97706;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.025);
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        /* Header */
        .header {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0;
        }}
        .header-top {{
            background: var(--primary);
            padding: 8px 0;
        }}
        .header-top-inner {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .institution {{
            color: white;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .header-main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 40px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .logo-section {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .logo {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Crimson Pro', Georgia, serif;
            font-size: 24px;
            font-weight: 600;
            color: white;
        }}
        .title-group h1 {{
            font-family: 'Crimson Pro', Georgia, serif;
            font-size: 28px;
            font-weight: 500;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }}
        .title-group .subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }}
        .header-meta {{
            text-align: right;
        }}
        .header-meta .date {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .header-meta .session {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            background: var(--bg-tertiary);
            padding: 4px 8px;
            border-radius: 4px;
            margin-top: 4px;
            display: inline-block;
        }}
        
        /* Main Content */
        .main-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 40px;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 32px;
        }}
        
        /* Viewer Panel */
        .viewer-panel {{
            background: var(--bg-secondary);
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .viewer-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-light);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .viewer-header h2 {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .viewer-header h2::before {{
            content: '';
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
        }}
        .viewer-badge {{
            font-size: 11px;
            font-weight: 500;
            color: var(--accent);
            background: rgba(30, 58, 95, 0.08);
            padding: 4px 10px;
            border-radius: 100px;
        }}
        #viewer {{
            width: 100%;
            height: {height}px;
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        }}
        .viewer-controls {{
            padding: 16px 20px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-light);
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .ctrl-btn {{
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 500;
            padding: 8px 16px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .ctrl-btn:hover {{
            border-color: var(--accent);
            color: var(--accent);
            background: rgba(30, 58, 95, 0.04);
        }}
        .ctrl-btn.active {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}
        .ctrl-divider {{
            width: 1px;
            background: var(--border);
            margin: 0 8px;
        }}
        .status-bar {{
            padding: 12px 20px;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-light);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-indicator {{
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        /* Info Panel */
        .info-panel {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .info-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border);
            overflow: hidden;
        }}
        .card-header {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-light);
            background: var(--bg-tertiary);
        }}
        .card-header h3 {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card-body {{
            padding: 18px;
        }}
        .data-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 10px 0;
            border-bottom: 1px solid var(--border-light);
        }}
        .data-row:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}
        .data-row:first-child {{
            padding-top: 0;
        }}
        .data-label {{
            font-size: 13px;
            color: var(--text-muted);
        }}
        .data-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
            font-weight: 500;
            text-align: right;
            max-width: 200px;
            word-break: break-all;
        }}
        .data-value.highlight {{
            color: var(--primary);
        }}
        
        /* Metrics Card */
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        .metric-box {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px;
            font-weight: 600;
            color: var(--accent);
        }}
        .metric-label {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        /* Legend Card */
        .legend-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 3px;
        }}
        
        /* Footer */
        .footer {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 40px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid var(--border);
        }}
        .footer-left {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .footer-right {{
            display: flex;
            gap: 24px;
        }}
        .footer-link {{
            font-size: 12px;
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.15s;
        }}
        .footer-link:hover {{
            color: var(--primary);
        }}
        
        /* Print Styles */
        @media print {{
            .viewer-controls, .ctrl-btn {{ display: none; }}
            .viewer-panel {{ box-shadow: none; border: 1px solid #ddd; }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-top">
            <div class="header-top-inner">
                <span class="institution">Huxley Computational Biology Laboratory</span>
                <span class="institution">Structural Analysis Division</span>
            </div>
        </div>
        <div class="header-main">
            <div class="logo-section">
                <div class="logo">H</div>
                <div class="title-group">
                    <h1>{title}</h1>
                    <div class="subtitle">Interactive Molecular Visualization</div>
                </div>
            </div>
            <div class="header-meta">
                <div class="date">{datetime.now().strftime("%B %d, %Y")}</div>
                <div class="session">PDB: {pdb_id or 'N/A'}</div>
            </div>
        </div>
    </header>
    
    <main class="main-content">
        <div class="viewer-panel">
            <div class="viewer-header">
                <h2>3D Structure Viewer</h2>
                <span class="viewer-badge">WebGL Accelerated</span>
            </div>
            <div id="viewer"></div>
            <div class="viewer-controls">
                <button class="ctrl-btn active" onclick="setStyle('cartoon', this)">Cartoon</button>
                <button class="ctrl-btn" onclick="setStyle('stick', this)">Ball & Stick</button>
                <button class="ctrl-btn" onclick="setStyle('sphere', this)">Space Fill</button>
                <button class="ctrl-btn" onclick="setStyle('surface', this)">Surface</button>
                <div class="ctrl-divider"></div>
                <button class="ctrl-btn" onclick="toggleSpin(this)">Auto-Rotate</button>
                <button class="ctrl-btn" onclick="resetView()">Reset View</button>
                <button class="ctrl-btn" onclick="toggleLabels(this)">Labels</button>
            </div>
            <div class="status-bar">
                <div class="status-indicator"></div>
                <span id="load-status">Initializing molecular renderer...</span>
            </div>
        </div>
        
        <aside class="info-panel">
            <div class="info-card">
                <div class="card-header">
                    <h3>Structure Information</h3>
                </div>
                <div class="card-body">
                    <div class="data-row">
                        <span class="data-label">Molecule Name</span>
                        <span class="data-value">{name}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">PDB Accession</span>
                        <span class="data-value highlight">{pdb_id or 'N/A'}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Data Source</span>
                        <span class="data-value">RCSB PDB</span>
                    </div>
                    <div id="extra-info"></div>
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-header">
                    <h3>Structural Metrics</h3>
                </div>
                <div class="card-body">
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-value" id="atom-count">--</div>
                            <div class="metric-label">Atoms</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="residue-count">--</div>
                            <div class="metric-label">Residues</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="chain-count">--</div>
                            <div class="metric-label">Chains</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="mol-weight">--</div>
                            <div class="metric-label">kDa</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-header">
                    <h3>Element Legend</h3>
                </div>
                <div class="card-body">
                    <div class="legend-grid">
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #EF4444;"></div>
                            <span>Oxygen (O)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #3B82F6;"></div>
                            <span>Nitrogen (N)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #FACC15;"></div>
                            <span>Sulfur (S)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #6B7280;"></div>
                            <span>Carbon (C)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #F97316;"></div>
                            <span>Phosphorus (P)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #10B981;"></div>
                            <span>Other</span>
                        </div>
                    </div>
                </div>
            </div>
        </aside>
    </main>
    
    <footer class="footer">
        <div class="footer-left">
            Generated by Huxley Biological Intelligence Framework v0.6.0 &nbsp;|&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
        </div>
        <div class="footer-right">
            <a href="https://www.rcsb.org/structure/{pdb_id}" class="footer-link" target="_blank">View on RCSB PDB →</a>
        </div>
    </footer>
    
    <script>
        let viewer = null;
        let spinning = false;
        let labelsVisible = false;
        let currentStyle = 'cartoon';
        
        $(document).ready(function() {{
            viewer = $3Dmol.createViewer('viewer', {{
                backgroundColor: '0x0F172A'
            }});
            
            {load_script}
        }});
        
        function updateMetrics() {{
            if (!viewer) return;
            let model = viewer.getModel();
            if (model) {{
                let atoms = model.selectedAtoms({{}});
                document.getElementById('atom-count').textContent = atoms.length.toLocaleString();
                
                let residues = new Set();
                let chains = new Set();
                atoms.forEach(a => {{
                    if (a.resi) residues.add(a.chain + '_' + a.resi);
                    if (a.chain) chains.add(a.chain);
                }});
                document.getElementById('residue-count').textContent = residues.size.toLocaleString();
                document.getElementById('chain-count').textContent = chains.size;
                
                // Estimate molecular weight (~110 Da per residue for proteins)
                let mw = (residues.size * 110 / 1000).toFixed(1);
                document.getElementById('mol-weight').textContent = mw;
            }}
        }}
        
        function setStyle(style, btn) {{
            document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            currentStyle = style;
            
            viewer.setStyle({{}}, {{}});
            switch(style) {{
                case 'cartoon':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    break;
                case 'stick':
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.25}}}});
                    break;
                case 'sphere':
                    viewer.setStyle({{}}, {{sphere: {{}}}});
                    break;
                case 'surface':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum', opacity: 0.8}}}});
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.85, color: 'white'}});
                    break;
            }}
            viewer.render();
        }}
        
        function toggleSpin(btn) {{
            spinning = !spinning;
            if (btn) btn.classList.toggle('active', spinning);
            viewer.spin(spinning ? 'y' : false, 0.5);
        }}
        
        function toggleLabels(btn) {{
            labelsVisible = !labelsVisible;
            if (btn) btn.classList.toggle('active', labelsVisible);
            if (labelsVisible) {{
                viewer.addResLabels({{backgroundOpacity: 0.8, fontColor: 'white', showBackground: true}});
            }} else {{
                viewer.removeAllLabels();
            }}
            viewer.render();
        }}
        
        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}
        
        // Update status when structure loads
        window.onStructureLoaded = function() {{
            document.getElementById('load-status').textContent = 'Structure loaded successfully • Click and drag to rotate';
            updateMetrics();
        }}
    </script>
</body>
</html>"""
    
    return html


def generate_protein_html(
    pdb_id: str | None = None,
    pdb_data: str | None = None,
    sequence: str | None = None,
    title: str = "Protein Structure",
    viability_score: float | None = None,
    metrics: dict | None = None,
) -> str:
    """
    Generate an interactive HTML page for protein visualization.
    
    Args:
        pdb_id: PDB identifier to fetch
        pdb_data: Raw PDB format string (if not using pdb_id)
        sequence: Amino acid sequence
        title: Page title
        viability_score: Optional viability/confidence score (0-1)
        metrics: Optional dictionary of metrics to display
    
    Returns:
        HTML string
    """
    metrics = metrics or {}
    
    if pdb_id:
        load_script = f"""
            function tryLoadProtein(format, url) {{
                $.ajax({{
                    url: url,
                    success: function(data) {{
                        loadStructure(data);
                    }},
                    error: function() {{
                        if (format === 'pdb') {{
                            tryLoadProtein('cif', 'https://files.rcsb.org/download/{pdb_id}.cif');
                        }} else {{
                            document.getElementById('viewer').innerHTML = 
                                '<div style="color:#F7768E;padding:20px;">Failed to load structure {pdb_id}</div>';
                        }}
                    }}
                }});
            }}
            tryLoadProtein('pdb', 'https://files.rcsb.org/download/{pdb_id}.pdb');
        """
    elif pdb_data:
        # Escape the PDB data for embedding
        escaped_pdb = json.dumps(pdb_data)
        load_script = f"""
            loadStructure({escaped_pdb});
        """
    else:
        load_script = """
            document.getElementById('viewer').innerHTML = 
                '<div style="color:#888;padding:20px;">No structure data provided</div>';
        """
    
    # Build metrics HTML
    metrics_html = ""
    for key, value in metrics.items():
        if isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        metrics_html += f"""
                <div class="item">
                    <span class="label">{key}</span>
                    <span class="value">{value_str}</span>
                </div>"""
    
    # Viability display
    if viability_score is not None:
        viability_percent = int(viability_score * 100)
        if viability_score >= 0.8:
            viability_color = "#9ECE6A"  # Green
            viability_text = "HIGH"
        elif viability_score >= 0.5:
            viability_color = "#E0AF68"  # Amber
            viability_text = "MODERATE"
        else:
            viability_color = "#F7768E"  # Red
            viability_text = "LOW"
    else:
        viability_percent = "--"
        viability_color = "#888"
        viability_text = "NOT ASSESSED"
    
    sequence_display = sequence[:50] + "..." if sequence and len(sequence) > 50 else (sequence or "N/A")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} - Huxley Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(92,200,255,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ color: #5CC8FF; font-size: 1.5em; font-weight: 500; letter-spacing: 2px; }}
        .header .subtitle {{ color: #7AA2F7; font-size: 0.9em; }}
        .container {{ display: flex; padding: 20px; gap: 20px; }}
        .viewer-container {{
            flex: 2;
            background: #0d1117;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        #viewer {{ width: 100%; height: 550px; }}
        .controls {{
            padding: 15px;
            background: rgba(0,0,0,0.4);
            border-top: 1px solid rgba(92,200,255,0.1);
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .controls button {{
            background: linear-gradient(135deg, #5CC8FF 0%, #7AA2F7 100%);
            border: none;
            color: #0d1117;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .controls button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(92,200,255,0.3); }}
        .controls button.secondary {{
            background: rgba(255,255,255,0.1);
            color: #e0e0e0;
        }}
        .info-panel {{ flex: 1; min-width: 320px; }}
        .info-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(92,200,255,0.1);
        }}
        .info-card h3 {{
            color: #5CC8FF;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }}
        .info-card .item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-card .item:last-child {{ border-bottom: none; }}
        .info-card .label {{ color: #888; }}
        .info-card .value {{ color: #9ECE6A; font-family: 'SF Mono', Consolas, monospace; }}
        .viability {{ text-align: center; padding: 30px; }}
        .viability-score {{
            font-size: 3em;
            font-weight: bold;
            color: {viability_color};
        }}
        .viability-label {{ color: #888; margin-top: 5px; }}
        .viability-status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(0,0,0,0.3);
            color: {viability_color};
            font-size: 0.85em;
            margin-top: 10px;
            letter-spacing: 1px;
        }}
        .sequence-box {{
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 8px;
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.8em;
            word-break: break-all;
            color: #7AA2F7;
        }}
        .ss-legend {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .ss-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }}
        .ss-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>HUXLEY</h1>
            <div class="subtitle">Protein Structure Visualization</div>
        </div>
        <div style="text-align:right;">
            <div style="color:#5CC8FF;font-size:1.2em;">{title}</div>
            <div style="color:#666;font-size:0.8em;">{pdb_id or 'Generated Structure'}</div>
        </div>
    </div>
    
    <div class="container">
        <div class="viewer-container">
            <div id="viewer"></div>
            <div class="controls">
                <button onclick="setStyle('cartoon')">Cartoon</button>
                <button onclick="setStyle('ribbon')">Ribbon</button>
                <button onclick="setStyle('stick')">Stick</button>
                <button onclick="setStyle('sphere')">Space Fill</button>
                <button onclick="setStyle('surface')">Surface</button>
                <button class="secondary" onclick="colorBy('spectrum')">Rainbow</button>
                <button class="secondary" onclick="colorBy('ss')">Secondary Structure</button>
                <button class="secondary" onclick="colorBy('chain')">By Chain</button>
                <button class="secondary" onclick="toggleSpin()">Spin</button>
                <button class="secondary" onclick="resetView()">Reset</button>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-card viability">
                <h3>Viability Assessment</h3>
                <div class="viability-score">{viability_percent}%</div>
                <div class="viability-label">Estimated Viability</div>
                <div class="viability-status">{viability_text}</div>
            </div>
            
            <div class="info-card">
                <h3>Structure Metrics</h3>
                {metrics_html or '<div class="item"><span class="label">No metrics available</span></div>'}
            </div>
            
            <div class="info-card">
                <h3>Sequence</h3>
                <div class="sequence-box">{sequence_display}</div>
            </div>
            
            <div class="info-card">
                <h3>Secondary Structure Legend</h3>
                <div class="ss-legend">
                    <div class="ss-item">
                        <div class="ss-color" style="background:#FF6B6B;"></div>
                        <span>Alpha Helix</span>
                    </div>
                    <div class="ss-item">
                        <div class="ss-color" style="background:#4ECDC4;"></div>
                        <span>Beta Sheet</span>
                    </div>
                    <div class="ss-item">
                        <div class="ss-color" style="background:#95E1D3;"></div>
                        <span>Loop/Coil</span>
                    </div>
                    <div class="ss-item">
                        <div class="ss-color" style="background:#FFE66D;"></div>
                        <span>Turn</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Generated by Huxley Biological Intelligence Framework | {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
    
    <script>
        let viewer = null;
        let spinning = false;
        
        function loadStructure(data) {{
            viewer.addModel(data, 'pdb');
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.zoomTo();
            viewer.render();
        }}
        
        $(document).ready(function() {{
            viewer = $3Dmol.createViewer('viewer', {{
                backgroundColor: '0x0d1117'
            }});
            {load_script}
        }});
        
        function setStyle(style) {{
            viewer.setStyle({{}}, {{}});
            viewer.removeAllSurfaces();
            switch(style) {{
                case 'cartoon':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    break;
                case 'ribbon':
                    viewer.setStyle({{}}, {{cartoon: {{style: 'ribbon', color: 'spectrum'}}}});
                    break;
                case 'stick':
                    viewer.setStyle({{}}, {{stick: {{}}}});
                    break;
                case 'sphere':
                    viewer.setStyle({{}}, {{sphere: {{scale: 0.25}}}});
                    break;
                case 'surface':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum', opacity: 0.5}}}});
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.8, color: 'white'}});
                    break;
            }}
            viewer.render();
        }}
        
        function colorBy(scheme) {{
            viewer.setStyle({{}}, {{}});
            if (scheme === 'spectrum') {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            }} else if (scheme === 'ss') {{
                viewer.setStyle({{}}, {{cartoon: {{colorfunc: function(atom) {{
                    if (atom.ss === 'h') return '#FF6B6B';
                    if (atom.ss === 's') return '#4ECDC4';
                    return '#95E1D3';
                }}}}}});
            }} else if (scheme === 'chain') {{
                viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'chain'}}}});
            }}
            viewer.render();
        }}
        
        function toggleSpin() {{
            spinning = !spinning;
            viewer.spin(spinning ? 'y' : false, 1);
        }}
        
        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}
    </script>
</body>
</html>"""
    
    return html


def generate_dna_html(
    sequence: str,
    title: str = "DNA Structure",
    complementary: bool = True,
) -> str:
    """
    Generate an interactive HTML page for DNA visualization.
    
    Args:
        sequence: DNA sequence (A, T, G, C)
        title: Page title
        complementary: Whether to show complementary strand
    
    Returns:
        HTML string
    """
    # Generate complementary sequence
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    comp_seq = ''.join(complement_map.get(b, 'N') for b in sequence.upper())
    
    # Count bases
    base_counts = {
        'A': sequence.upper().count('A'),
        'T': sequence.upper().count('T'),
        'G': sequence.upper().count('G'),
        'C': sequence.upper().count('C'),
    }
    gc_content = (base_counts['G'] + base_counts['C']) / len(sequence) * 100 if sequence else 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} - Huxley DNA Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(92,200,255,0.2);
        }}
        .header h1 {{ color: #5CC8FF; font-size: 1.5em; font-weight: 500; letter-spacing: 2px; }}
        .container {{ display: flex; padding: 20px; gap: 20px; }}
        .main-panel {{ flex: 2; }}
        .viewer-container {{
            background: #0d1117;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}
        #viewer {{ width: 100%; height: 400px; }}
        .sequence-display {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(92,200,255,0.1);
        }}
        .sequence-display h3 {{
            color: #5CC8FF;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }}
        .dna-helix {{
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 14px;
            line-height: 2;
        }}
        .strand {{ margin: 10px 0; }}
        .strand-label {{ color: #888; display: inline-block; width: 30px; }}
        .base {{
            display: inline-block;
            width: 20px;
            height: 20px;
            text-align: center;
            line-height: 20px;
            border-radius: 4px;
            margin: 1px;
        }}
        .base-A {{ background: #FF6B6B; color: #fff; }}
        .base-T {{ background: #4ECDC4; color: #fff; }}
        .base-G {{ background: #FFE66D; color: #333; }}
        .base-C {{ background: #95E1D3; color: #333; }}
        .info-panel {{ flex: 1; min-width: 280px; }}
        .info-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(92,200,255,0.1);
        }}
        .info-card h3 {{
            color: #5CC8FF;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }}
        .info-card .item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-card .item:last-child {{ border-bottom: none; }}
        .info-card .label {{ color: #888; }}
        .info-card .value {{ color: #9ECE6A; font-family: monospace; }}
        .gc-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .gc-fill {{
            height: 100%;
            background: linear-gradient(90deg, #5CC8FF, #9ECE6A);
            border-radius: 4px;
            transition: width 0.5s;
        }}
        .base-legend {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HUXLEY</h1>
        <div style="color:#7AA2F7;font-size:0.9em;">{title}</div>
    </div>
    
    <div class="container">
        <div class="main-panel">
            <div class="viewer-container">
                <div id="viewer"></div>
            </div>
            
            <div class="sequence-display">
                <h3>Sequence ({len(sequence)} bp)</h3>
                <div class="dna-helix">
                    <div class="strand">
                        <span class="strand-label">5'</span>
                        {''.join(f'<span class="base base-{b}">{b}</span>' for b in sequence[:60].upper())}
                        {'...' if len(sequence) > 60 else ''}
                        <span class="strand-label">3'</span>
                    </div>
                    <div class="strand">
                        <span class="strand-label">3'</span>
                        {''.join(f'<span class="base base-{b}">{b}</span>' for b in comp_seq[:60])}
                        {'...' if len(comp_seq) > 60 else ''}
                        <span class="strand-label">5'</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-card">
                <h3>Sequence Properties</h3>
                <div class="item">
                    <span class="label">Length</span>
                    <span class="value">{len(sequence)} bp</span>
                </div>
                <div class="item">
                    <span class="label">GC Content</span>
                    <span class="value">{gc_content:.1f}%</span>
                </div>
                <div class="gc-bar">
                    <div class="gc-fill" style="width:{gc_content}%;"></div>
                </div>
            </div>
            
            <div class="info-card">
                <h3>Base Composition</h3>
                <div class="item">
                    <span class="label">Adenine (A)</span>
                    <span class="value">{base_counts['A']}</span>
                </div>
                <div class="item">
                    <span class="label">Thymine (T)</span>
                    <span class="value">{base_counts['T']}</span>
                </div>
                <div class="item">
                    <span class="label">Guanine (G)</span>
                    <span class="value">{base_counts['G']}</span>
                </div>
                <div class="item">
                    <span class="label">Cytosine (C)</span>
                    <span class="value">{base_counts['C']}</span>
                </div>
            </div>
            
            <div class="info-card">
                <h3>Base Legend</h3>
                <div class="base-legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FF6B6B;"></div>
                        <span>Adenine (A)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#4ECDC4;"></div>
                        <span>Thymine (T)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#FFE66D;"></div>
                        <span>Guanine (G)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:#95E1D3;"></div>
                        <span>Cytosine (C)</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Generated by Huxley Biological Intelligence Framework | {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
    
    <script>
        $(document).ready(function() {{
            let viewer = $3Dmol.createViewer('viewer', {{
                backgroundColor: '0x0d1117'
            }});
            
            // Load a sample DNA structure for visualization
            function tryLoadDNA(format, url) {{
                $.ajax({{
                    url: url,
                    success: function(data) {{
                        viewer.addModel(data, format);
                        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                        viewer.setStyle({{elem: 'P'}}, {{sphere: {{radius: 0.4, color: '#FF6B6B'}}}});
                        viewer.zoomTo();
                        viewer.spin('y', 0.5);
                        viewer.render();
                    }},
                    error: function() {{
                        if (format === 'pdb') {{
                            tryLoadDNA('cif', 'https://files.rcsb.org/download/1BNA.cif');
                        }}
                    }}
                }});
            }}
            tryLoadDNA('pdb', 'https://files.rcsb.org/download/1BNA.pdb');
        }});
    </script>
</body>
</html>"""
    
    return html


def generate_viability_report(
    results: list[dict],
    title: str = "Research Viability Report",
    goal: str = "",
) -> str:
    """
    Generate an HTML report summarizing viability analysis of multiple candidates.
    
    Args:
        results: List of candidate results with viability scores
        title: Report title
        goal: Research goal description
    
    Returns:
        HTML string
    """
    # Sort by viability
    sorted_results = sorted(results, key=lambda x: x.get("viability_score", 0), reverse=True)
    
    # Build results rows
    results_html = ""
    for i, result in enumerate(sorted_results, 1):
        score = result.get("viability_score", 0)
        name = result.get("name", f"Candidate {i}")
        details = result.get("details", "")
        
        if score >= 0.8:
            status_class = "high"
            status_text = "VIABLE"
        elif score >= 0.5:
            status_class = "medium"
            status_text = "POSSIBLE"
        else:
            status_class = "low"
            status_text = "UNLIKELY"
        
        results_html += f"""
        <div class="result-card">
            <div class="result-rank">#{i}</div>
            <div class="result-content">
                <div class="result-name">{name}</div>
                <div class="result-details">{details}</div>
            </div>
            <div class="result-score">
                <div class="score-value {status_class}">{int(score * 100)}%</div>
                <div class="score-label">{status_text}</div>
            </div>
        </div>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} - Huxley</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            color: #5CC8FF;
            font-size: 2em;
            font-weight: 500;
            letter-spacing: 3px;
        }}
        .header .subtitle {{
            color: #7AA2F7;
            margin-top: 10px;
        }}
        .goal-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(92,200,255,0.2);
            text-align: center;
        }}
        .goal-card h2 {{
            color: #5CC8FF;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }}
        .goal-text {{
            font-size: 1.2em;
            color: #fff;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(92,200,255,0.1);
        }}
        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #5CC8FF;
        }}
        .summary-label {{
            color: #888;
            margin-top: 5px;
            font-size: 0.85em;
        }}
        .results-section {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 30px;
            border: 1px solid rgba(92,200,255,0.1);
        }}
        .results-section h2 {{
            color: #5CC8FF;
            font-size: 0.85em;
            letter-spacing: 1px;
            margin-bottom: 20px;
        }}
        .result-card {{
            display: flex;
            align-items: center;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .result-rank {{
            font-size: 1.5em;
            font-weight: bold;
            color: #7AA2F7;
            width: 60px;
        }}
        .result-content {{
            flex: 1;
        }}
        .result-name {{
            font-size: 1.1em;
            color: #fff;
            margin-bottom: 5px;
        }}
        .result-details {{
            color: #888;
            font-size: 0.9em;
        }}
        .result-score {{
            text-align: center;
            min-width: 100px;
        }}
        .score-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .score-value.high {{ color: #9ECE6A; }}
        .score-value.medium {{ color: #E0AF68; }}
        .score-value.low {{ color: #F7768E; }}
        .score-label {{
            font-size: 0.75em;
            letter-spacing: 1px;
            color: #888;
        }}
        .footer {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HUXLEY</h1>
        <div class="subtitle">{title}</div>
    </div>
    
    <div class="goal-card">
        <h2>RESEARCH OBJECTIVE</h2>
        <div class="goal-text">{goal or 'No goal specified'}</div>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <div class="summary-value">{len(results)}</div>
            <div class="summary-label">Candidates Analyzed</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{len([r for r in results if r.get('viability_score', 0) >= 0.8])}</div>
            <div class="summary-label">Viable Solutions</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">{int(max([r.get('viability_score', 0) for r in results], default=0) * 100)}%</div>
            <div class="summary-label">Best Viability</div>
        </div>
    </div>
    
    <div class="results-section">
        <h2>RANKED RESULTS</h2>
        {results_html or '<div style="color:#888;text-align:center;padding:40px;">No results to display</div>'}
    </div>
    
    <div class="footer">
        Generated by Huxley Biological Intelligence Framework | {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
</body>
</html>"""
    
    return html


def save_visualization(html_content: str, output_path: Path | str) -> Path:
    """
    Save HTML visualization to file.
    
    Args:
        html_content: HTML string
        output_path: Path to save file
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path
