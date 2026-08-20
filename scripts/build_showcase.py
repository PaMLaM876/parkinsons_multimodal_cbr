"""
Build the interactive showcase HTML with embedded real dataset analysis data.
Generates a self-contained HTML file that works offline (no server needed).
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def build_showcase_html(analysis_dir: str, output_path: str):
    """Build the complete showcase HTML with embedded data."""

    # Load analysis data
    speech_data = json.load(open(os.path.join(analysis_dir, "speech_analysis.json")))
    tele_data = json.load(open(os.path.join(analysis_dir, "telemonitoring_analysis.json")))

    # Compact JSON for embedding
    speech_json = json.dumps(speech_data, separators=(',', ':'))
    tele_json = json.dumps(tele_data, separators=(',', ':'))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson's Disease — Real Dataset Preprocessing Showcase</title>
    <meta name="description" content="Interactive showcase of multimodal preprocessing pipeline applied to real UCI Parkinson's datasets for clinical decision support.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #06080f;
            --bg-surface: #0c1120;
            --bg-card: rgba(14, 20, 38, 0.9);
            --bg-elevated: rgba(20, 28, 50, 0.95);
            --border: rgba(99, 102, 241, 0.12);
            --border-active: rgba(99, 102, 241, 0.35);
            --cyan: #22d3ee;
            --blue: #3b82f6;
            --indigo: #6366f1;
            --violet: #8b5cf6;
            --rose: #f43f5e;
            --emerald: #10b981;
            --amber: #f59e0b;
            --orange: #f97316;
            --text: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --radius: 16px;
            --radius-sm: 10px;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-base);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
        }}

        body::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 90% 80%, rgba(244,63,94,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 50% 40% at 50% 10%, rgba(34,211,238,0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }}

        /* ─── Header ─── */
        .top-bar {{
            position: sticky; top: 0; z-index: 100;
            background: rgba(6,8,15,0.85);
            backdrop-filter: blur(20px) saturate(1.2);
            border-bottom: 1px solid var(--border);
            padding: 0.9rem 2rem;
            display: flex; justify-content: space-between; align-items: center;
        }}

        .brand {{ display: flex; align-items: center; gap: 1rem; }}

        .brand-icon {{
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(135deg, var(--indigo), var(--cyan));
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem; font-weight: 800; color: #fff;
            box-shadow: 0 4px 20px rgba(99,102,241,0.3);
        }}

        .brand h1 {{
            font-size: 1.15rem; font-weight: 700;
            background: linear-gradient(135deg, #f1f5f9, #94a3b8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .brand p {{ font-size: 0.75rem; color: var(--text-muted); }}

        .header-badges {{ display: flex; gap: 0.5rem; align-items: center; }}
        .h-badge {{
            padding: 0.3rem 0.7rem; border-radius: 20px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
        }}
        .h-badge.real {{ background: rgba(16,185,129,0.15); color: var(--emerald); border: 1px solid rgba(16,185,129,0.3); }}
        .h-badge.uci {{ background: rgba(59,130,246,0.15); color: var(--blue); border: 1px solid rgba(59,130,246,0.3); }}

        /* ─── Main ─── */
        .main {{ max-width: 1520px; margin: 0 auto; padding: 1.5rem 1.5rem 3rem; position: relative; z-index: 1; }}

        /* ─── Hero ─── */
        .hero {{
            text-align: center; padding: 2.5rem 1rem 2rem;
            animation: fadeIn 0.8s ease-out;
        }}
        .hero h2 {{
            font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
            background: linear-gradient(135deg, #f1f5f9, var(--cyan), var(--indigo));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .hero p {{ color: var(--text-secondary); font-size: 1rem; max-width: 720px; margin: 0 auto; }}

        @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(12px); }} to {{ opacity:1; transform: translateY(0); }} }}

        /* ─── Stats Row ─── */
        .stats-row {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem; margin: 1.5rem 0;
        }}

        .stat-card {{
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 1.2rem;
            display: flex; align-items: center; gap: 1rem;
            transition: border-color 0.3s, transform 0.3s, box-shadow 0.3s;
        }}
        .stat-card:hover {{
            border-color: var(--border-active);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(99,102,241,0.1);
        }}

        .stat-icon {{
            width: 48px; height: 48px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem; flex-shrink: 0;
        }}
        .si-blue {{ background: rgba(59,130,246,0.12); color: var(--blue); }}
        .si-cyan {{ background: rgba(34,211,238,0.12); color: var(--cyan); }}
        .si-rose {{ background: rgba(244,63,94,0.12); color: var(--rose); }}
        .si-emerald {{ background: rgba(16,185,129,0.12); color: var(--emerald); }}
        .si-amber {{ background: rgba(245,158,11,0.12); color: var(--amber); }}
        .si-violet {{ background: rgba(139,92,246,0.12); color: var(--violet); }}

        .stat-text h4 {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); }}
        .stat-text .val {{ font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
        .stat-text .sub {{ font-size: 0.72rem; color: var(--text-muted); }}

        /* ─── Tabs ─── */
        .tab-bar {{
            display: flex; gap: 0.4rem; margin: 2rem 0 1rem;
            background: var(--bg-surface); padding: 0.35rem;
            border-radius: var(--radius-sm); border: 1px solid var(--border);
            width: fit-content;
        }}
        .tab-btn {{
            padding: 0.55rem 1.3rem; border: none; border-radius: 8px;
            background: transparent; color: var(--text-muted);
            font-family: inherit; font-size: 0.85rem; font-weight: 600;
            cursor: pointer; transition: all 0.25s;
        }}
        .tab-btn.active {{
            background: linear-gradient(135deg, var(--indigo), var(--blue));
            color: #fff; box-shadow: 0 4px 16px rgba(99,102,241,0.3);
        }}
        .tab-btn:hover:not(.active) {{ color: var(--text); background: rgba(255,255,255,0.04); }}

        .tab-panel {{ display: none; animation: fadeIn 0.4s ease-out; }}
        .tab-panel.active {{ display: block; }}

        /* ─── Section Title ─── */
        .section-title {{
            font-size: 1.15rem; font-weight: 700; margin: 1.8rem 0 1rem;
            display: flex; align-items: center; gap: 0.6rem;
        }}
        .section-title .dot {{ width: 8px; height: 8px; border-radius: 50%; }}

        /* ─── Cards ─── */
        .card {{
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 1.4rem;
            backdrop-filter: blur(10px);
            position: relative; overflow: hidden;
            transition: border-color 0.3s;
        }}
        .card:hover {{ border-color: var(--border-active); }}

        .card-head {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 1rem; padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .card-head h3 {{ font-size: 0.95rem; font-weight: 700; }}
        .card-head .badge {{
            font-size: 0.68rem; padding: 0.2rem 0.55rem; border-radius: 20px;
            font-weight: 600;
        }}

        /* ─── Grid Layouts ─── */
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}

        @media (max-width: 1024px) {{
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
        }}

        /* ─── Chart Canvas ─── */
        .chart-box {{
            background: rgba(0,0,0,0.25); border-radius: var(--radius-sm);
            border: 1px solid rgba(255,255,255,0.04); padding: 0.75rem;
            position: relative;
        }}
        .chart-box canvas {{ width: 100% !important; height: 260px !important; display: block; }}

        .chart-label {{
            font-size: 0.72rem; color: var(--text-muted); text-align: center;
            margin-top: 0.3rem; font-weight: 500;
        }}

        .legend {{ display: flex; gap: 1rem; justify-content: center; margin: 0.6rem 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.35rem; font-size: 0.75rem; color: var(--text-secondary); }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 3px; }}

        /* ─── Table ─── */
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
        .data-table th {{
            text-align: left; padding: 0.6rem 0.8rem;
            font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--text-muted); border-bottom: 1px solid var(--border);
            font-weight: 600; position: sticky; top: 0; background: var(--bg-card);
        }}
        .data-table td {{
            padding: 0.55rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.03);
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
            color: var(--text-secondary);
        }}
        .data-table tr:hover td {{ background: rgba(99,102,241,0.04); color: var(--text); }}

        .sig-high {{ color: var(--emerald); font-weight: 700; }}
        .sig-med {{ color: var(--amber); font-weight: 600; }}
        .sig-low {{ color: var(--text-muted); }}

        /* ─── Pipeline Flow ─── */
        .pipeline {{
            display: flex; align-items: center; gap: 0.4rem;
            flex-wrap: wrap; justify-content: center;
            padding: 1.5rem 1rem;
        }}
        .pipe-step {{
            background: var(--bg-elevated); border: 1px solid var(--border);
            border-radius: var(--radius-sm); padding: 0.8rem 1.2rem;
            text-align: center; min-width: 130px;
            transition: all 0.3s;
        }}
        .pipe-step:hover {{ border-color: var(--indigo); transform: translateY(-3px); box-shadow: 0 8px 24px rgba(99,102,241,0.15); }}
        .pipe-step .step-icon {{ font-size: 1.5rem; margin-bottom: 0.3rem; }}
        .pipe-step .step-title {{ font-size: 0.78rem; font-weight: 700; }}
        .pipe-step .step-sub {{ font-size: 0.68rem; color: var(--text-muted); }}
        .pipe-arrow {{ font-size: 1.2rem; color: var(--text-muted); }}

        /* ─── Correlation Heatmap ─── */
        .heatmap-wrap {{ overflow-x: auto; padding: 0.5rem; }}
        .heatmap {{ border-collapse: collapse; }}
        .heatmap td {{
            width: 36px; height: 36px; text-align: center; font-size: 0.6rem;
            font-family: 'JetBrains Mono', monospace; font-weight: 500;
            border: 1px solid rgba(255,255,255,0.03); position: relative;
            cursor: pointer; transition: transform 0.15s;
        }}
        .heatmap td:hover {{ transform: scale(1.3); z-index: 5; border-color: #fff; }}
        .heatmap th {{
            font-size: 0.6rem; padding: 0.3rem; color: var(--text-muted);
            font-weight: 500; max-width: 60px; word-break: break-all;
        }}
        .heatmap-tooltip {{
            position: fixed; background: var(--bg-elevated); border: 1px solid var(--border-active);
            border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.75rem;
            pointer-events: none; z-index: 200; display: none;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }}

        /* ─── Scroll Container ─── */
        .scroll-table-wrap {{ max-height: 400px; overflow-y: auto; border-radius: var(--radius-sm); }}
        .scroll-table-wrap::-webkit-scrollbar {{ width: 6px; }}
        .scroll-table-wrap::-webkit-scrollbar-track {{ background: transparent; }}
        .scroll-table-wrap::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.3); border-radius: 3px; }}

        /* ─── Code Block ─── */
        .code-block {{
            background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.06);
            border-radius: var(--radius-sm); padding: 1rem 1.2rem;
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
            color: var(--text-secondary); overflow-x: auto; line-height: 1.7;
            white-space: pre; tab-size: 4;
        }}
        .code-block .kw {{ color: var(--violet); }}
        .code-block .fn {{ color: var(--cyan); }}
        .code-block .str {{ color: var(--emerald); }}
        .code-block .cm {{ color: var(--text-muted); font-style: italic; }}
        .code-block .num {{ color: var(--amber); }}

        /* ─── Footer ─── */
        footer {{
            text-align: center; padding: 2rem 1rem;
            color: var(--text-muted); font-size: 0.78rem;
            border-top: 1px solid var(--border); margin-top: 2rem;
        }}
    </style>
</head>
<body>

<!-- ═══ Top Bar ═══ -->
<div class="top-bar">
    <div class="brand">
        <div class="brand-icon">PD</div>
        <div>
            <h1>Multimodal Parkinson's Disease CDSS</h1>
            <p>Real Dataset Preprocessing Showcase</p>
        </div>
    </div>
    <div class="header-badges">
        <span class="h-badge real">✓ Real Clinical Data</span>
        <span class="h-badge uci">UCI ML Repository</span>
    </div>
</div>

<!-- ═══ Main Content ═══ -->
<div class="main">

    <!-- Hero -->
    <div class="hero">
        <h2>Preprocessing Pipeline on Real Patient Data</h2>
        <p>Interactive exploration of acoustic biomarker preprocessing applied to 6,070 real clinical voice recordings from the UCI Machine Learning Repository.</p>
    </div>

    <!-- Stats Row -->
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-icon si-blue">🗂</div>
            <div class="stat-text">
                <h4>Datasets</h4>
                <div class="val">2</div>
                <div class="sub">UCI Parkinson's Speech + Telemonitoring</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon si-cyan">🧑‍⚕️</div>
            <div class="stat-text">
                <h4>Total Samples</h4>
                <div class="val" id="totalSamples">6,070</div>
                <div class="sub">Voice recordings analyzed</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon si-rose">🧬</div>
            <div class="stat-text">
                <h4>Unique Subjects</h4>
                <div class="val">73</div>
                <div class="sub">31 (Speech) + 42 (Telemonitoring)</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon si-emerald">📊</div>
            <div class="stat-text">
                <h4>Acoustic Features</h4>
                <div class="val">19</div>
                <div class="sub">Jitter, Shimmer, HNR, RPDE, DFA, PPE, …</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon si-amber">🎯</div>
            <div class="stat-text">
                <h4>Classification</h4>
                <div class="val">PD vs HC</div>
                <div class="sub">147 PD + 48 Healthy Controls</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon si-violet">📈</div>
            <div class="stat-text">
                <h4>Regression Target</h4>
                <div class="val">UPDRS</div>
                <div class="sub">Motor & Total UPDRS scores</div>
            </div>
        </div>
    </div>

    <!-- Pipeline Flow -->
    <div class="section-title"><div class="dot" style="background:var(--indigo)"></div> Preprocessing Pipeline Architecture</div>
    <div class="card">
        <div class="pipeline">
            <div class="pipe-step">
                <div class="step-icon">📥</div>
                <div class="step-title">Raw UCI Data</div>
                <div class="step-sub">CSV Download via API</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="step-icon">🔄</div>
                <div class="step-title">Schema Mapping</div>
                <div class="step-sub">Column name normalization</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="step-icon">🩹</div>
                <div class="step-title">Imputation</div>
                <div class="step-sub">Median / Mode fill</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="step-icon">📐</div>
                <div class="step-title">Robust Scaling</div>
                <div class="step-sub">Percentile clip + Z-score</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="step-icon">🧮</div>
                <div class="step-title">Feature Matrix</div>
                <div class="step-sub">NumPy float32 tensor</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="step-icon">⚡</div>
                <div class="step-title">PyTorch Dataset</div>
                <div class="step-sub">DataLoader ready</div>
            </div>
        </div>
    </div>

    <!-- Tabs -->
    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('speech')">🎤 Speech Dataset (PD vs HC)</button>
        <button class="tab-btn" onclick="switchTab('tele')">📈 Telemonitoring (UPDRS)</button>
        <button class="tab-btn" onclick="switchTab('transform')">🔬 Raw → Scaled Transform</button>
    </div>

    <!-- ═══════════════════════════════ TAB 1: SPEECH ═══════════════════════════════ -->
    <div class="tab-panel active" id="panel-speech">

        <div class="section-title"><div class="dot" style="background:var(--rose)"></div> Feature Distributions — PD vs Healthy Controls</div>
        <div class="grid-2" id="histogramGrid"></div>

        <div class="section-title"><div class="dot" style="background:var(--amber)"></div> Statistical Significance — PD vs HC (Welch's t-test)</div>
        <div class="card">
            <div class="scroll-table-wrap">
                <table class="data-table" id="sigTable">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>PD Mean ± SD</th>
                            <th>HC Mean ± SD</th>
                            <th>t-statistic</th>
                            <th>p-value</th>
                            <th>Cohen's d</th>
                            <th>Significance</th>
                        </tr>
                    </thead>
                    <tbody id="sigTableBody"></tbody>
                </table>
            </div>
        </div>

        <div class="section-title"><div class="dot" style="background:var(--cyan)"></div> Feature Correlation Heatmap</div>
        <div class="card">
            <div class="heatmap-wrap" id="heatmapContainer"></div>
        </div>

    </div>

    <!-- ═══════════════════════════════ TAB 2: TELEMONITORING ═══════════════════════════════ -->
    <div class="tab-panel" id="panel-tele">

        <div class="section-title"><div class="dot" style="background:var(--blue)"></div> Acoustic Features vs Motor UPDRS — Regression Analysis</div>
        <div class="grid-2" id="scatterGrid"></div>

        <div class="section-title"><div class="dot" style="background:var(--emerald)"></div> Per-Subject Summary (42 PD Patients)</div>
        <div class="card">
            <div class="scroll-table-wrap">
                <table class="data-table" id="subjectTable">
                    <thead>
                        <tr>
                            <th>Subject ID</th>
                            <th>Age</th>
                            <th>Sex</th>
                            <th>Recordings</th>
                            <th>Motor UPDRS (mean)</th>
                            <th>Total UPDRS (mean)</th>
                        </tr>
                    </thead>
                    <tbody id="subjectTableBody"></tbody>
                </table>
            </div>
        </div>

        <div class="section-title"><div class="dot" style="background:var(--violet)"></div> Telemonitoring Feature Correlation Matrix</div>
        <div class="card">
            <div class="heatmap-wrap" id="teleHeatmapContainer"></div>
        </div>
    </div>

    <!-- ═══════════════════════════════ TAB 3: TRANSFORM ═══════════════════════════════ -->
    <div class="tab-panel" id="panel-transform">

        <div class="section-title"><div class="dot" style="background:var(--emerald)"></div> Raw Values → Z-Score Scaled (First 5 Patients)</div>
        <div class="grid-2">
            <div class="card">
                <div class="card-head">
                    <h3>📄 Raw Feature Values</h3>
                    <span class="badge" style="background:rgba(244,63,94,0.15);color:var(--rose);border:1px solid rgba(244,63,94,0.3)">Before Preprocessing</span>
                </div>
                <div class="scroll-table-wrap">
                    <table class="data-table" id="rawTable"></table>
                </div>
            </div>
            <div class="card">
                <div class="card-head">
                    <h3>⚡ Z-Score Scaled Values</h3>
                    <span class="badge" style="background:rgba(16,185,129,0.15);color:var(--emerald);border:1px solid rgba(16,185,129,0.3)">After Preprocessing</span>
                </div>
                <div class="scroll-table-wrap">
                    <table class="data-table" id="scaledTable"></table>
                </div>
            </div>
        </div>

        <div class="section-title"><div class="dot" style="background:var(--indigo)"></div> Scaling Statistics Used</div>
        <div class="card">
            <div class="scroll-table-wrap">
                <table class="data-table" id="scalingStatsTable">
                    <thead>
                        <tr><th>Feature</th><th>Mean (μ)</th><th>Std Dev (σ)</th><th>Formula</th></tr>
                    </thead>
                    <tbody id="scalingStatsBody"></tbody>
                </table>
            </div>
        </div>

        <div class="section-title"><div class="dot" style="background:var(--violet)"></div> Preprocessing Code Excerpt</div>
        <div class="card">
            <div class="code-block"><span class="cm"># ClinicalTabularPreprocessor.transform() — Core scaling logic</span>

<span class="kw">for</span> col <span class="kw">in</span> self.NUMERICAL_FEATURES:
    stats = self.num_stats[col]
    series = pd.to_numeric(df[col], errors=<span class="str">"coerce"</span>).fillna(stats[<span class="str">"median"</span>])

    <span class="cm"># 1. Robust percentile clipping (remove outliers)</span>
    clipped = np.clip(series, stats[<span class="str">"p1"</span>], stats[<span class="str">"p99"</span>])

    <span class="cm"># 2. Z-score standardization (zero mean, unit variance)</span>
    scaled = (clipped - stats[<span class="str">"mean"</span>]) / stats[<span class="str">"std"</span>]

    feature_matrix.append(scaled)

<span class="cm"># 3. One-hot encode categorical features</span>
<span class="kw">for</span> col <span class="kw">in</span> self.CATEGORICAL_FEATURES:
    one_hot = np.<span class="fn">zeros</span>((N, n_cats), dtype=np.float32)
    <span class="kw">for</span> row_idx, val <span class="kw">in</span> <span class="fn">enumerate</span>(df[col]):
        cat_idx = mapping.get(str(val), missing_idx)
        one_hot[row_idx, cat_idx] = <span class="num">1.0</span>

X = np.<span class="fn">hstack</span>(feature_matrix).astype(np.float32)
<span class="cm"># Output shape: (N_patients, D_features) → Ready for Deep Tabular MLP</span></div>
        </div>
    </div>

</div>

<footer>
    Multimodal Case-Based Clinical Decision Support for Parkinson's Disease &nbsp;·&nbsp;
    Data: UCI ML Repository (Datasets 174 & 189) &nbsp;·&nbsp;
    Preprocessing Pipeline Showcase
</footer>

<div class="heatmap-tooltip" id="hmTooltip"></div>

<script>
// ═══════════════════════════════════════════════════════════════
// EMBEDDED REAL DATA
// ═══════════════════════════════════════════════════════════════
const SPEECH = {speech_json};
const TELE = {tele_json};

// ═══ Tab Switching ═══
function switchTab(tab) {{
    document.querySelectorAll('.tab-btn').forEach((b,i) => {{
        b.classList.toggle('active', (tab==='speech'&&i===0)||(tab==='tele'&&i===1)||(tab==='transform'&&i===2));
    }});
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + tab).classList.add('active');
}}

// ═══ Color Utils ═══
function corrColor(v) {{
    const abs = Math.abs(v);
    if (v > 0) return `rgba(34,211,238,${{abs*0.8+0.1}})`;
    return `rgba(244,63,94,${{abs*0.8+0.1}})`;
}}
function corrTextColor(v) {{ return Math.abs(v) > 0.6 ? '#fff' : 'rgba(255,255,255,0.7)'; }}

// ═══ Histogram Rendering (Canvas) ═══
function drawHistogram(canvas, pdData, hcData, featureName) {{
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = {{ t:20, r:15, b:35, l:45 }};
    const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);

    const pdEdges = pdData.bin_edges, pdCounts = pdData.counts;
    const hcEdges = hcData.bin_edges, hcCounts = hcData.counts;

    const allEdges = [...pdEdges, ...hcEdges];
    const allCounts = [...pdCounts, ...hcCounts];
    const minX = Math.min(...allEdges), maxX = Math.max(...allEdges);
    const maxY = Math.max(...allCounts) * 1.15;

    const xScale = v => pad.l + ((v - minX) / (maxX - minX)) * cw;
    const yScale = v => pad.t + ch - (v / maxY) * ch;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {{
        const y = pad.t + (ch / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    }}

    // Draw bars
    function drawBars(edges, counts, color, alpha) {{
        ctx.fillStyle = color.replace('1)', alpha + ')');
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.8;
        for (let i = 0; i < counts.length; i++) {{
            const x1 = xScale(edges[i]), x2 = xScale(edges[i + 1]);
            const y = yScale(counts[i]), h = yScale(0) - y;
            ctx.fillRect(x1, y, x2 - x1, h);
            ctx.strokeRect(x1, y, x2 - x1, h);
        }}
    }}

    drawBars(hcEdges, hcCounts, 'rgba(34,211,238,1)', 0.3);
    drawBars(pdEdges, pdCounts, 'rgba(244,63,94,1)', 0.35);

    // Axes
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {{
        const v = minX + (maxX - minX) * i / nTicks;
        ctx.fillText(v.toPrecision(3), xScale(v), H - pad.b + 15);
    }}

    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
        const v = Math.round(maxY * (4 - i) / 4);
        ctx.fillText(v, pad.l - 6, pad.t + (ch / 4) * i + 4);
    }}

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(featureName, pad.l + cw / 2, H - 3);
}}

// ═══ Scatter Plot Rendering (Canvas) ═══
function drawScatter(canvas, data, xLabel, yLabel) {{
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = {{ t:20, r:15, b:38, l:55 }};
    const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);

    const xs = data.x, ys = data.y;
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;

    const xScale = v => pad.l + ((v - minX) / rangeX) * cw;
    const yScale = v => pad.t + ch - ((v - minY) / rangeY) * ch;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {{
        const y = pad.t + (ch / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        const x = pad.l + (cw / 4) * i;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ch); ctx.stroke();
    }}

    // Regression line
    if (data.slope !== undefined) {{
        ctx.strokeStyle = 'rgba(245,158,11,0.7)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        const lx1 = minX, ly1 = data.slope * minX + data.intercept;
        const lx2 = maxX, ly2 = data.slope * maxX + data.intercept;
        ctx.beginPath(); ctx.moveTo(xScale(lx1), yScale(ly1)); ctx.lineTo(xScale(lx2), yScale(ly2)); ctx.stroke();
        ctx.setLineDash([]);

        // R² label
        ctx.fillStyle = 'rgba(245,158,11,0.9)';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`r = ${{data.pearson_r.toFixed(3)}}  R² = ${{data.r_squared.toFixed(3)}}`, W - pad.r - 5, pad.t + 14);
    }}

    // Points
    ctx.fillStyle = 'rgba(99,102,241,0.6)';
    for (let i = 0; i < xs.length; i++) {{
        ctx.beginPath();
        ctx.arc(xScale(xs[i]), yScale(ys[i]), 2.5, 0, Math.PI * 2);
        ctx.fill();
    }}

    // Axes labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {{
        const v = minX + rangeX * i / 4;
        ctx.fillText(v.toPrecision(3), xScale(v), H - pad.b + 15);
    }}
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
        const v = minY + rangeY * (4 - i) / 4;
        ctx.fillText(v.toFixed(1), pad.l - 6, pad.t + (ch / 4) * i + 4);
    }}

    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, pad.l + cw / 2, H - 3);

    ctx.save();
    ctx.translate(12, pad.t + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
}}

// ═══ Build Heatmap ═══
function buildHeatmap(containerId, corrData) {{
    const labels = corrData.labels;
    const matrix = corrData.matrix;
    const container = document.getElementById(containerId);

    let html = '<table class="heatmap"><tr><th></th>';
    labels.forEach(l => html += `<th>${{l}}</th>`);
    html += '</tr>';

    for (let i = 0; i < labels.length; i++) {{
        html += `<tr><th>${{labels[i]}}</th>`;
        for (let j = 0; j < labels.length; j++) {{
            const v = matrix[i][j];
            const bg = corrColor(v);
            const tc = corrTextColor(v);
            html += `<td style="background:${{bg}};color:${{tc}}" data-r="${{labels[i]}}" data-c="${{labels[j]}}" data-v="${{v}}">${{Math.abs(v) >= 0.01 ? v.toFixed(2) : ''}}</td>`;
        }}
        html += '</tr>';
    }}
    html += '</table>';
    container.innerHTML = html;

    // Tooltip
    const tooltip = document.getElementById('hmTooltip');
    container.querySelectorAll('.heatmap td').forEach(td => {{
        td.addEventListener('mouseenter', e => {{
            tooltip.style.display = 'block';
            tooltip.innerHTML = `<b>${{td.dataset.r}}</b> × <b>${{td.dataset.c}}</b><br>r = ${{parseFloat(td.dataset.v).toFixed(4)}}`;
        }});
        td.addEventListener('mousemove', e => {{
            tooltip.style.left = e.clientX + 12 + 'px';
            tooltip.style.top = e.clientY - 10 + 'px';
        }});
        td.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    }});
}}

// ═══ INIT ═══
document.addEventListener('DOMContentLoaded', () => {{

    // ─── 1. Histograms ───
    const histGrid = document.getElementById('histogramGrid');
    const histFeatures = Object.keys(SPEECH.histograms);
    const featureLabels = {{
        'jitter_local': 'Jitter (%)', 'shimmer_local': 'Shimmer',
        'hnr': 'Harmonics-to-Noise Ratio', 'f0_mean': 'Fundamental Frequency (Hz)',
        'rpde': 'RPDE (Recurrence)', 'dfa': 'DFA (Fractal Scaling)',
        'nhr': 'Noise-to-Harmonics Ratio', 'ppe': 'Pitch Period Entropy'
    }};

    histFeatures.forEach(feat => {{
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `
            <div class="card-head">
                <h3>${{featureLabels[feat] || feat}}</h3>
                <span class="badge" style="background:rgba(99,102,241,0.12);color:var(--indigo);border:1px solid rgba(99,102,241,0.25)">PD vs HC</span>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:rgba(244,63,94,0.7)"></div>PD (n=147)</div>
                <div class="legend-item"><div class="legend-dot" style="background:rgba(34,211,238,0.7)"></div>HC (n=48)</div>
            </div>
            <div class="chart-box"><canvas id="hist-${{feat}}" width="520" height="260"></canvas></div>
        `;
        histGrid.appendChild(div);

        setTimeout(() => {{
            drawHistogram(
                document.getElementById('hist-' + feat),
                SPEECH.histograms[feat].PD,
                SPEECH.histograms[feat].HC,
                featureLabels[feat] || feat
            );
        }}, 100);
    }});

    // ─── 2. Significance Table ───
    const sigBody = document.getElementById('sigTableBody');
    Object.entries(SPEECH.feature_stats).forEach(([feat, data]) => {{
        if (!data.groups || !data.groups.test) return;
        const pd = data.groups.PD, hc = data.groups.HC, test = data.groups.test;
        const pVal = test.p_value;
        let sigClass, sigLabel;
        if (pVal < 0.001) {{ sigClass = 'sig-high'; sigLabel = '★★★ p < 0.001'; }}
        else if (pVal < 0.05) {{ sigClass = 'sig-med'; sigLabel = '★★ p < 0.05'; }}
        else {{ sigClass = 'sig-low'; sigLabel = 'Not significant'; }}

        sigBody.innerHTML += `<tr>
            <td style="font-weight:600;color:var(--text)">${{feat}}</td>
            <td>${{pd.mean.toFixed(4)}} ± ${{pd.std.toFixed(4)}}</td>
            <td>${{hc.mean.toFixed(4)}} ± ${{hc.std.toFixed(4)}}</td>
            <td>${{test.t_statistic.toFixed(3)}}</td>
            <td>${{pVal < 0.0001 ? pVal.toExponential(2) : pVal.toFixed(4)}}</td>
            <td>${{test.cohens_d.toFixed(3)}}</td>
            <td class="${{sigClass}}">${{sigLabel}}</td>
        </tr>`;
    }});

    // ─── 3. Correlation Heatmap (Speech) ───
    buildHeatmap('heatmapContainer', SPEECH.correlation);

    // ─── 4. Scatter Plots (Telemonitoring) ───
    const scatterGrid = document.getElementById('scatterGrid');
    const scatters = TELE.regression.updrs_part_3_scatters || {{}};
    const scatterLabels = {{
        'jitter_local': 'Jitter (%)', 'shimmer_local': 'Shimmer',
        'hnr': 'HNR (dB)', 'nhr': 'NHR', 'rpde': 'RPDE', 'dfa': 'DFA'
    }};

    Object.entries(scatters).forEach(([feat, data]) => {{
        const div = document.createElement('div');
        div.className = 'card';
        const pSig = data.p_value < 0.001 ? 'p < 0.001' : data.p_value < 0.05 ? 'p < 0.05' : `p = ${{data.p_value.toFixed(3)}}`;
        div.innerHTML = `
            <div class="card-head">
                <h3>${{scatterLabels[feat] || feat}} vs Motor UPDRS</h3>
                <span class="badge" style="background:rgba(59,130,246,0.12);color:var(--blue);border:1px solid rgba(59,130,246,0.25)">${{pSig}}</span>
            </div>
            <div class="chart-box"><canvas id="scatter-${{feat}}" width="520" height="280"></canvas></div>
            <div class="chart-label">n = ${{data.x.length}} samples · Pearson r = ${{data.pearson_r.toFixed(3)}}</div>
        `;
        scatterGrid.appendChild(div);

        setTimeout(() => {{
            drawScatter(
                document.getElementById('scatter-' + feat),
                data,
                scatterLabels[feat] || feat,
                'Motor UPDRS'
            );
        }}, 150);
    }});

    // ─── 5. Subject Table ───
    const subBody = document.getElementById('subjectTableBody');
    (TELE.subjects || []).forEach(s => {{
        subBody.innerHTML += `<tr>
            <td style="color:var(--text);font-weight:600">${{s.subject_id}}</td>
            <td>${{s.age || '—'}}</td>
            <td>${{s.sex || '—'}}</td>
            <td>${{s.n_recordings}}</td>
            <td>${{s.motor_updrs_mean !== undefined ? s.motor_updrs_mean.toFixed(1) : '—'}}</td>
            <td>${{s.total_updrs_mean !== undefined ? s.total_updrs_mean.toFixed(1) : '—'}}</td>
        </tr>`;
    }});

    // ─── 6. Telemonitoring Heatmap ───
    buildHeatmap('teleHeatmapContainer', TELE.correlation);

    // ─── 7. Raw vs Scaled Transform ───
    const demo = SPEECH.preprocessing_demo;
    if (demo && demo.raw_vs_scaled) {{
        const feats = demo.raw_vs_scaled.feature_names.slice(0, 10); // top 10
        const rawRows = demo.raw_vs_scaled.raw;
        const scaledRows = demo.raw_vs_scaled.scaled;

        // Raw table
        const rawT = document.getElementById('rawTable');
        let rh = '<thead><tr><th>#</th>';
        feats.forEach(f => rh += `<th>${{f}}</th>`);
        rh += '</tr></thead><tbody>';
        rawRows.forEach((row, i) => {{
            rh += `<tr><td style="color:var(--text);font-weight:600">P${{i+1}}</td>`;
            feats.forEach(f => rh += `<td>${{row[f] !== undefined ? parseFloat(row[f]).toFixed(5) : '—'}}</td>`);
            rh += '</tr>';
        }});
        rh += '</tbody>';
        rawT.innerHTML = rh;

        // Scaled table
        const scT = document.getElementById('scaledTable');
        let sh = '<thead><tr><th>#</th>';
        feats.forEach(f => sh += `<th>${{f}}</th>`);
        sh += '</tr></thead><tbody>';
        scaledRows.forEach((row, i) => {{
            sh += `<tr><td style="color:var(--text);font-weight:600">P${{i+1}}</td>`;
            feats.forEach(f => {{
                const v = row[f];
                const cls = v > 0 ? 'color:var(--cyan)' : v < 0 ? 'color:var(--rose)' : '';
                sh += `<td style="${{cls}}">${{v !== undefined ? parseFloat(v).toFixed(4) : '—'}}</td>`;
            }});
            sh += '</tr>';
        }});
        sh += '</tbody>';
        scT.innerHTML = sh;

        // Scaling stats
        const ssBody = document.getElementById('scalingStatsBody');
        feats.forEach(f => {{
            const mu = demo.raw_vs_scaled.scaling_means[f];
            const sigma = demo.raw_vs_scaled.scaling_stds[f];
            ssBody.innerHTML += `<tr>
                <td style="color:var(--text);font-weight:600">${{f}}</td>
                <td>${{mu !== undefined ? mu.toFixed(6) : '—'}}</td>
                <td>${{sigma !== undefined ? sigma.toFixed(6) : '—'}}</td>
                <td style="color:var(--violet)">z = (x − ${{mu !== undefined ? mu.toFixed(3) : 'μ'}}) / ${{sigma !== undefined ? sigma.toFixed(3) : 'σ'}}</td>
            </tr>`;
        }});
    }}
}});
</script>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[+] Showcase HTML built: {output_path}")
    print(f"    Size: {len(html):,} bytes")


if __name__ == "__main__":
    analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real", "analysis"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "showcase", "real_data_showcase.html"))
    build_showcase_html(analysis_dir, output_path)
