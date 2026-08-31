"""
Build Review 2 Showcase HTML.
"""
import os
import json
import base64
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def build_html():
    results_path = os.path.join(PROJECT_ROOT, "data", "real", "analysis", "review2_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # Get a few MRI JPEGs
    jpeg_dir = os.path.join(PROJECT_ROOT, "data", "real", "ppmi", "mri_jpeg")
    jpegs = glob.glob(os.path.join(jpeg_dir, "*.jpg"))
    jpegs = jpegs[:6] # take first 6
    
    img_tags = ""
    for j in jpegs:
        with open(j, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        plane = os.path.basename(j).split('_')[-1].replace('.jpg', '').capitalize()
        img_tags += f'<div style="display: inline-block; margin: 10px;"><img src="data:image/jpeg;base64,{b64}" style="width: 220px; height: 220px; object-fit: cover; border-radius: 8px; border: 1px solid #444;" /><br/><span style="color:#94a3b8; font-size: 0.8rem;">{plane} Plane</span></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Review 2: Real Dataset Models</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 3rem; }}
        h1 {{ color: #38bdf8; text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem; }}
        .card-container {{ display: flex; gap: 2rem; justify-content: center; margin-top: 3rem; flex-wrap: wrap; }}
        .card {{ background: #1e293b; padding: 2rem; border-radius: 16px; width: 320px; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-5px); border-color: #475569; }}
        .card h2 {{ color: #a78bfa; margin-top: 0; font-size: 1.4rem; }}
        .metric {{ font-size: 2.5rem; font-weight: 800; color: #22d3ee; margin: 0.5rem 0 1.5rem 0; font-family: monospace; }}
        .label {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em; }}
        .gallery {{ text-align: center; margin-top: 4rem; background: #1e293b; padding: 3rem; border-radius: 16px; border: 1px solid #334155; }}
        .gallery h2 {{ color: #34d399; margin-top: 0; font-size: 1.8rem; }}
    </style>
</head>
<body>
    <h1>Review 2: Independent Real Dataset Training</h1>
    <p style="text-align: center; color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
        As requested for Review 2, the multimodal fusion architecture has been bypassed. Models are now trained and evaluated strictly on specific real-world modalities.
    </p>
    
    <div class="card-container">
        <!-- UCI Speech -->
        <div class="card">
            <h2>UCI Parkinson's (Speech)</h2>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.5;">Trained on 195 real voice samples to classify PD vs HC using acoustic features.</p>
            <div class="label">Test Accuracy</div>
            <div class="metric">{results['uci_speech']['accuracy']*100:.1f}%</div>
            <div class="label">F1 Score</div>
            <div class="metric">{results['uci_speech']['f1_score']:.3f}</div>
        </div>
        
        <!-- UCI Telemonitoring -->
        <div class="card">
            <h2>UCI Telemonitoring (Progression)</h2>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.5;">Trained on {results['uci_telemonitoring']['dataset_size']:,} longitudinal voice samples to predict Motor UPDRS progression.</p>
            <div class="label">R² Score</div>
            <div class="metric">{results['uci_telemonitoring']['r2_score']:.3f}</div>
            <div class="label">Mean Squared Error</div>
            <div class="metric">{results['uci_telemonitoring']['mse']:.2f}</div>
        </div>
        
        <!-- PPMI Clinical -->
        <div class="card">
            <h2>PPMI Clinical (Tabular)</h2>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.5;">Trained on {results['ppmi_clinical']['dataset_size']:,} real patient tabular records (MDS-UPDRS, MoCA, DaTscan).</p>
            <div class="label">Test Accuracy</div>
            <div class="metric">{results['ppmi_clinical']['accuracy']*100:.1f}%</div>
            <div class="label">F1 Score</div>
            <div class="metric">{results['ppmi_clinical']['f1_score']:.3f}</div>
        </div>
    </div>
    
    <div class="gallery">
        <h2>Real PPMI MRI Conversion (DICOM to JPEG)</h2>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 2rem;">Successfully processed 1,948 raw DICOM slices into stacked 3D volumes and extracted the central visualization JPEGs.</p>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem;">
            {img_tags}
        </div>
    </div>
</body>
</html>
"""
    out_path = os.path.join(PROJECT_ROOT, "showcase", "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Replaced showcase/index.html with Review 2 Dashboard!")

if __name__ == "__main__":
    build_html()
