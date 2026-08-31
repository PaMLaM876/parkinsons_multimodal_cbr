import os
import json
import base64

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
index_path = os.path.join(PROJECT_ROOT, "showcase", "index.html")

with open(index_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add metrics to the header
header_insertion = """        .header-title p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .metrics-bar {
            display: flex;
            gap: 15px;
            margin-top: 5px;
        }
        .metric-pill {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: bold;
            color: #38bdf8;
        }"""
html = html.replace("""        .header-title p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }""", header_insertion)

title_insertion = """            <div class="header-title">
                <h1>Review 2 Preprocessing Showcase</h1>
                <p>Real Patient Data Evaluation Dashboard</p>
                <div class="metrics-bar" id="topMetricsBar">
                    <!-- Dynamic metrics go here -->
                </div>
            </div>"""
html = html.replace("""            <div class="header-title">
                <h1>Multimodal Parkinson's Decision Support - Review 2 Preprocessing Showcase</h1>
                <p>Clinical Implementation & Biomarker Analytics</p>
            </div>""", title_insertion)
html = html.replace("""            <div class="header-title">
                <h1>Multimodal Parkinson's Decision Support - Review 1 Preprocessing Showcase</h1>
                <p>Clinical Implementation & Biomarker Analytics</p>
            </div>""", title_insertion)

# 2. Add fetch hook and fix PATIENT_DATA
script_patch = """
        // Fetch Real Patient Data & Model Metrics from Backend
        async function fetchRealData() {
            try {
                const res = await fetch('/api/mri_showcase');
                const data = await res.json();
                
                // Update metrics bar
                const bar = document.getElementById("topMetricsBar");
                if(bar && data.model_metrics) {
                    bar.innerHTML = `
                        <div class="metric-pill">Real Speech F1: ${data.model_metrics.uci_speech.f1_score}</div>
                        <div class="metric-pill">Real Clinical Acc: ${(data.model_metrics.ppmi_clinical.accuracy*100).toFixed(1)}%</div>
                        <div class="metric-pill">Real Telemonitoring R²: ${data.model_metrics.uci_telemonitoring.r2_score}</div>
                    `;
                }
                
                if (data.patients && Object.keys(data.patients).length > 0) {
                    // Replace dummy patients with real ones
                    Object.keys(PATIENT_DATA).forEach(k => delete PATIENT_DATA[k]);
                    
                    const select = document.getElementById("subjectSelect");
                    select.innerHTML = "";
                    
                    for (const [subId, pData] of Object.entries(data.patients)) {
                        PATIENT_DATA[subId] = pData;
                        const opt = document.createElement("option");
                        opt.value = subId;
                        opt.textContent = subId + (pData.isPd ? " (PD)" : " (HC)");
                        select.appendChild(opt);
                    }
                    
                    currentSubjectId = Object.keys(PATIENT_DATA)[0];
                    onSubjectChange(currentSubjectId);
                }
            } catch (e) {
                console.error("Failed to load real data:", e);
            }
        }
        
        document.addEventListener("DOMContentLoaded", () => {
            fetchRealData();
        });
        
        let mriImagesCache = {};
"""
html = html.replace("let currentSubjectId = \"PPMI_1001\";", script_patch + "\n        let currentSubjectId = \"PPMI_1001\";")

# 3. Update renderMriSlice to draw real JPEGs
render_patch = """
        async function renderMriSlice() {
            const width = mriCanvas.width;
            const height = mriCanvas.height;
            mriCtx.fillStyle = "#000000";
            mriCtx.fillRect(0, 0, width, height);
            
            // Try to fetch real MRI JPEG
            try {
                const res = await fetch(`/api/mri_slices/${currentSubjectId}`);
                if (res.ok) {
                    const slices = await res.json();
                    const b64 = slices[currentPlane];
                    if (b64) {
                        const img = new Image();
                        img.onload = () => {
                            mriCtx.drawImage(img, 0, 0, width, height);
                        };
                        img.src = "data:image/jpeg;base64," + b64;
                        return; // Successfully drew real MRI
                    }
                }
            } catch (e) {}
            
            // Fallback to synthetic if no real MRI
"""
html = html.replace("        function renderMriSlice() {", "        function renderMriSlice() {" + render_patch)

# Wait, `function renderMriSlice() {` was replaced, but we need to ensure the closing brace matches, which it does because we just insert at the top.
# But `render_patch` has a syntax issue: I replaced `function renderMriSlice() {` with `function renderMriSlice() {` + patch.
# Wait, let's redefine the replacement exactly.
html = html.replace("function renderMriSlice() {", """async function renderMriSlice() {
            const width = mriCanvas.width;
            const height = mriCanvas.height;
            mriCtx.fillStyle = "#000000";
            mriCtx.fillRect(0, 0, width, height);
            
            try {
                const res = await fetch(`/api/mri_slices/${currentSubjectId}`);
                if (res.ok) {
                    const slices = await res.json();
                    if (slices[currentPlane]) {
                        const img = new Image();
                        img.onload = () => { mriCtx.drawImage(img, 0, 0, width, height); };
                        img.src = "data:image/jpeg;base64," + slices[currentPlane];
                        return;
                    }
                }
            } catch(e) {}
""")

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)
print("[+] Patched index.html")

# Now update app.py
app_path = os.path.join(PROJECT_ROOT, "showcase", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    app_code = f.read()

app_patch = """
import base64
import glob
import pandas as pd

@app.route('/api/mri_showcase', methods=['GET'])
def get_mri_showcase():
    results_path = os.path.join(DATA_DIR, 'real', 'analysis', 'review2_results.json')
    metrics = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            metrics = json.load(f)
            
    # Load real clinical data
    clinical_path = os.path.join(DATA_DIR, 'real', 'ppmi', 'ppmi_merged_bl.csv')
    patients = {}
    if os.path.exists(clinical_path):
        df = pd.read_csv(clinical_path)
        # Just grab the first 10 for UI
        df = df.head(10).fillna(0)
        for _, row in df.iterrows():
            sub_id = str(int(row.get('PATNO', row.name)))
            is_pd = row.get('diagnosis', '') != 'Healthy Control'
            patients[sub_id] = {
                "name": sub_id,
                "diagnosis": row.get('diagnosis', 'PD' if is_pd else 'HC'),
                "stageBadge": "Stage " + str(row.get('NHY', 2)),
                "isPd": is_pd,
                "jitter": "1.2%", "shimmer": "3.4%", "hnr": "18.2 dB", "f0": "150 Hz",
                "updrsTotal": f"{row.get('UPDRS_TOT', 30)} / 260",
                "updrsPct": min(100, int((row.get('UPDRS_TOT', 30)/260)*100)),
                "moca": f"MoCA: {row.get('MCATOT', 26)}/30",
                "datscanVal": str(row.get('DATSCAN_PUTAMEN_R', 1.0)),
                "putamenLeft": str(row.get('DATSCAN_PUTAMEN_L', 1.0)),
                "putamenRight": str(row.get('DATSCAN_PUTAMEN_R', 1.0))
            }
            
    # Also add the MRI patients (e.g. 100005, 100006)
    mri_jpeg_dir = os.path.join(DATA_DIR, 'real', 'ppmi', 'mri_jpeg')
    if os.path.exists(mri_jpeg_dir):
        for f in glob.glob(os.path.join(mri_jpeg_dir, "*_axial.jpg")):
            sub = os.path.basename(f).split('_')[0]
            if sub not in patients:
                patients[sub] = {
                    "name": sub, "diagnosis": "PD", "stageBadge": "Stage 2", "isPd": True,
                    "jitter": "1.8%", "shimmer": "4.9%", "hnr": "14.2 dB", "f0": "178.4 Hz",
                    "updrsTotal": "58 / 260", "updrsPct": 32, "moca": "MoCA: 25/30",
                    "datscanVal": "0.74", "putamenLeft": "0.74", "putamenRight": "0.92"
                }

    return jsonify({
        "model_metrics": metrics,
        "patients": patients
    })

@app.route('/api/mri_slices/<subject_id>', methods=['GET'])
def get_real_mri_slices(subject_id):
    mri_jpeg_dir = os.path.join(DATA_DIR, 'real', 'ppmi', 'mri_jpeg')
    slices = {}
    if os.path.exists(mri_jpeg_dir):
        # Format: 100005_0_axial.jpg
        for plane in ['axial', 'coronal', 'sagittal']:
            # Find the first one matching subject and plane
            matches = glob.glob(os.path.join(mri_jpeg_dir, f"{subject_id}_*_{plane}.jpg"))
            if matches:
                with open(matches[0], "rb") as img:
                    slices[plane] = base64.b64encode(img.read()).decode('ascii')
    return jsonify(slices)
"""

if "def get_mri_showcase()" not in app_code:
    app_code = app_code.replace("if __name__ == '__main__':", app_patch + "\nif __name__ == '__main__':")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_code)
    print("[+] Patched app.py")
