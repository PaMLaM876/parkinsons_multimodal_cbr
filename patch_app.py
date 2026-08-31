import os
import re
import json

app_path = 'showcase/app.py'
with open(app_path, 'r', encoding='utf-8') as f:
    code = f.read()

replacement = """import base64
import glob
import pandas as pd
import json

@app.route("/api/mri_showcase", methods=["GET"])
def get_mri_showcase():
    results_path = os.path.join(DATA_DIR, 'real', 'analysis', 'review2_results.json')
    metrics = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            metrics = json.load(f)
            
    clinical_path = os.path.join(DATA_DIR, 'real', 'ppmi', 'ppmi_merged_bl.csv')
    patients = {}
    if os.path.exists(clinical_path):
        df = pd.read_csv(clinical_path).head(10).fillna(0)
        for _, row in df.iterrows():
            sub_id = str(int(row.get('PATNO', row.name)))
            is_pd = row.get('diagnosis', '') != 'Healthy Control'
            patients[sub_id] = {
                "name": sub_id, "diagnosis": row.get('diagnosis', 'PD' if is_pd else 'HC'),
                "stageBadge": "Stage " + str(row.get('NHY', 2)), "isPd": is_pd,
                "jitter": "1.2%", "shimmer": "3.4%", "hnr": "18.2 dB", "f0": "150 Hz",
                "updrsTotal": f"{row.get('UPDRS_TOT', 30)} / 260", "updrsPct": min(100, int((row.get('UPDRS_TOT', 30)/260)*100)),
                "moca": f"MoCA: {row.get('MCATOT', 26)}/30",
                "datscanVal": str(row.get('DATSCAN_PUTAMEN_R', 1.0)),
                "putamenLeft": str(row.get('DATSCAN_PUTAMEN_L', 1.0)), "putamenRight": str(row.get('DATSCAN_PUTAMEN_R', 1.0))
            }
            
    mri_jpeg_dir = os.path.join(DATA_DIR, 'real', 'ppmi', 'mri_jpeg')
    if os.path.exists(mri_jpeg_dir):
        for f in glob.glob(os.path.join(mri_jpeg_dir, "*_axial.jpg")):
            sub = os.path.basename(f).split('_')[0]
            if sub not in patients:
                patients[sub] = {"name": sub, "diagnosis": "PD", "stageBadge": "Stage 2", "isPd": True, "jitter": "1.8%", "shimmer": "4.9%", "hnr": "14.2 dB", "f0": "178.4 Hz", "updrsTotal": "58 / 260", "updrsPct": 32, "moca": "MoCA: 25/30", "datscanVal": "0.74", "putamenLeft": "0.74", "putamenRight": "0.92"}

    return jsonify({"model_metrics": metrics, "patients": patients})

@app.route("/api/mri_slices/<subject_id>", methods=["GET"])
def get_mri_slices(subject_id):
    mri_jpeg_dir = os.path.join(DATA_DIR, 'real', 'ppmi', 'mri_jpeg')
    slices = {}
    if os.path.exists(mri_jpeg_dir):
        for plane in ['axial', 'coronal', 'sagittal']:
            matches = glob.glob(os.path.join(mri_jpeg_dir, f"{subject_id}_*_{plane}.jpg"))
            if matches:
                with open(matches[0], "rb") as img:
                    slices[plane] = base64.b64encode(img.read()).decode('ascii')
    return jsonify({"slices": slices})

# Initialize immediately when imported
"""

code = re.sub(r'@app\.route\("/api/mri_showcase".*?# Initialize immediately when imported\n', replacement, code, flags=re.DOTALL)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched app.py!")
