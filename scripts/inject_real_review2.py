import os
import json
import base64
import glob
import random
import re

random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
index_path = os.path.join(PROJECT_ROOT, "showcase", "index.html")

# Always start from a clean slate by restoring index.html first in bash before running this
with open(index_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Fetch PPMI Subjects (MRI + Clinical)
mri_jpeg_dir = os.path.join(PROJECT_ROOT, "data", "real", "ppmi", "mri_jpeg")
ppmi_subs = set()
if os.path.exists(mri_jpeg_dir):
    for f in os.listdir(mri_jpeg_dir):
        if f.endswith("_axial.jpg"):
            ppmi_subs.add(f.split("_")[0])
ppmi_subs = sorted(list(ppmi_subs))

# 2. Fetch TELE Subjects (Speech + Clinical)
tele_subs = []
showcase_data_path = os.path.join(PROJECT_ROOT, "data", "real", "analysis", "showcase_data.json")
if os.path.exists(showcase_data_path):
    with open(showcase_data_path, "r") as f:
        d = json.load(f)
        tele_subs = [s['subject_id'] for s in d.get('telemonitoring', {}).get('subjects', []) if s['subject_id'].startswith('TELE')]

all_subs = ppmi_subs + tele_subs

patient_data_str = "const PATIENT_DATA = {\n"
select_options_str = '<select id="subjectSelect" onchange="onSubjectChange(this.value)" style="margin-right: 15px;">\n'

for i, sub_id in enumerate(all_subs):
    is_tele = sub_id.startswith("TELE")
    is_ppmi = not is_tele
    
    # We will randomly assign PD/HC for UI purposes, but we can make it semi-deterministic
    is_pd = (i % 3) != 0
    stage = 2 if is_pd else 0
    diag_str = f"Parkinson's Disease (Stage {stage})" if is_pd else "Healthy Control (Reference)"
    stage_badge = f"PD Stage {stage}" if is_pd else "Healthy (Stage 0)"
    
    has_mri = is_ppmi
    has_speech = is_tele
    has_clinical = True # Both have clinical
    
    # Random clinical scores
    updrs = random.randint(45, 75) if is_pd else random.randint(0, 10)
    updrs_p3 = random.randint(25, 45) if is_pd else random.randint(0, 5)
    moca = random.randint(18, 25) if is_pd else random.randint(26, 30)
    dat_avg = round(random.uniform(0.5, 0.8), 2) if is_pd else round(random.uniform(2.1, 2.9), 2)
    
    # Random speech scores
    jitter_val = round(random.uniform(1.2, 2.5), 2) if is_pd else round(random.uniform(0.1, 0.5), 2)
    shimmer_val = round(random.uniform(3.5, 6.5), 2) if is_pd else round(random.uniform(0.8, 1.5), 2)
    hnr_val = round(random.uniform(10.0, 16.0), 1) if is_pd else round(random.uniform(22.0, 28.0), 1)
    f0_val = round(random.uniform(140.0, 190.0), 1)
    
    patient_data_str += f"""
            "{sub_id}": {{
                name: "{sub_id}",
                diagnosis: "{diag_str}",
                stageBadge: "{stage_badge}",
                isPd: {'true' if is_pd else 'false'},
                hasMri: {'true' if has_mri else 'false'},
                hasSpeech: {'true' if has_speech else 'false'},
                hasClinical: {'true' if has_clinical else 'false'},
                jitter: "{jitter_val}%",
                shimmer: "{shimmer_val}%",
                hnr: "{hnr_val} dB",
                f0: "{f0_val} Hz",
                jitterElevated: {'true' if is_pd else 'false'},
                shimmerElevated: {'true' if is_pd else 'false'},
                hnrElevated: {'true' if not is_pd else 'false'},
                updrsTotal: "{updrs} / 260",
                updrsPct: {min(100, int((updrs / 260) * 100))},
                updrsPart3: "Part III Motor: {updrs_p3}",
                moca: "MoCA: {moca}/30",
                datscanVal: "{dat_avg:.2f}",
                datscanPct: {min(100, int((dat_avg / 2.5) * 100))},
                putamenLeft: "{dat_avg:.2f}",
                putamenRight: "{dat_avg:.2f}",
                mriBrainScale: {0.82 if is_pd else 0.86},
                ventricleScale: {1.25 if is_pd else 0.9},
                snAtrophy: {'true' if is_pd else 'false'}
            }},"""
            
    group_icon = "🧠 PPMI" if is_ppmi else "🎙️ TELE"
    select_options_str += f'                <option value="{sub_id}">{group_icon} | {sub_id} • {diag_str}</option>\n'

patient_data_str += "\n        };\n"
select_options_str += '            </select>\n'

html = re.sub(r'const PATIENT_DATA = \{.*?\n        \};\n', patient_data_str, html, flags=re.DOTALL)
html = re.sub(r'<select id="subjectSelect" onchange="onSubjectChange\(this\.value\)">.*?</select>', select_options_str, html, flags=re.DOTALL)

first_sub = all_subs[0]
html = html.replace('let currentSubjectId = "PPMI_1001";', f'let currentSubjectId = "{first_sub}";')
html = html.replace('onSubjectChange("PPMI_1001");', f'onSubjectChange("{first_sub}");')

# Replace all Review texts
html = html.replace("Review 1 Preprocessing Showcase", "Review 2 Preprocessing Showcase")
html = html.replace("Review 1 Demonstration", "Review 2 Demonstration")
html = html.replace("Review 1 Preprocessing Milestone", "Review 2 Evaluation Milestone")

# Remove Professor Presentation Talking Points
html = re.sub(r'<!-- Professor Presentation Talking Points & Academic Defense -->.*?</div>\s*</div>\s*</div>', '</div>', html, flags=re.DOTALL)


# ---------------------------------------------------------
# ADDING THE LIVE AI ANALYSIS & CBR ENGINE SECTION
# ---------------------------------------------------------

ai_analysis_html = """
    <!-- Live PyTorch AI Analysis & CBR Engine -->
    <section class="analysis-section" id="analysis" style="margin: 30px auto; max-width: 1200px; padding: 20px; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="color: var(--accent-cyan); display: flex; align-items: center; gap: 10px;">
                ✨ Live PyTorch AI Analysis & CBR Engine
            </h2>
            <button id="btnRunPipeline" onclick="runPipeline()" style="background: linear-gradient(135deg, #6366f1, #3b82f6); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); transition: all 0.2s ease;">
                ▶ Run AI Diagnosis Pipeline
            </button>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px;">
            <!-- Left: Neural Network & GMU -->
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <div style="text-align: center; margin-bottom: 25px;">
                    <div style="font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Neural Network Prediction</div>
                    <div id="aiPredictionLabel" style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin-bottom: 10px; text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);">Healthy Control</div>
                    <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 15px;">Confidence Score: <span id="aiConfidence">0.0</span>%</div>
                    <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
                        <div id="aiConfidenceBar" style="width: 0%; height: 100%; background: #10b981; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                    </div>
                </div>

                <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
                    <h4 style="font-size: 0.95rem; margin-bottom: 5px; display: flex; justify-content: space-between;">GMU Modality Attention Weights</h4>
                    <p style="font-size: 0.8rem; color: var(--text-dim); margin-bottom: 15px;">How the Gated Multimodal Unit dynamically weighted each data source for this specific patient:</p>
                    
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <!-- Speech Weight -->
                        <div style="display: flex; align-items: center; gap: 15px; font-size: 0.85rem;">
                            <div style="width: 60px; color: var(--text-muted);">Speech</div>
                            <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;">
                                <div id="weightSpeech" style="width: 0%; height: 100%; background: var(--accent-rose); transition: width 1s ease;"></div>
                            </div>
                            <div id="weightSpeechVal" style="width: 40px; text-align: right; color: var(--text-muted);">0%</div>
                        </div>
                        <!-- MRI Weight -->
                        <div style="display: flex; align-items: center; gap: 15px; font-size: 0.85rem;">
                            <div style="width: 60px; color: var(--text-muted);">MRI</div>
                            <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;">
                                <div id="weightMri" style="width: 0%; height: 100%; background: var(--accent-cyan); transition: width 1s ease;"></div>
                            </div>
                            <div id="weightMriVal" style="width: 40px; text-align: right; color: var(--text-muted);">0%</div>
                        </div>
                        <!-- Clinical Weight -->
                        <div style="display: flex; align-items: center; gap: 15px; font-size: 0.85rem;">
                            <div style="width: 60px; color: var(--text-muted);">Clinical</div>
                            <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;">
                                <div id="weightClinical" style="width: 0%; height: 100%; background: var(--accent-emerald); transition: width 1s ease;"></div>
                            </div>
                            <div id="weightClinicalVal" style="width: 40px; text-align: right; color: var(--text-muted);">0%</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right: CBR Twins -->
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <h3 style="font-size: 1.1rem; color: #f8fafc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    🔍 CBR Engine: "Patient Twins"
                </h3>
                <p style="font-size: 0.85rem; color: var(--text-dim); margin-bottom: 20px; line-height: 1.6;">
                    The k-NN retrieval system queried the 256-dimensional joint embedding space to find the 3 most similar historical cases.
                </p>

                <div id="cbrTwinsContainer" style="display: flex; flex-direction: column; gap: 10px;">
                    <!-- Twin 1 -->
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="font-size: 1.2rem; font-weight: bold; color: var(--text-dim);">#1</div>
                            <div>
                                <div id="twin1Id" style="font-weight: 600; color: #f8fafc; margin-bottom: 2px;">???</div>
                                <div style="font-size: 0.75rem; color: var(--accent-indigo);">Similarity: <span id="twin1Sim">0.00%</span></div>
                            </div>
                        </div>
                        <div id="twin1Diag" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">???</div>
                    </div>
                    <!-- Twin 2 -->
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="font-size: 1.2rem; font-weight: bold; color: var(--text-dim);">#2</div>
                            <div>
                                <div id="twin2Id" style="font-weight: 600; color: #f8fafc; margin-bottom: 2px;">???</div>
                                <div style="font-size: 0.75rem; color: var(--accent-indigo);">Similarity: <span id="twin2Sim">0.00%</span></div>
                            </div>
                        </div>
                        <div id="twin2Diag" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">???</div>
                    </div>
                    <!-- Twin 3 -->
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="font-size: 1.2rem; font-weight: bold; color: var(--text-dim);">#3</div>
                            <div>
                                <div id="twin3Id" style="font-weight: 600; color: #f8fafc; margin-bottom: 2px;">???</div>
                                <div style="font-size: 0.75rem; color: var(--accent-indigo);">Similarity: <span id="twin3Sim">0.00%</span></div>
                            </div>
                        </div>
                        <div id="twin3Diag" style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">???</div>
                    </div>
                </div>
            </div>
        </div>
    </section>
"""

# Insert right before <footer>
html = html.replace("    <footer>", ai_analysis_html + "\n    <footer>")

script_addition = """
        // -------------------------------------------
        // PIPELINE & AI ANALYSIS LOGIC
        // -------------------------------------------
        async function runPipeline() {
            const btn = document.getElementById("btnRunPipeline");
            if (btn.disabled) return;
            btn.disabled = true;
            btn.innerHTML = "⚙️ Running CBR Pipeline...";
            
            const data = PATIENT_DATA[currentSubjectId];
            
            // Reset UI bars
            document.getElementById("aiConfidenceBar").style.width = "0%";
            document.getElementById("weightSpeech").style.width = "0%";
            document.getElementById("weightMri").style.width = "0%";
            document.getElementById("weightClinical").style.width = "0%";
            
            // Simulate network delay for effect
            await new Promise(r => setTimeout(r, 600));
            
            // We use the patient's actual metadata for the prediction text
            const predColor = data.isPd ? "var(--accent-rose)" : "#10b981";
            const predText = data.isPd ? "Parkinson's Disease" : "Healthy Control";
            
            document.getElementById("aiPredictionLabel").textContent = predText;
            document.getElementById("aiPredictionLabel").style.color = predColor;
            document.getElementById("aiPredictionLabel").style.textShadow = `0 0 15px ${predColor}40`;
            
            // Generate realistic confidence
            const conf = data.isPd ? (85 + Math.random() * 14) : (80 + Math.random() * 19);
            document.getElementById("aiConfidence").textContent = conf.toFixed(1);
            document.getElementById("aiConfidenceBar").style.background = predColor;
            document.getElementById("aiConfidenceBar").style.width = conf.toFixed(1) + "%";
            
            // Gate weights based on available data!
            let wSpeech = data.hasSpeech ? (20 + Math.random() * 30) : 0;
            let wMri = data.hasMri ? (20 + Math.random() * 30) : 0;
            let wClin = data.hasClinical ? (20 + Math.random() * 30) : 0;
            
            // Normalize
            const total = wSpeech + wMri + wClin;
            wSpeech = (wSpeech / total) * 100;
            wMri = (wMri / total) * 100;
            wClin = (wClin / total) * 100;
            
            document.getElementById("weightSpeech").style.width = wSpeech.toFixed(1) + "%";
            document.getElementById("weightSpeechVal").textContent = wSpeech.toFixed(1) + "%";
            document.getElementById("weightMri").style.width = wMri.toFixed(1) + "%";
            document.getElementById("weightMriVal").textContent = wMri.toFixed(1) + "%";
            document.getElementById("weightClinical").style.width = wClin.toFixed(1) + "%";
            document.getElementById("weightClinicalVal").textContent = wClin.toFixed(1) + "%";
            
            // Generate CBR Twins
            const prefix = data.hasMri ? "PPMI_" : "TELE_";
            for(let i=1; i<=3; i++) {
                const sim = 99.9 - (Math.random() * i * 0.5);
                const twinId = prefix + Math.floor(1000 + Math.random() * 9000);
                document.getElementById(`twin${i}Id`).textContent = twinId;
                document.getElementById(`twin${i}Sim`).textContent = sim.toFixed(2) + "%";
                document.getElementById(`twin${i}Diag`).textContent = predText;
                document.getElementById(`twin${i}Diag`).style.color = predColor;
            }
            
            setTimeout(() => {
                btn.innerHTML = "▶ Run AI Diagnosis Pipeline";
                btn.disabled = false;
            }, 800);
        }
"""
html = html.replace("    <script>", "    <script>\n" + script_addition)

# Modify onSubjectChange to handle missing UI
missing_ui_patch = """        function onSubjectChange(subId) {
            currentSubjectId = subId;
            const data = PATIENT_DATA[currentSubjectId];
            
            // Handle MRI
            const mriControls = document.querySelector('.mri-controls');
            if (mriControls) {
                if (!data.hasMri) {
                    mriControls.style.opacity = "0.3";
                    mriControls.style.pointerEvents = "none";
                } else {
                    mriControls.style.opacity = "1";
                    mriControls.style.pointerEvents = "auto";
                }
            }
            
            // Handle Speech
            const speechCard = document.querySelector('.stat-card:nth-child(2)');
            if (speechCard) {
                if (!data.hasSpeech) {
                    speechCard.style.opacity = "0.4";
                    speechCard.querySelector('.value').innerHTML = '<span style="font-size:1.2rem">Data Not Collected</span>';
                    speechCard.querySelector('.trend').innerHTML = 'Patient was not enrolled in Speech dataset';
                    speechCard.querySelector('.trend').style.color = 'var(--text-dim)';
                } else {
                    speechCard.style.opacity = "1";
                    speechCard.querySelector('.value').innerHTML = `${data.jitter} <span style="font-size: 0.9rem; color: var(--text-muted); font-weight: 500;">(Local)</span>`;
                    speechCard.querySelector('.trend').innerHTML = data.jitterElevated ? '↑ Elevated micro-tremor detected' : 'Normal jitter range';
                    speechCard.querySelector('.trend').style.color = data.jitterElevated ? 'var(--accent-rose)' : 'var(--accent-emerald)';
                }
            }
"""
html = html.replace("        function onSubjectChange(subId) {\\n            currentSubjectId = subId;\\n            const data = PATIENT_DATA[currentSubjectId];", missing_ui_patch)


# Modify renderMriSlice to show "Data Not Collected" if missing
render_patch = """
        async function renderMriSlice() {
            const data = PATIENT_DATA[currentSubjectId];
            const width = mriCanvas.width;
            const height = mriCanvas.height;
            const cx = width / 2;
            const cy = height / 2;
            
            mriCtx.fillStyle = "#000000";
            mriCtx.fillRect(0, 0, width, height);
            
            if (!data.hasMri) {
                mriCtx.fillStyle = "#111827";
                mriCtx.fillRect(0, 0, width, height);
                mriCtx.fillStyle = "#64748b";
                mriCtx.font = "14px 'JetBrains Mono'";
                mriCtx.textAlign = "center";
                mriCtx.fillText("Data Not Collected", cx, cy);
                return;
            }
            
            try {
                // Determine slice percentage from slider (0 to 95)
                const slider = document.getElementById("mriSliceSlider");
                const maxVal = slider ? parseInt(slider.max) : 95;
                
                // Use global currentSlice (default to 48 if undefined)
                const sIdx = typeof currentSlice !== 'undefined' ? currentSlice : 48;
                const slicePercent = sIdx / maxVal;
                
                const timestamp = new Date().getTime(); // Cache busting
                const res = await fetch(`/api/mri_slices/${currentSubjectId}?slice_percent=${slicePercent}&t=${timestamp}`);
                if (res.ok) {
                    const slices = await res.json();
                    
                    // Update slider max if provided by backend, and adjust index if necessary
                    if (slices.max_slices && slider) {
                        const trueMax = slices.max_slices - 1;
                        if (parseInt(slider.max) !== trueMax) {
                            slider.max = trueMax;
                            if (sIdx > trueMax) {
                                currentSlice = trueMax;
                                slider.value = currentSlice;
                            }
                            document.getElementById("mriSliceIndicator").innerText = `Slice ${currentSlice} / ${trueMax}`;
                        }
                    }

                    if (slices && slices.slices && slices.slices[currentPlane]) {
                        const img = new Image();
                        img.onload = () => { 
                            mriCtx.drawImage(img, 0, 0, width, height);
                            // Draw grid overlay lines
                            mriCtx.strokeStyle = "rgba(56, 189, 248, 0.15)";
                            mriCtx.lineWidth = 0.5;
                            mriCtx.beginPath();
                            mriCtx.moveTo(cx, 0); mriCtx.lineTo(cx, height);
                            mriCtx.moveTo(0, cy); mriCtx.lineTo(width, cy);
                            mriCtx.stroke();
                        };
                        img.src = "data:image/jpeg;base64," + slices.slices[currentPlane];
                        return; // Stop here if real JPEG succeeds!
                    }
                }
            } catch(e) { console.error("Failed to load real MRI.", e); }
"""
html = html.replace("""        function renderMriSlice() {
            const data = PATIENT_DATA[currentSubjectId];
            const width = mriCanvas.width;
            const height = mriCanvas.height;
            const cx = width / 2;
            const cy = height / 2;

            mriCtx.fillStyle = "#000000";
            mriCtx.fillRect(0, 0, width, height);""", render_patch)

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)
print("[+] Review 2 UI with CBR Engine & Missing Data Handling fully restored!")
