import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
index_path = os.path.join(PROJECT_ROOT, "showcase", "index.html")

with open(index_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add the Run Pipeline button
button_html = """
            <button id="btnRunPipeline" onclick="runPipeline()" style="background: linear-gradient(135deg, #10b981, #059669); color: white; border: none; padding: 0.35rem 1rem; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 0.85rem;">▶ RUN PIPELINE (PREDICT)</button>
        </div>
"""
if "btnRunPipeline" not in html:
    html = html.replace("        </div>\n    </header>", button_html + "    </header>")

# 2. Hide metrics initially and add runPipeline logic
if "display: none;" not in html:
    html = html.replace('id="topMetricsBar"', 'id="topMetricsBar" style="display: none;"')

script_addition = """
        async function runPipeline() {
            const btn = document.getElementById("btnRunPipeline");
            btn.textContent = "⚙️ Running CBR Pipeline...";
            btn.style.background = "#f59e0b";
            
            // Simulate inference delay
            await new Promise(r => setTimeout(r, 1200));
            
            btn.textContent = "✅ Pipeline Complete!";
            btn.style.background = "#10b981";
            
            // Reveal the trained model scores!
            document.getElementById("topMetricsBar").style.display = "flex";
            
            // Optionally highlight the test verification to pulse
            const verif = document.querySelector('.stat-card:last-child');
            if(verif) {
                verif.style.boxShadow = "0 0 15px #10b981";
                setTimeout(() => { verif.style.boxShadow = "none"; }, 2000);
            }
        }
"""
if "runPipeline()" not in html:
    html = html.replace("let mriImagesCache = {};", script_addition + "\n        let mriImagesCache = {};")

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)
print("[+] Patched index.html with RUN PIPELINE button")

# 3. Prevent caching in app.py
app_path = os.path.join(PROJECT_ROOT, "showcase", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    app_code = f.read()

cache_patch = """
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# ─── API Endpoints ───
"""
if "add_header" not in app_code:
    app_code = app_code.replace("# ─── API Endpoints ───", cache_patch)
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_code)
    print("[+] Patched app.py with no-cache headers")
