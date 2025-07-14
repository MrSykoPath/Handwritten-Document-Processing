#!/usr/bin/env python3
"""
analyze_local_results_gemini.py
--------------------------------
• Reads every Documents/result_*.json (tolerates wrappers)
• Cleans & aggregates commodity records    → aggregated_commodities.csv
• Plots price trends                       → commodity_prices.png
• Asks Gemini for 3 bullet-point insights  → insights_summary.txt
Outputs land in analysis_outputs/
"""

import os, re, json, tempfile
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai  # pip install google-genai

# ─── PATHS ─────────────────────────────────────────────────────────────────────
INPUT_DIR   = os.path.join(os.getcwd(), "Documents")
OUTPUT_DIR  = os.path.join(os.getcwd(), "analysis_outputs")
GEM_MODEL   = "gemini-2.5-flash"           # or gemini-pro, gemini-1.5-pro, etc.
MAX_JSON_CHARS = 15_000                   # truncate to keep prompt small

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── ENV / CLIENT ──────────────────────────────────────────────────────────────
load_dotenv()                # expects GEMINI_API_KEY in .env or env variable
client = genai.Client()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_json_block(text: str):
    """Return dict parsed from the first {...} block in `text`."""
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None

def load_all_docs(folder):
    """Yield parsed JSON dicts from every result_*.json in `folder`."""
    for fname in sorted(os.listdir(folder)):
        if not re.match(r"result_.*\.json$", fname):
            continue
        path = os.path.join(folder, fname)
        if os.path.getsize(path) == 0:
            print(f"⚠️  Empty file skipped: {fname}")
            continue
        raw = open(path, encoding="utf-8", errors="ignore").read()
        doc = extract_json_block(raw)
        if doc is None:
            print(f"⚠️  Could not parse JSON in: {fname}")
            continue
        yield fname, doc

def clean_commodities(comm_list):
    rows = []
    for item in comm_list:
        # date
        try:
            dt = pd.to_datetime(item.get("date",""), dayfirst=True)
        except Exception:
            dt = pd.NaT
        # price
        text = (item.get("price") or "").replace(" francs","")
        parts = re.split(r"\s*to\s*|\s*-\s*", text)
        try:
            pmin = float(parts[0]); pmax = float(parts[1]) if len(parts)>1 else pmin
            pavg = (pmin + pmax)/2
        except Exception:
            pavg = None
        rows.append({"name": item.get("name"), "date": dt, "price_avg": pavg})
    return rows

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    all_docs, commodity_rows = [], []

    for fname, doc in load_all_docs(INPUT_DIR):
        all_docs.append(doc)
        commodity_rows.extend(clean_commodities(doc.get("commodities", [])))

    if not all_docs:
        print("❌ No valid JSON files found in", INPUT_DIR)
        return

    # ── Aggregate & CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(commodity_rows).dropna(subset=["price_avg"])
    if df.empty:
        print("❗ No numeric commodity data.")
    else:
        df.sort_values("date", inplace=True)
        csv_path = os.path.join(OUTPUT_DIR, "aggregated_commodities.csv")
        df.to_csv(csv_path, index=False)
        print("✅ CSV saved →", csv_path)

        # ── Plot ───────────────────────────────────────────────────────────────
        plt.figure(figsize=(12,6))
        for cname, grp in df.groupby("name"):
            plt.plot(grp["date"], grp["price_avg"], marker="o", label=cname)
        plt.title("Historical Commodity Prices")
        plt.xlabel("Date"); plt.ylabel("Average Price (francs)")
        plt.grid(); plt.legend()
        png_path = os.path.join(OUTPUT_DIR, "commodity_prices.png")
        plt.tight_layout(); plt.savefig(png_path); plt.close()
        print("✅ Plot saved →", png_path)

    # ── Gemini insights ───────────────────────────────────────────────────────
    prompt = (
        "You are an economic historian. Given this structured JSON archive data, "
        "provide exactly three concise, non-obvious bullet-point insights covering:\n"
        "• commodity price movements\n"
        "• networks of persons/companies\n"
        "• macroeconomic context\n\n"
        "DATA:\n" + json.dumps(all_docs)[:MAX_JSON_CHARS]
    )
    resp = client.models.generate_content(model=GEM_MODEL, contents=prompt)
    insights = resp.text.strip()

    txt_path = os.path.join(OUTPUT_DIR, "insights_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(insights)
    print("✅ Insights saved →", txt_path)

    print("\n🎉 Pipeline complete! Check", OUTPUT_DIR)

if __name__ == "__main__":
    main()
