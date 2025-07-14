"""
analyze_results_drive.py
------------------------
Pull every result_*.json from Drive → clean, aggregate, plot, GPT-insights → upload back.
"""

import os
import io
import json
import tempfile
import re
import pandas as pd
import matplotlib.pyplot as plt

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.service_account import Credentials

from dotenv import load_dotenv
import openai

# ─── CONFIG / ENV LOADING ─────────────────────────────────────────────────────
load_dotenv()  # loads GOOGLE_APPLICATION_CREDENTIALS & OPENAI_API_KEY
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
openai.api_key = OPENAI_KEY

# Your Drive folder that contains result_*.json
RESULT_FOLDER_ID = "1NN8_VERmh4xe0Z2mZiQSqYkwxmj5fNRs"

# Resolve the service-account keyfile path
script_dir = os.path.dirname(os.path.abspath(__file__))
keyfile_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not keyfile_env:
    raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS in environment or .env")

# If the path is not absolute, join it to the script directory
KEYFILE = keyfile_env if os.path.isabs(keyfile_env) else os.path.join(script_dir, keyfile_env)
if not os.path.exists(KEYFILE):
    raise RuntimeError(f"Service keyfile not found at: {KEYFILE}")

# ─── GOOGLE DRIVE CLIENT ───────────────────────────────────────────────────────
def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(KEYFILE, scopes=scopes)
    return build("drive", "v3", credentials=creds)

# ─── LIST result_*.json FILES ─────────────────────────────────────────────────
def list_result_jsons(svc, folder_id):
    q = (
        f"'{folder_id}' in parents and trashed=false "
        "and mimeType='application/json' and name contains 'result_'"
    )
    files, token = [], None
    while True:
        resp = svc.files().list(
            q=q,
            fields="nextPageToken, files(id,name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=token
        ).execute()
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return files

# ─── DOWNLOAD A FILE ───────────────────────────────────────────────────────────
def download_file(svc, file_id, dest_path):
    request = svc.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

# ─── CLEAN COMMODITY RECORDS ───────────────────────────────────────────────────
def clean_commodities(comm_list):
    rows = []
    for item in comm_list:
        # parse date
        try:
            date = pd.to_datetime(item.get("date",""), dayfirst=True)
        except:
            date = pd.NaT
        # parse price range
        txt = (item.get("price") or "").replace(" francs","")
        parts = re.split(r"\s*to\s*|\s*-\s*", txt)
        try:
            pmin = float(parts[0])
            pmax = float(parts[1]) if len(parts)>1 else pmin
            pavg = (pmin + pmax)/2
        except:
            pavg = None
        rows.append({"name": item.get("name"), "date": date, "price_avg": pavg})
    return rows

# ─── AGGREGATE & PLOT ──────────────────────────────────────────────────────────
def aggregate_and_plot(rows, outdir):
    df = pd.DataFrame(rows).dropna(subset=["price_avg"])
    if df.empty:
        print("❗ No numeric commodity data found.")
        return None, None
    df.sort_values("date", inplace=True)

    # CSV
    csv_path = os.path.join(outdir, "aggregated_commodities.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(12,6))
    for name, grp in df.groupby("name"):
        plt.plot(grp["date"], grp["price_avg"], marker="o", label=name)
    plt.title("Historical Commodity Prices")
    plt.xlabel("Date")
    plt.ylabel("Average Price (francs)")
    plt.grid()
    plt.legend()
    png_path = os.path.join(outdir, "commodity_prices.png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return csv_path, png_path

# ─── GPT-4 INSIGHTS ────────────────────────────────────────────────────────────
def gpt_insights(all_docs, outdir):
    prompt = (
        "You are an economic historian. Given this structured JSON archive data, provide "
        "exactly three concise, non-obvious bullet-point insights about:\n"
        "• commodity price trends\n"
        "• networks of persons/companies\n"
        "• macroeconomic context\n\n"
        f"DATA:\n{json.dumps(all_docs)[:15000]}"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    insights = resp.choices[0].message.content.strip()
    path = os.path.join(outdir, "insights_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(insights)
    return path

# ─── UPLOAD IF ABSENT ───────────────────────────────────────────────────────────
def upload_if_absent(svc, folder_id, local_path):
    name = os.path.basename(local_path)
    exists = svc.files().list(
        q=f"'{folder_id}' in parents and name='{name}' and trashed=false",
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute().get("files")
    if exists:
        print(f"↩️  {name} already exists in Drive; skipping.")
        return
    media = MediaFileUpload(local_path)
    svc.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=media,
        supportsAllDrives=True
    ).execute()
    print(f"⬆️  Uploaded {name}")

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def main():
    svc = get_drive_service()
    metas = list_result_jsons(svc, RESULT_FOLDER_ID)
    if not metas:
        print("No result_*.json files found; exiting.")
        return

    tmpdir = tempfile.mkdtemp(prefix="resdl_")
    print(f"▼ Downloading {len(metas)} JSON files → {tmpdir}")

    all_docs, rows = [], []
    for m in metas:
        local = os.path.join(tmpdir, m["name"])
        download_file(svc, m["id"], local)
        doc = json.load(open(local, encoding="utf-8"))
        all_docs.append(doc)
        rows.extend(clean_commodities(doc.get("commodities", [])))

    # Aggregate & plot
    csv_p, png_p = aggregate_and_plot(rows, tmpdir)
    # GPT insights
    txt_p = gpt_insights(all_docs, tmpdir)

    print("✔ Artifacts:")
    for p in (csv_p, png_p, txt_p):
        print("   ", p)

    # Upload back to Drive
    for p in (csv_p, png_p, txt_p):
        if p:
            upload_if_absent(svc, RESULT_FOLDER_ID, p)

    print("🎉 Done.")

if __name__ == "__main__":
    main()
