#!/usr/bin/env python3
"""
analyze_drive_results_gemini.py
--------------------------------
• Downloads every result_*.json from Google Drive folder
• Cleans & aggregates commodity records    → aggregated_commodities.csv
• Plots price trends                       → commodity_prices.png
• Asks Gemini for 3 bullet-point insights  → insights_summary.txt
• NEW: Extracts transactions & merchants, deduplicates names
• NEW: Generates merchant analytics, network analysis, price-activity correlations
• Uploads all outputs back to Drive folder
"""

import os, re, json, tempfile
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai  # pip install google-genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.service_account import Credentials
import io

# ─── PATHS & CONFIG ──────────────────────────────────────────────────────────
OUTPUT_DIR  = "temp_analysis"  # Temporary local directory for processing
GEM_MODEL   = "gemini-2.5-flash"           # or gemini-pro, gemini-1.5-pro, etc.
MAX_JSON_CHARS = 15_000                   # truncate to keep prompt small

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'documentextraction-465311-6d37979e03e0.json'

# Create temporary directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── ENV / CLIENTS ───────────────────────────────────────────────────────────
load_dotenv()                # expects GEMINI_API_KEY in .env or env variable
gemini_client = genai.Client()

# ─── GOOGLE DRIVE SETUP ──────────────────────────────────────────────────────
def get_drive_service():
    """Authenticate and create the Drive API service."""
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

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

def download_drive_files(service, folder_id):
    """Download all result_*.json files from Drive folder."""
    files = []
    page_token = None
    
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token
        ).execute()
        
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break
    
    # Filter for result_*.json files (only newer names like result_001_o3-2025-04-16.json, NOT old result_001.json)
    result_files = [f for f in files if re.match(r"result_\d+_.*\.json$", f['name'])]
    print(f"Found {len(result_files)} result files in Drive folder")
    
    downloaded_docs = []
    for file in result_files:
        try:
            # Download file
            request = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            content = fh.read().decode('utf-8', errors='ignore')
            
            # Parse JSON
            doc = extract_json_block(content)
            if doc is not None:
                downloaded_docs.append((file['name'], doc))
                print(f"✅ Downloaded and parsed: {file['name']}")
            else:
                print(f"⚠️  Could not parse JSON in: {file['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {file['name']}: {e}")
    
    return downloaded_docs

def upload_to_drive(service, folder_id, local_path, filename, mime_type):
    """Upload a file to Google Drive folder."""
    try:
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(local_path, mimetype=mime_type)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()
        
        print(f"✅ Uploaded {filename} to Drive")
        return file.get('id')
    except Exception as e:
        print(f"❌ Error uploading {filename}: {e}")
        return None

def clean_commodities(comm_list):
    rows = []
    for item in comm_list:
        # date
        try:
            dt = pd.to_datetime(item.get("date",""), dayfirst=True, errors="coerce")
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

# ─── NEW TRANSACTION ANALYSIS FUNCTIONS ──────────────────────────────────────
def walk_json(obj, path=()):
    """Recursively walk JSON object and yield (node, path) tuples."""
    if isinstance(obj, dict):
        yield obj, path
        for k, v in obj.items():
            yield from walk_json(v, path + (str(k),))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            yield from walk_json(v, path + (str(idx),))

# Transaction detection keys in multiple languages
MERCHANT_KEYS = {"merchant","vendor","seller","payee","trader","house","company","compagnie","maison","marchand","تاجر","تاجِر","شركة","المورد","بيت","بائع"}
AMOUNT_KEYS = {"amount","montant","sum","value","price","prix","قيمة","مبلغ","ثمن"}
TRANSACTION_ARRAYS = {"transactions","entries","operations","purchases","sales","lignes","items","بنود","عمليات","مشتريات","مبيعات"}

def parse_transactions(doc: dict, source_file: str):
    """Extract transaction records from JSON document."""
    rows = []
    
    # Extract from companies array (these are merchants)
    companies = doc.get("companies", [])
    commodities = doc.get("commodities", [])
    places = doc.get("Places", [])
    doc_date = doc.get("date")
    
    # Parse document date
    try:
        parsed_date = pd.to_datetime(doc_date, dayfirst=True, errors="coerce") if doc_date else None
    except:
        parsed_date = None
    
    # Create transactions from companies and commodities
    for company in companies:
        if company:  # Skip empty strings
            # Look for matching commodity info
            matching_commodity = None
            for commodity in commodities:
                if commodity.get("name"):
                    matching_commodity = commodity
                    break
            
            # Extract amount from commodity price if available
            amount = None
            currency = None
            if matching_commodity and matching_commodity.get("price"):
                price_text = str(matching_commodity["price"])
                # Try to extract numeric amount and currency
                amount_match = re.search(r'(\d+(?:\.\d+)?)', price_text)
                if amount_match:
                    amount = float(amount_match.group(1))
                    # Extract currency (everything after the number)
                    currency_text = price_text[amount_match.end():].strip()
                    if currency_text:
                        currency = currency_text
            
            # Create transaction record
            row = {
                "source_file": source_file,
                "date": parsed_date,
                "merchant_raw": company,
                "merchant_canonical": None,  # Will be filled by dedupe step
                "counterparty": None,  # Could be extracted from text analysis later
                "commodity": matching_commodity.get("name") if matching_commodity else None,
                "quantity": None,  # Could be extracted from text analysis
                "unit": None,
                "amount": amount,
                "currency": currency,
                "notes": f"Extracted from {source_file}",
                "location": places[0] if places else None,
            }
            
            rows.append(row)
    
    # Also try to extract from commodities if no companies found
    if not rows and commodities:
        for commodity in commodities:
            if commodity.get("name") and commodity.get("price"):
                price_text = str(commodity["price"])
                amount_match = re.search(r'(\d+(?:\.\d+)?)', price_text)
                amount = float(amount_match.group(1)) if amount_match else None
                
                row = {
                    "source_file": source_file,
                    "date": parsed_date,
                    "merchant_raw": "Unknown Merchant",  # Placeholder
                    "merchant_canonical": None,
                    "counterparty": None,
                    "commodity": commodity.get("name"),
                    "quantity": None,
                    "unit": None,
                    "amount": amount,
                    "currency": None,
                    "notes": f"Commodity transaction from {source_file}",
                    "location": places[0] if places else None,
                }
                
                rows.append(row)
    
    return rows

def normalize_name(s: str) -> str:
    """Normalize merchant names for deduplication."""
    if not s: 
        return ""
    
    import re
    s = str(s)
    
    # Remove accents if unidecode available
    try:
        from unidecode import unidecode
        s = unidecode(s)
    except ImportError:
        pass
    
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    
    # Remove Arabic articles/company words & French legal suffixes
    stop = {"ال","شركة","شركه","ش.","ش.م.","s.a.","sa","societe","compagnie","la","le","l'","maison"}
    tokens = [t for t in s.split() if t not in stop]
    
    return " ".join(tokens)

def dedupe_merchants(df):
    """Deduplicate merchant names using fuzzy matching."""
    df = df.copy()
    df["name_key"] = df["merchant_raw"].astype(str).map(normalize_name)
    
    # Group by normalized name key
    groups = {k: [k] for k in df["name_key"].unique() if k}
    
    try:
        from rapidfuzz import fuzz
        keys = list(groups.keys())
        used = set()
        clusters = []
        
        for i, k in enumerate(keys):
            if k in used: 
                continue
            cluster = [k]
            used.add(k)
            
            for j in range(i+1, len(keys)):
                kk = keys[j]
                if kk in used: 
                    continue
                # Use first 4 chars for blocking, then fuzzy match
                if k[:4] == kk[:4] and fuzz.token_sort_ratio(k, kk) >= 90:
                    cluster.append(kk)
                    used.add(kk)
            
            clusters.append(cluster)
        
        # Map each key to a cluster id
        mapping = {}
        for idx, cl in enumerate(clusters, start=1):
            for k in cl: 
                mapping[k] = f"M{idx:04d}"
                
    except ImportError:
        print("⚠️  rapidfuzz not installed → using simple deduplication")
        # Fallback: one id per key
        mapping = {k: f"M{i:04d}" for i, k in enumerate(groups.keys(), start=1)}
    except Exception as e:
        print(f"⚠️  Fuzzy deduplication failed: {e} → using simple deduplication")
        mapping = {k: f"M{i:04d}" for i, k in enumerate(groups.keys(), start=1)}
    
    df["merchant_id"] = df["name_key"].map(mapping)
    
    # Get canonical name (most common form)
    canon = (df.groupby(["merchant_id", "merchant_raw"])
               .size().sort_values(ascending=False)
               .groupby(level=0).head(1).reset_index())
    canon = canon.drop(columns=0).groupby("merchant_id")["merchant_raw"].first()
    df["merchant_canonical"] = df["merchant_id"].map(canon)
    
    # Create merchants summary table
    agg = (df.groupby("merchant_id")
             .agg({
                 "merchant_canonical": "first",
                 "date": ["min", "max"],
                 "amount": ["size", "sum"],
                 "currency": lambda x: list(x.dropna().unique())
             }))
    
    # Flatten column names
    agg.columns = ["merchant_canonical", "first_seen", "last_seen", "txn_count", "total_amount", "currencies"]
    df_merchants = agg.reset_index()
    
    return df, df_merchants

# ─── NEW PLOTTING FUNCTIONS ─────────────────────────────────────────────────
def plot_merchant_timeseries(df_txn, output_dir):
    """Plot merchant transaction timeseries."""
    try:
        # Resample by month and get top merchants
        df_monthly = df_txn.set_index("date").resample("M")["amount"].sum().reset_index()
        top_merchants = df_txn.groupby("merchant_canonical")["amount"].sum().nlargest(10).index
        
        plt.figure(figsize=(14, 8))
        
        for merchant in top_merchants:
            merchant_data = df_txn[df_txn["merchant_canonical"] == merchant].set_index("date").resample("M")["amount"].sum()
            plt.plot(merchant_data.index, merchant_data.values, marker="o", label=merchant, linewidth=2)
        
        plt.title("Top 10 Merchants - Monthly Transaction Amounts", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Total Amount", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "merchant_timeseries.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ Merchant timeseries plot created")
        return png_path
        
    except Exception as e:
        print(f"⚠️  Could not create merchant timeseries plot: {e}")
        return None

def plot_merchant_heatmap(df_txn, output_dir):
    """Plot merchant activity heatmap."""
    try:
        # Get top 25 merchants by total amount
        top_merchants = df_txn.groupby("merchant_canonical")["amount"].sum().nlargest(25).index
        
        # Create pivot table: merchants x months
        df_pivot = df_txn[df_txn["merchant_canonical"].isin(top_merchants)].copy()
        df_pivot["month"] = df_pivot["date"].dt.to_period("M")
        
        heatmap_data = df_pivot.pivot_table(
            index="merchant_canonical", 
            columns="month", 
            values="amount", 
            aggfunc="sum", 
            fill_value=0
        )
        
        plt.figure(figsize=(16, 10))
        plt.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")
        
        # Set labels
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index, fontsize=10)
        plt.xticks(range(len(heatmap_data.columns)), [str(col) for col in heatmap_data.columns], rotation=45, fontsize=9)
        
        plt.title("Merchant Activity Heatmap (Top 25 by Total Amount)", fontsize=16)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Merchant", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label("Total Amount", fontsize=12)
        
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "merchant_heatmap.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ Merchant heatmap created")
        return png_path
        
    except Exception as e:
        print(f"⚠️  Could not create merchant heatmap: {e}")
        return None

def build_network(df_txn, output_dir):
    """Build merchant network and export GEXF + optional PNG."""
    try:
        # Create bipartite graph: merchants vs counterparties
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes
        merchants = df_txn["merchant_canonical"].unique()
        counterparties = df_txn["counterparty"].dropna().unique()
        
        for merchant in merchants:
            G.add_node(merchant, type="merchant")
        
        for cp in counterparties:
            G.add_node(cp, type="person_company")
        
        # Add edges with weights
        edge_weights = df_txn.groupby(["merchant_canonical", "counterparty"]).size()
        for (merchant, cp), weight in edge_weights.items():
            if pd.notna(cp):
                G.add_edge(merchant, cp, weight=weight)
        
        # Export GEXF
        gexf_path = os.path.join(output_dir, "merchant_network.gexf")
        nx.write_gexf(G, gexf_path)
        print("✅ Network GEXF exported")
        
        # Create PNG visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes by type
        merchant_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "merchant"]
        cp_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "person_company"]
        
        nx.draw_networkx_nodes(G, pos, nodelist=merchant_nodes, node_color="lightblue", 
                              node_size=[G.degree(n) * 100 for n in merchant_nodes], alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=cp_nodes, node_color="lightcoral", 
                              node_size=[G.degree(n) * 100 for n in cp_nodes], alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=[G[u][v]["weight"] * 0.5 for u, v in G.edges()])
        
        plt.title("Merchant Network", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "merchant_network.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ Network PNG created")
        return gexf_path, png_path
        
    except ImportError:
        print("⚠️  networkx not installed → skipping network analysis")
        return None, None
    except Exception as e:
        print(f"⚠️  Could not create network: {e}")
        return None, None

def join_price_activity(df_commodities, df_txn, output_dir):
    """Create price vs transaction activity correlation plot."""
    try:
        if df_commodities.empty or df_txn.empty:
            print("⚠️  Insufficient data for price-activity correlation")
            return None
        
        # Resample both datasets to monthly
        df_comm_monthly = df_commodities.set_index("date").resample("M")["price_avg"].mean().reset_index()
        df_txn_monthly = df_txn.set_index("date").resample("M").agg({
            "amount": "sum",
            "source_file": "count"
        }).reset_index().rename(columns={"source_file": "txn_count"})
        
        # Join on month
        df_comm_monthly["month"] = df_comm_monthly["date"].dt.to_period("M")
        df_txn_monthly["month"] = df_txn_monthly["date"].dt.to_period("M")
        
        df_joined = df_comm_monthly.merge(df_txn_monthly, on="month", how="inner")
        
        if df_joined.empty:
            print("⚠️  No overlapping dates for price-activity correlation")
            return None
        
        # Create correlation plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Price vs transaction count
        ax1.scatter(df_joined["txn_count"], df_joined["price_avg"], alpha=0.7, s=100)
        ax1.set_xlabel("Monthly Transaction Count")
        ax1.set_ylabel("Average Commodity Price")
        ax1.set_title("Price vs Transaction Activity")
        ax1.grid(True, alpha=0.3)
        
        # Price vs transaction amount
        ax2.scatter(df_joined["amount"], df_joined["price_avg"], alpha=0.7, s=100)
        ax2.set_xlabel("Monthly Transaction Amount")
        ax2.set_ylabel("Average Commodity Price")
        ax2.set_title("Price vs Transaction Volume")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "price_activity_corr.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ Price-activity correlation plot created")
        return png_path
        
    except Exception as e:
        print(f"⚠️  Could not create price-activity correlation: {e}")
        return None

def detect_anomalies(df_txn, output_dir):
    """Detect anomalous transaction amounts per merchant."""
    try:
        if df_txn.empty:
            print("⚠️  No transaction data for anomaly detection")
            return None
        
        anomalies = []
        
        for merchant_id in df_txn["merchant_id"].unique():
            merchant_data = df_txn[df_txn["merchant_id"] == merchant_id].copy()
            
            if len(merchant_data) < 3:  # Need at least 3 transactions for rolling stats
                continue
                
            merchant_data = merchant_data.sort_values("date")
            
            # Calculate rolling statistics (3-month window)
            rolling_mean = merchant_data["amount"].rolling(window=3, min_periods=1).mean()
            rolling_std = merchant_data["amount"].rolling(window=3, min_periods=1).std()
            
            # Calculate z-scores
            z_scores = (merchant_data["amount"] - rolling_mean) / rolling_std
            
            # Find anomalies (z > 2.5)
            anomaly_mask = z_scores > 2.5
            
            if anomaly_mask.any():
                anomaly_rows = merchant_data[anomaly_mask].copy()
                anomaly_rows["z_score"] = z_scores[anomaly_mask]
                anomaly_rows["merchant_id"] = merchant_id
                
                for _, row in anomaly_rows.iterrows():
                    anomalies.append({
                        "merchant_id": merchant_id,
                        "merchant_canonical": row["merchant_canonical"],
                        "date": row["date"],
                        "amount": row["amount"],
                        "z_score": row["z_score"],
                        "source_file": row["source_file"]
                    })
        
        if anomalies:
            df_anomalies = pd.DataFrame(anomalies)
            csv_path = os.path.join(output_dir, "anomalies.csv")
            df_anomalies.to_csv(csv_path, index=False)
            print(f"✅ Anomalies detected: {len(anomalies)} records")
            return csv_path
        else:
            print("✅ No anomalies detected")
            return None
            
    except Exception as e:
        print(f"⚠️  Could not detect anomalies: {e}")
        return None

def enhance_gemini_prompt(all_docs, df_txn, df_merchants, max_chars):
    """Create enhanced Gemini prompt with transaction/merchant context."""
    try:
        # Create schema summary
        schema_summary = """SCHEMA:
transactions(date, merchant_canonical, counterparty, commodity, quantity, unit, amount, currency)
merchants(merchant_id, merchant_canonical, first_seen, last_seen, txn_count, total_amount)

SAMPLES:"""
        
        # Add sample transaction rows
        tx_samples = df_txn.head(10).to_dict("records") if not df_txn.empty else []
        tx_json = json.dumps(tx_samples, default=str, ensure_ascii=False)
        
        # Add sample merchant rows
        merchant_samples = df_merchants.head(10).to_dict("records") if not df_merchants.empty else []
        merchant_json = json.dumps(merchant_samples, default=str, ensure_ascii=False)
        
        # Combine everything
        enhanced_prompt = f"""{schema_summary}

TRANSACTIONS:
{tx_json}

MERCHANTS:
{merchant_json}

You are an economic historian. Given this structured JSON archive data with transaction and merchant information, provide exactly three concise, non-obvious bullet-point insights covering:

• merchant concentration & transaction spikes
• merchant–person/company network structure  
• relation between commodity price and transaction activity

DATA:
{json.dumps(all_docs, default=str, ensure_ascii=False)}"""
        
        # Truncate if too long
        if len(enhanced_prompt) > max_chars:
            enhanced_prompt = enhanced_prompt[:max_chars-100] + "...\n[truncated]"
        
        return enhanced_prompt
        
    except Exception as e:
        print(f"⚠️  Could not enhance Gemini prompt: {e}")
        # Fallback to original prompt
        return (
            "You are an economic historian. Given this structured JSON archive data, "
            "provide exactly three concise, non-obvious bullet-point insights covering:\n"
            "• commodity price movements\n"
            "• networks of persons/companies\n"
            "• macroeconomic context\n\n"
            "DATA:\n" + json.dumps(all_docs, default=str, ensure_ascii=False)[:max_chars]
        )

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Get Drive service
    drive_service = get_drive_service()
    
    # Drive folder IDs - update these with your actual folder IDs
    source_folder_id = "1loLg-htSD0XtU5MgzofbVCd4lFMsiKpg"  # Source images folder
    result_folder_id = "1NN8_VERmh4xe0Z2mZiQSqYkwxmj5fNRs"  # Results folder (same as ChatGptTest.py)
    
    print("🔍 Downloading result files from Google Drive...")
    all_docs = download_drive_files(drive_service, result_folder_id)
    
    if not all_docs:
        print("❌ No valid JSON files found in Drive folder")
        return
    
    # Extract commodity data (existing functionality)
    commodity_rows = []
    for fname, doc in all_docs:
        commodity_rows.extend(clean_commodities(doc.get("commodities", [])))
    
    # NEW: Extract transaction data
    print("🔍 Extracting transactions from documents...")
    all_transactions = []
    for fname, doc in all_docs:
        print(f"  Processing {fname}...")
        transactions = parse_transactions(doc, fname)
        print(f"    Found {len(transactions)} transactions")
        all_transactions.extend(transactions)
    
    # Create transaction DataFrame
    if all_transactions:
        df_txn_raw = pd.DataFrame(all_transactions)
        print(f"✅ Extracted {len(df_txn_raw)} transaction records")
        
        # Clean transaction data
        df_txn_raw = df_txn_raw.dropna(subset=["merchant_raw", "amount"], how="all")
        df_txn_raw = df_txn_raw[df_txn_raw["amount"].notna()]
        
        if not df_txn_raw.empty:
            # Deduplicate merchants
            print("🔍 Deduplicating merchant names...")
            df_txn, df_merchants = dedupe_merchants(df_txn_raw)
            print(f"✅ Deduplicated to {df_merchants['merchant_id'].nunique()} unique merchants")
            
            # Save transaction and merchant data
            tx_csv_path = os.path.join(OUTPUT_DIR, "transactions.csv")
            df_txn.to_csv(tx_csv_path, index=False)
            print("✅ Transactions CSV created locally")
            
            merch_csv_path = os.path.join(OUTPUT_DIR, "merchants.csv")
            df_merchants.to_csv(merch_csv_path, index=False)
            print("✅ Merchants CSV created locally")
            
            # Upload to Drive
            upload_to_drive(drive_service, result_folder_id, tx_csv_path, "transactions.csv", "text/csv")
            upload_to_drive(drive_service, result_folder_id, merch_csv_path, "merchants.csv", "text/csv")
            
            # Create new analytics plots
            print("📊 Creating merchant analytics...")
            
            # Merchant timeseries
            timeseries_path = plot_merchant_timeseries(df_txn, OUTPUT_DIR)
            if timeseries_path:
                upload_to_drive(drive_service, result_folder_id, timeseries_path, "merchant_timeseries.png", "image/png")
            
            # Merchant heatmap
            heatmap_path = plot_merchant_heatmap(df_txn, OUTPUT_DIR)
            if heatmap_path:
                upload_to_drive(drive_service, result_folder_id, heatmap_path, "merchant_heatmap.png", "image/png")
            
            # Network analysis
            gexf_path, network_png_path = build_network(df_txn, OUTPUT_DIR)
            if gexf_path:
                upload_to_drive(drive_service, result_folder_id, gexf_path, "merchant_network.gexf", "application/xml")
            if network_png_path:
                upload_to_drive(drive_service, result_folder_id, network_png_path, "merchant_network.png", "image/png")
            
            # Anomaly detection
            anomalies_path = detect_anomalies(df_txn, OUTPUT_DIR)
            if anomalies_path:
                upload_to_drive(drive_service, result_folder_id, anomalies_path, "anomalies.csv", "text/csv")
        else:
            print("⚠️  No valid transaction data found")
            df_txn = pd.DataFrame()
            df_merchants = pd.DataFrame()
    else:
        print("⚠️  No transactions found in documents")
        df_txn = pd.DataFrame()
        df_merchants = pd.DataFrame()
    
    # ── Aggregate & CSV (existing functionality) ─────────────────────────────
    df = pd.DataFrame(commodity_rows).dropna(subset=["price_avg"])
    if df.empty:
        print("❗ No numeric commodity data.")
    else:
        df.sort_values("date", inplace=True)
        
        # Save CSV locally first
        csv_path = os.path.join(OUTPUT_DIR, "aggregated_commodities.csv")
        df.to_csv(csv_path, index=False)
        print("✅ CSV created locally →", csv_path)
        
        # Upload CSV to Drive
        upload_to_drive(drive_service, result_folder_id, csv_path, "aggregated_commodities.csv", "text/csv")
        
        # ── Plot (existing functionality) ─────────────────────────────────────
        plt.figure(figsize=(12,6))
        for cname, grp in df.groupby("name"):
            plt.plot(grp["date"], grp["price_avg"], marker="o", label=cname)
        plt.title("Historical Commodity Prices")
        plt.xlabel("Date"); plt.ylabel("Average Price (francs)")
        plt.grid(); plt.legend()
        
        png_path = os.path.join(OUTPUT_DIR, "commodity_prices.png")
        plt.tight_layout(); plt.savefig(png_path); plt.close()
        print("✅ Plot created locally →", png_path)
        
        # Upload plot to Drive
        upload_to_drive(drive_service, result_folder_id, png_path, "commodity_prices.png", "image/png")
        
        # NEW: Price-activity correlation
        if not df_txn.empty:
            corr_path = join_price_activity(df, df_txn, OUTPUT_DIR)
            if corr_path:
                upload_to_drive(drive_service, result_folder_id, corr_path, "price_activity_corr.png", "image/png")

    # ── Gemini insights (enhanced) ───────────────────────────────────────────
    print("🤖 Generating insights with Gemini...")
    
    # Use enhanced prompt if we have transaction data
    if not df_txn.empty and not df_merchants.empty:
        prompt = enhance_gemini_prompt([doc for _, doc in all_docs], df_txn, df_merchants, MAX_JSON_CHARS)
    else:
        prompt = (
            "You are an economic historian. Given this structured JSON archive data, "
            "provide exactly three concise, non-obvious bullet-point insights covering:\n"
            "• commodity price movements\n"
            "• networks of persons/companies\n"
            "• macroeconomic context\n\n"
            "DATA:\n" + json.dumps([doc for _, doc in all_docs], default=str, ensure_ascii=False)[:MAX_JSON_CHARS]
        )
    
    try:
        resp = gemini_client.models.generate_content(model=GEM_MODEL, contents=prompt)
        insights = resp.text.strip()
        
        # Save insights locally first
        txt_path = os.path.join(OUTPUT_DIR, "insights_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(insights)
        print("✅ Insights created locally →", txt_path)
        
        # Upload insights to Drive
        upload_to_drive(drive_service, result_folder_id, txt_path, "insights_summary.txt", "text/plain")
        
    except Exception as e:
        print(f"❌ Error generating Gemini insights: {e}")

    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print("🧹 Cleaned up temporary files")
    except Exception as e:
        print(f"⚠️  Could not clean up temp files: {e}")

    print("\n🎉 Pipeline complete! All outputs uploaded to Google Drive folder:", result_folder_id)

if __name__ == "__main__":
    main()
