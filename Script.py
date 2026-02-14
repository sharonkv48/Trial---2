import os
import json
import logging
import re
import pandas as pd
from dotenv import load_dotenv
from dateutil import parser
from openai import AzureOpenAI

# =========================================================
# CONFIG & SCHEMA
# =========================================================

FINAL_COLUMNS = [
    "account", "account description", "beginning balance", "credit", 
    "debit", "ending balance", "property name", "property code", 
    "period starting", "period ending", "reporting book", 
    "account tree", "location", "database", "report id", "current activity"
]

IGNORE_SHEET_NAMES = {"upload", "mapping", "starwood property tb"}
IGNORE_KEYWORDS = {"chart of accounts"}
REQUIRED_HEADER_KEYWORDS = {"account", "balance", "debit", "credit", "activity"}
MAX_HEADER_ROWS = 3
TOTAL_REGEX = re.compile(r"^\s*total[:.;'\s-]*.*", re.IGNORECASE)

PROPERTY_MASTER_PATH = "reference/property_master.xlsx"

# Property Master Headers
PM_ENTITY_ID = "Entity ID"
PM_ENTITY_NAME = "EntityName"
PM_PRIORITY_1 = ["RF Raw ID", "RF Raw Entity Name"] 
PM_PRIORITY_2 = ["QIU Name", "EntityName"]

# =========================================================
# FEW-SHOT SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a financial data mapper. Return STRICT JSON with 'mapped_to' and 'confidence'.
Categories: [account, account description, beginning balance, ending balance]

RULES:
1. 'ending balance' and 'closing balance' are synonymous.
2. 'beginning balance' and 'forwarding balance' are synonymous.
3. Return JSON ONLY: {"mapped_to": "category", "confidence": 0.0-1.0}"""

# =========================================================
# HELPERS & LOGGING
# =========================================================

def setup_logging(input_folder):
    if not os.path.exists("logs"): os.makedirs("logs")
    folder_name = os.path.basename(input_folder.strip("/\\"))
    log_file = f"logs/{folder_name}.txt"
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger("ingestion")

def normalize_strict(text):
    """Normalizes text by lowercase, stripping, and removing all non-alphanumeric characters."""
    if not text or text == "nan": return ""
    # Remove all special characters/punctuation, keep only letters and numbers
    clean = re.sub(r'[^a-zA-Z0-9]', '', str(text))
    return clean.lower()

def validate_excel_file(path):
    ext = os.path.splitext(path)[1].lower()
    return "openpyxl" if ext == ".xlsx" else "xlrd"

def extract_date_from_filename(filename):
    patterns = [r"(\d{4})(\d{2})", r"(\d{2})_(\d{2})_(\d{2})", r"(\d{2})_(\d{2})"]
    for p in patterns:
        match = re.search(p, filename)
        if match:
            try:
                g = match.groups()
                if len(g) == 2:
                    return parser.parse(f"{g[0]}-{g[1]}-01") if len(g[0]) == 4 else parser.parse(f"20{g[1]}-{g[0]}-01")
                return parser.parse(f"20{g[2]}-{g[0]}-{g[1]}")
            except: continue
    return None

def make_unique_headers(headers):
    seen = {}
    unique_headers = []
    for h in headers:
        clean_h = str(h).strip().lower()
        if clean_h not in seen:
            seen[clean_h] = 0
            unique_headers.append(clean_h)
        else:
            seen[clean_h] += 1
            unique_headers.append(f"{clean_h}_{seen[clean_h]}")
    return unique_headers

# =========================================================
# PROPERTY LOOKUP (CLEANED ALPHANUMERIC MATCH)
# =========================================================

def match_property_tiered(df, master_df):
    """Tiered lookup using strict alphanumeric normalization for both sheet and master data."""
    # Flatten top 20 rows and clean every cell
    top_rows = df.head(20).astype(str).values.flatten()
    clean_sheet_text = [normalize_strict(cell) for cell in top_rows if normalize_strict(cell) != ""]
    
    priority_groups = [PM_PRIORITY_1, PM_PRIORITY_2]
    
    for group in priority_groups:
        for _, row in master_df.iterrows():
            prop_id = row.get(PM_ENTITY_ID, None)
            prop_display_name = row.get(PM_ENTITY_NAME, "Unknown") 
            
            for col_name in group:
                keyword = normalize_strict(row.get(col_name, ""))
                if not keyword: continue
                
                # Check if the cleaned keyword exists as a full match in any cleaned cell
                if any(keyword == cell for cell in clean_sheet_text):
                    return prop_display_name, prop_id
    return None, None

# =========================================================
# AI MAPPING
# =========================================================

def ask_llm(client, deployment, col_name, sample, report_year):
    user_msg = f"Report Year: {report_year}\nHeader: {col_name}\nSamples: {sample}"
    try:
        r = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0, response_format={"type": "json_object"}
        )
        return json.loads(r.choices[0].message.content)
    except: return {"mapped_to": "other", "confidence": 0}

# =========================================================
# CORE PROCESSING
# =========================================================

def process_sheet(df, file_name, sheet_name, master_df, client, deployment, logger):
    # Ignore Keyword Check
    top_text = " ".join(df.head(10).astype(str).values.flatten()).lower()
    if any(k in top_text for k in IGNORE_KEYWORDS): return None

    # Date Detection
    report_date = extract_date_from_filename(file_name)
    report_year = report_date.year if report_date else "Unknown"

    # Header Detection
    header_start = None
    for i, row in df.iterrows():
        text = " ".join(row.astype(str)).lower()
        if sum(k in text for k in REQUIRED_HEADER_KEYWORDS) >= 2:
            header_start = i
            break
    if header_start is None: return None

    # Merge Multi-Row Headers
    header_rows = [header_start]
    for i in range(header_start + 1, min(header_start + MAX_HEADER_ROWS, len(df))):
        if pd.to_numeric(df.iloc[i], errors="coerce").notna().mean() < 0.3:
            header_rows.append(i)
        else: break
    
    raw_headers = []
    for c in range(df.shape[1]):
        parts = [str(df.iloc[r, c]).strip().lower() for r in header_rows if str(df.iloc[r, c]).strip().lower() not in ["nan", ""]]
        raw_headers.append(" ".join(parts) if parts else f"empty_gap_{c}")
    
    headers = make_unique_headers(raw_headers)

    data_block = df.iloc[max(header_rows) + 1:].copy()
    data_block.columns = headers
    
    rows_list = []
    for _, row in data_block.iterrows():
        if any(TOTAL_REGEX.match(str(s)) for s in row): break
        rows_list.append(row)
    if not rows_list: return None
    final_data = pd.DataFrame(rows_list)

    prop_name, prop_code = match_property_tiered(df, master_df)
    clean = pd.DataFrame(index=final_data.index)
    
    # EXACT KEYWORD MATCHES
    exact_targets = ["reporting book", "account tree", "location", "database", "report id", "period starting", "debit", "credit"]
    for target in exact_targets:
        col = next((c for c in final_data.columns if c == target), None)
        if col: clean[target] = final_data[col]

    # Map Closing Balance to Ending Balance
    bal_col = next((c for c in final_data.columns if "ending balance" in c or "closing balance" in c), None)
    if bal_col: clean["ending balance"] = final_data[bal_col]

    act_col = next((c for c in final_data.columns if "current activity" in c or c == "activity"), None)
    if act_col: clean["current activity"] = final_data[act_col]

    # LLM Mapping
    llm_targets = ["account", "account description", "beginning balance", "ending balance"]
    for col in final_data.columns:
        if "empty_gap" in col or col in clean.columns: continue
        res = ask_llm(client, deployment, col, final_data[col].dropna().head(3).tolist(), report_year)
        if res.get('mapped_to') in llm_targets and res.get('confidence', 0) > 0.6:
            if res['mapped_to'] not in clean: clean[res['mapped_to']] = final_data[col]

    clean["property name"] = prop_name
    clean["property code"] = prop_code
    clean["period ending"] = report_date.strftime("%Y-%m-%d") if report_date else None
    
    for c in FINAL_COLUMNS:
        if c not in clean: clean[c] = None
    
    return clean[FINAL_COLUMNS]

def run_pipeline(input_folder):
    logger = setup_logging(input_folder)
    load_dotenv()
    output_folder = "outputs"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Updated API Keys per request
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"), 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    master_df = pd.read_excel(PROPERTY_MASTER_PATH) if os.path.exists(PROPERTY_MASTER_PATH) else pd.DataFrame()
    
    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        if not f.lower().endswith((".xls", ".xlsx")): continue
        file_results = []
        try:
            xl = pd.ExcelFile(path, engine=validate_excel_file(path))
            for sheet in xl.sheet_names:
                if sheet.lower().strip() in IGNORE_SHEET_NAMES: continue
                df = xl.parse(sheet, header=None).astype(str)
                res = process_sheet(df, f, sheet, master_df, client, deployment, logger)
                if res is not None: file_results.append(res)
            
            if file_results:
                out_path = os.path.join(output_folder, f"{os.path.splitext(f)[0]}_new.xlsx")
                pd.concat(file_results, ignore_index=True).to_excel(out_path, index=False)
                logger.info(f"SUCCESS: {f}")
        except Exception as e: logger.error(f"FAILED {f}: {e}")

if __name__ == "__main__":
    run_pipeline("input/")
