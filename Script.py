import os
import json
import logging
import re
import pandas as pd
import numpy as np
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
# SYSTEM PROMPT (BATCH MAPPING)
# =========================================================

SYSTEM_PROMPT = """You are a financial data mapper. Return STRICT JSON.
Task: Map a list of raw headers to: [account, account description, beginning balance, ending balance]

RULES:
1. 'ending balance' and 'closing balance' are synonymous.
2. 'beginning balance' and 'forwarding balance' are synonymous.
3. Use 'account' for numeric codes (e.g., 1116, 1010-6172).
4. Use 'account description' for text labels (e.g., Checking Operating).
5. Return a JSON object where keys are raw headers and values are the categories or "other".
Example: {"Raw Header 1": "account", "Raw Header 2": "other"}"""

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
    if pd.isna(text) or str(text).lower() == "nan": return ""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def validate_excel_file(path):
    ext = os.path.splitext(path)[1].lower()
    return "openpyxl" if ext == ".xlsx" else "xlrd"

def extract_date_flexible(text):
    try:
        match = re.search(r"(\d{2})[/_-](\d{2,4})", text)
        if match:
            m, y = match.groups()
            y = f"20{y}" if len(y) == 2 else y
            return parser.parse(f"{y}-{m}-01")
        return parser.parse(text, fuzzy=True)
    except: return None

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
    seen, unique = {}, []
    for h in headers:
        c = str(h).strip().lower()
        if c not in seen:
            seen[c] = 0
            unique.append(c)
        else:
            seen[c] += 1
            unique.append(f"{c}_{seen[c]}")
    return unique

# =========================================================
# BATCH AI MAPPING (One call per sheet)
# =========================================================

def ask_llm_batch(client, deployment, headers_with_samples, report_year):
    """Sends all headers in one prompt to minimize API calls."""
    user_msg = f"Report Year: {report_year}\nHeaders and Samples:\n{json.dumps(headers_with_samples)}"
    try:
        r = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0, response_format={"type": "json_object"}
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {}

# =========================================================
# PROPERTY LOOKUP
# =========================================================

def match_property_tiered(df, master_df):
    top_rows = df.head(20).astype(str).values.flatten()
    clean_sheet = [normalize_strict(cell) for cell in top_rows if normalize_strict(cell) != ""]
    for group in [PM_PRIORITY_1, PM_PRIORITY_2]:
        for _, row in master_df.iterrows():
            tid, tname = row.get(PM_ENTITY_ID, ""), row.get(PM_ENTITY_NAME, "")
            for col in group:
                kw = normalize_strict(row.get(col, ""))
                if kw and any(kw == c for c in clean_sheet):
                    return str(tname), str(tid)
    return "", ""

# =========================================================
# CORE PROCESSING
# =========================================================

def process_sheet(df, file_name, sheet_name, master_df, client, deployment, logger):
    if any(k in " ".join(df.head(10).astype(str).values.flatten()).lower() for k in IGNORE_KEYWORDS): return None

    # Date Detection
    report_date = None
    for i in range(min(20, len(df))):
        line = " ".join(df.iloc[i].astype(str))
        if any(c.isdigit() for c in line):
            report_date = extract_date_flexible(line)
            if report_date: break
    if not report_date: report_date = extract_date_from_filename(file_name)
    report_year = report_date.year if report_date else "Unknown"

    # Header Detection
    h_start = None
    for i, row in df.iterrows():
        if sum(k in " ".join(row.astype(str)).lower() for k in REQUIRED_HEADER_KEYWORDS) >= 2:
            h_start = i; break
    if h_start is None: return None

    # Merge Headers
    h_rows = [h_start]
    for i in range(h_start + 1, min(h_start + MAX_HEADER_ROWS, len(df))):
        if pd.to_numeric(df.iloc[i], errors="coerce").notna().mean() < 0.3: h_rows.append(i)
        else: break
    
    raw_h = []
    for c in range(df.shape[1]):
        parts = [str(df.iloc[r, c]).strip().lower() for r in h_rows if str(df.iloc[r, c]).strip().lower() not in ["nan", ""]]
        raw_h.append(" ".join(parts) if parts else f"empty_gap_{c}")
    
    headers = make_unique_headers(raw_h)
    data_block = df.iloc[max(h_rows) + 1:].copy()
    data_block.columns = headers
    
    valid_rows = []
    for _, row in data_block.iterrows():
        if any(TOTAL_REGEX.match(str(s)) for s in row): break
        valid_rows.append(row)
    if not valid_rows: return None
    final_data = pd.DataFrame(valid_rows)

    # 1. Exact Metadata Mapping
    clean = pd.DataFrame(index=final_data.index)
    exact = ["reporting book", "account tree", "location", "database", "report id", "period starting", "debit", "credit"]
    for t in exact:
        col = next((c for c in final_data.columns if c == t), None)
        if col: clean[t] = final_data[col]

    # 2. Activity & Balance Keyword Mapping
    bal_col = next((c for c in final_data.columns if "ending balance" in c or "closing balance" in c), None)
    if bal_col: clean["ending balance"] = final_data[bal_col]
    act_col = next((c for c in final_data.columns if "current activity" in c or c == "activity"), None)
    if act_col: clean["current activity"] = final_data[act_col]

    # 3. Batch LLM Call for remaining columns (Only one call here)
    unmapped = [c for c in final_data.columns if c not in clean.columns and "empty_gap" not in c]
    if unmapped:
        header_samples = {c: final_data[c].dropna().head(3).tolist() for c in unmapped}
        mappings = ask_llm_batch(client, deployment, header_samples, report_year)
        for raw_col, target_cat in mappings.items():
            if target_cat in ["account", "account description", "beginning balance", "ending balance"] and target_cat not in clean:
                clean[target_cat] = final_data[raw_col]

    # 4. Final Metadata
    p_name, p_code = match_property_tiered(df, master_df)
    clean["property name"], clean["property code"] = p_name, p_code
    clean["period ending"] = report_date.strftime("%Y-%m-%d") if report_date else ""
    
    for c in FINAL_COLUMNS:
        if c not in clean: clean[c] = ""
    
    return clean[FINAL_COLUMNS].replace("nan", "").replace(np.nan, "")

def run_pipeline(input_folder):
    logger = setup_logging(input_folder)
    load_dotenv()
    if not os.path.exists("outputs"): os.makedirs("outputs")
    client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
                         azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"), 
                         api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
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
                res = process_sheet(xl.parse(sheet, header=None).astype(str), f, sheet, master_df, client, deployment, logger)
                if res is not None: file_results.append(res)
            if file_results:
                pd.concat(file_results, ignore_index=True).to_excel(f"outputs/{os.path.splitext(f)[0]}_new.xlsx", index=False)
                logger.info(f"SUCCESS: {f}")
        except Exception as e: logger.error(f"FAILED {f}: {e}")

if __name__ == "__main__":
    run_pipeline("input/")
