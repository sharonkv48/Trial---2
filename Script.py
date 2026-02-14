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

# Priority 1: RF Raw ID/Entity Name | Priority 2: QIU Name/EntityName
PM_PRIORITY_1 = ["RF Raw ID", "RF Raw Entity Name"] 
PM_PRIORITY_2 = ["QIU Name", "EntityName"]

# =========================================================
# SYSTEM PROMPT (BATCH MAPPING)
# =========================================================

SYSTEM_PROMPT = """You are a financial data mapper. Return STRICT JSON.
Task: Map raw headers to: [account, account description, beginning balance, ending balance]
RULES:
1. 'ending balance'/'closing balance' are synonyms.
2. 'beginning balance'/'forwarding balance' are synonyms.
3. 'account' for codes, 'account description' for text.
Return JSON: {"Raw Header": "category"}"""

# =========================================================
# LOGGING & TRACKING
# =========================================================

stats = {
    "total_files": 0,
    "files_processed": 0,
    "files_skipped": [],
    "total_sheets": 0,
    "sheets_processed": 0,
    "sheets_skipped": [],
    "corrupted_files": 0
}

def setup_logging(input_folder):
    if not os.path.exists("logs"): os.makedirs("logs")
    log_file = f"logs/{os.path.basename(input_folder.strip('/\\'))}.txt"
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger("ingestion")

# =========================================================
# HELPERS
# =========================================================

def normalize_strict(text):
    if pd.isna(text) or str(text).lower() == "nan": return ""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

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
    pats = [r"(\d{4})(\d{2})", r"(\d{2})_(\d{2})_(\d{2})", r"(\d{2})_(\d{2})"]
    for p in pats:
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
            seen[c] = 0; unique.append(c)
        else:
            seen[c] += 1; unique.append(f"{c}_{seen[c]}")
    return unique

# =========================================================
# CORE LOGIC
# =========================================================

def match_property_tiered(df, master_df):
    """Matches keyword in top 20 rows and returns Entity ID from Column E."""
    top_rows = df.head(20).astype(str).values.flatten()
    clean_sheet = [normalize_strict(cell) for cell in top_rows if normalize_strict(cell) != ""]
    
    for group in [PM_PRIORITY_1, PM_PRIORITY_2]:
        for _, row in master_df.iterrows():
            # Strict mapping: Entity ID is in Column E (Index 4)
            p_id = row.iloc[4] if len(row) > 4 else ""
            p_name = row.get("EntityName", "")
            
            for col in group:
                kw = normalize_strict(row.get(col, ""))
                if kw and any(kw == c for c in clean_sheet):
                    return str(p_name), str(p_id)
    return "", ""

def ask_llm_batch(client, deployment, headers_with_samples, report_year):
    msg = f"Report Year: {report_year}\nHeaders:\n{json.dumps(headers_with_samples)}"
    try:
        r = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
            temperature=0, response_format={"type": "json_object"}
        )
        return json.loads(r.choices[0].message.content)
    except: return {}

def process_sheet(df, file_name, sheet_name, master_df, client, deployment, logger):
    stats["total_sheets"] += 1
    
    # 1. Ignore Check
    if any(k in " ".join(df.head(10).astype(str).values.flatten()).lower() for k in IGNORE_KEYWORDS):
        stats["sheets_skipped"].append({"file": file_name, "sheet": sheet_name, "reason": "Chart of Accounts keyword detected"})
        return None

    # 2. Date & Header Detection
    report_date = None
    for i in range(min(20, len(df))):
        line = " ".join(df.iloc[i].astype(str))
        if any(c.isdigit() for c in line):
            report_date = extract_date_flexible(line)
            if report_date: break
    if not report_date: report_date = extract_date_from_filename(file_name)
    report_year = report_date.year if report_date else "Unknown"

    h_start = None
    for i, row in df.iterrows():
        if sum(k in " ".join(row.astype(str)).lower() for k in REQUIRED_HEADER_KEYWORDS) >= 2:
            h_start = i; break
    
    if h_start is None:
        stats["sheets_skipped"].append({"file": file_name, "sheet": sheet_name, "reason": "Insufficient headers found"})
        return None

    # 3. Merge Headers & Extract Data
    h_rows = [h_start]
    for i in range(h_start + 1, min(h_start + MAX_HEADER_ROWS, len(df))):
        if pd.to_numeric(df.iloc[i], errors="coerce").notna().mean() < 0.3: h_rows.append(i)
        else: break
    
    raw_h = [(" ".join([str(df.iloc[r, c]).strip().lower() for r in h_rows if str(df.iloc[r, c]).strip().lower() not in ["nan", ""]]) 
              if any(str(df.iloc[r, c]).strip().lower() not in ["nan", ""] for r in h_rows) else f"empty_gap_{c}") 
             for c in range(df.shape[1])]
    
    headers = make_unique_headers(raw_h)
    data_block = df.iloc[max(h_rows) + 1:].copy()
    data_block.columns = headers
    
    rows = []
    for _, row in data_block.iterrows():
        if any(TOTAL_REGEX.match(str(s)) for s in row): break
        rows.append(row)
    if not rows: 
        stats["sheets_skipped"].append({"file": file_name, "sheet": sheet_name, "reason": "Empty table/Total reached immediately"})
        return None
    final_data = pd.DataFrame(rows)

    # 4. Mapping
    clean = pd.DataFrame(index=final_data.index)
    exact = ["reporting book", "account tree", "location", "database", "report id", "period starting", "debit", "credit"]
    for t in exact:
        col = next((c for c in final_data.columns if c == t), None)
        if col: clean[t] = final_data[col]

    bal_col = next((c for c in final_data.columns if "ending balance" in c or "closing balance" in c), None)
    if bal_col: clean["ending balance"] = final_data[bal_col]
    act_col = next((c for c in final_data.columns if "current activity" in c or c == "activity"), None)
    if act_col: clean["current activity"] = final_data[act_col]

    unmapped = [c for c in final_data.columns if c not in clean.columns and "empty_gap" not in c]
    if unmapped:
        mappings = ask_llm_batch(client, deployment, {c: final_data[c].dropna().head(3).tolist() for c in unmapped}, report_year)
        for raw, target in mappings.items():
            if target in ["account", "account description", "beginning balance", "ending balance"] and target not in clean:
                clean[target] = final_data[raw]

    p_name, p_code = match_property_tiered(df, master_df)
    clean["property name"], clean["property code"] = p_name, p_code
    clean["period ending"] = report_date.strftime("%Y-%m-%d") if report_date else ""
    
    for c in FINAL_COLUMNS:
        if c not in clean: clean[c] = ""
    
    stats["sheets_processed"] += 1
    return clean[FINAL_COLUMNS].replace("nan", "").replace(np.nan, "")

# =========================================================
# RUNNER
# =========================================================

def run_pipeline(input_folder):
    logger = setup_logging(input_folder)
    load_dotenv()
    if not os.path.exists("outputs"): os.makedirs("outputs")
    client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
                         azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"), 
                         api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    master_df = pd.read_excel(PROPERTY_MASTER_PATH) if os.path.exists(PROPERTY_MASTER_PATH) else pd.DataFrame()
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".xls", ".xlsx"))]
    stats["total_files"] = len(files)

    for f in files:
        path = os.path.join(input_folder, f)
        try:
            xl = pd.ExcelFile(path, engine=("openpyxl" if f.endswith(".xlsx") else "xlrd"))
            output_path = f"outputs/{os.path.splitext(f)[0]}_new.xlsx"
            valid_sheets_found = False
            
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                for sheet in xl.sheet_names:
                    if sheet.lower().strip() in IGNORE_SHEET_NAMES:
                        stats["sheets_skipped"].append({"file": f, "sheet": sheet, "reason": "Sheet name in ignore list"})
                        continue
                    
                    res = process_sheet(xl.parse(sheet, header=None).astype(str), f, sheet, master_df, client, deployment, logger)
                    if res is not None:
                        res.to_excel(writer, sheet_name=sheet, index=False)
                        valid_sheets_found = True
            
            if valid_sheets_found: stats["files_processed"] += 1
            else: 
                os.remove(output_path)
                stats["files_skipped"].append({"file": f, "reason": "No valid data extracted from any sheet"})

        except Exception as e:
            stats["corrupted_files"] += 1
            stats["files_skipped"].append({"file": f, "reason": f"Corruption or read error: {str(e)}"})

    # Final Summary
    logger.info("\n" + "="*40 + "\nPROCESSING SUMMARY\n" + "="*40)
    logger.info(f"Total Files Found: {stats['total_files']}")
    logger.info(f"Files Processed: {stats['files_processed']}")
    logger.info(f"Files Skipped: {len(stats['files_skipped'])}")
    logger.info(f"Corrupted Files: {stats['corrupted_files']}")
    for fs in stats["files_skipped"]: logger.info(f"  - [FILE] {fs['file']}: {fs['reason']}")
    
    logger.info("-" * 20)
    logger.info(f"Total Sheets Attempted: {stats['total_sheets']}")
    logger.info(f"Sheets Processed: {stats['sheets_processed']}")
    logger.info(f"Sheets Skipped: {len(stats['sheets_skipped'])}")
    for ss in stats["sheets_skipped"]: logger.info(f"  - [SHEET] [{ss['file']}] {ss['sheet']}: {ss['reason']}")
    logger.info("="*40)

if __name__ == "__main__":
    run_pipeline("input/")
