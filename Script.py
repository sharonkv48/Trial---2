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

# Priority 1: N (13) and O (14) | Priority 2: B (1) and F (5)
PRIORITY_GROUPS = [[13, 14], [1, 5]]

# =========================================================
# FEW-SHOT SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a financial data mapper. Return STRICT JSON with 'mapped_to' and 'confidence'.
Categories: [account, account description, beginning balance, ending balance]

FEW-SHOT EXAMPLES:
- Input: '1116', '7627', '1010-6172' -> {"mapped_to": "account", "confidence": 1.0}
- Input: 'Checking Operating 1 - HQ Level', 'Maintainance Equipment' -> {"mapped_to": "account description", "confidence": 1.0}
- Input: 'Forwarding Balance', 'Opening Balance' -> {"mapped_to": "beginning balance", "confidence": 0.95}
- Input: 'Ending Balance June 2025' (for Report Year 2025) -> {"mapped_to": "ending balance", "confidence": 1.0}

Rules:
1. 'ending balance' MUST match the Report Reference Year.
2. If column is unrelated or year mismatch, return 'other'.
3. Return JSON ONLY: {"mapped_to": "category", "confidence": 0.0-1.0}"""

# =========================================================
# HELPERS & LOGGING
# =========================================================

def setup_logging(input_folder):
    if not os.path.exists("logs"): 
        os.makedirs("logs")
    folder_name = os.path.basename(input_folder.strip("/\\"))
    log_file = f"logs/{folder_name}.txt"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger("ingestion")

def normalize(text):
    return str(text).lower().strip()

def validate_excel_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx": return "openpyxl"
    if ext == ".xls": return "xlrd"
    raise ValueError(f"Unsupported format: {ext}")

def extract_date_from_filename(filename):
    patterns = [
        r"(\d{4})(\d{2})",           # 202506
        r"(\d{2})_(\d{2})_(\d{2})",  # 06_30_25
        r"(\d{2})_(\d{2})",           # 06_25
    ]
    for p in patterns:
        match = re.search(p, filename)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 2:
                    if len(groups[0]) == 4:
                        return parser.parse(f"{groups[0]}-{groups[1]}-01")
                    else:
                        return parser.parse(f"20{groups[1]}-{groups[0]}-01")
                elif len(groups) == 3:
                    return parser.parse(f"20{groups[2]}-{groups[0]}-{groups[1]}")
            except: continue
    return None

# =========================================================
# AI & PROPERTY LOGIC
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

def match_property_tiered(df, master_df):
    sample_cells = [normalize(c) for c in df.head(15).astype(str).values.flatten()]
    for group in PRIORITY_GROUPS:
        for _, row in master_df.iterrows():
            p_code, p_name_master = row.iloc[0], normalize(row.iloc[1])
            for idx in group:
                val = normalize(row.iloc[idx])
                if not val or val == "nan": continue
                pattern = re.compile(rf"^\s*{re.escape(val)}\s*$", re.IGNORECASE)
                if any(pattern.fullmatch(cell) for cell in sample_cells):
                    return p_name_master, p_code
    return None, None

# =========================================================
# PROCESSING ENGINE
# =========================================================

def process_sheet(df, file_name, sheet_name, master_df, client, deployment, logger):
    top_text = " ".join(df.head(10).astype(str).values.flatten()).lower()
    if any(k in top_text for k in IGNORE_KEYWORDS):
        logger.warning(f"SKIPPED: {file_name} [{sheet_name}] - Chart of Accounts detected.")
        return None

    report_date = None
    for i in range(min(15, len(df))):
        try:
            line = " ".join(df.iloc[i].astype(str))
            if any(c.isdigit() for c in line):
                report_date = parser.parse(line, fuzzy=True)
                if report_date: break
        except: continue
    
    if not report_date:
        report_date = extract_date_from_filename(file_name)
    report_year = report_date.year if report_date else "Unknown"

    header_start = None
    for i, row in df.iterrows():
        text = " ".join(row.astype(str)).lower()
        if sum(k in text for k in REQUIRED_HEADER_KEYWORDS) >= 2:
            header_start = i
            break
    if header_start is None: return None

    header_rows = [header_start]
    for i in range(header_start + 1, min(header_start + MAX_HEADER_ROWS, len(df))):
        if pd.to_numeric(df.iloc[i], errors="coerce").notna().mean() < 0.3:
            header_rows.append(i)
        else: break
    
    headers = []
    for c in range(df.shape[1]):
        parts = [normalize(df.iloc[r, c]) for r in header_rows if normalize(df.iloc[r, c]) not in ["nan", ""]]
        # Handling gaps: if a column is empty across all header rows, label it 'empty_gap'
        headers.append(" ".join(parts) if parts else f"empty_gap_{c}")

    data_block = df.iloc[max(header_rows) + 1:].copy()
    data_block.columns = headers
    valid_rows = []
    for _, row in data_block.iterrows():
        if any(TOTAL_REGEX.match(str(s)) for s in row): break
        valid_rows.append(row)
    if not valid_rows: return None
    final_data = pd.DataFrame(valid_rows)

    prop_name, prop_code = match_property_tiered(df, master_df)
    clean = pd.DataFrame(index=final_data.index)
    
    # EXACT KEYWORD MATCHES (Collapses gaps because we search by name, not index)
    exact_targets = ["reporting book", "account tree", "location", "database", "report id", "period starting", "debit", "credit"]
    for target in exact_targets:
        col = next((c for c in final_data.columns if normalize(c) == target), None)
        if col: clean[target] = final_data[col]

    activity_col = next((c for c in final_data.columns if "current activity" in normalize(c) or normalize(c) == "activity"), None)
    if activity_col: clean["current activity"] = final_data[activity_col]

    # LLM MAPPING
    llm_targets = ["account", "account description", "beginning balance", "ending balance"]
    for col in final_data.columns:
        # Skip gaps and already mapped columns
        if "empty_gap" in col or col in clean.columns: continue
        
        sample = final_data[col].dropna().head(3).tolist()
        res = ask_llm(client, deployment, col, sample, report_year)
        if res.get('mapped_to') in llm_targets and res.get('confidence', 0) > 0.7:
            if res['mapped_to'] not in clean: clean[res['mapped_to']] = final_data[col]

    clean["property name"] = prop_name
    clean["property code"] = prop_code
    clean["period ending"] = report_date.strftime("%Y-%m-%d") if report_date else None
    
    for c in FINAL_COLUMNS:
        if c not in clean: clean[c] = None
    
    return clean[FINAL_COLUMNS]

# =========================================================
# MAIN EXECUTION
# =========================================================

def run_pipeline(input_folder):
    logger = setup_logging(input_folder)
    load_dotenv()
    
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-05-15"
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    master_df = pd.read_excel(PROPERTY_MASTER_PATH) if os.path.exists(PROPERTY_MASTER_PATH) else pd.DataFrame()
    
    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        if not f.lower().endswith((".xls", ".xlsx")): continue
        
        logger.info(f"PROCESSING: {f}")
        file_results = []
        try:
            xl = pd.ExcelFile(path, engine=validate_excel_file(path))
            for sheet in xl.sheet_names:
                if normalize(sheet) in IGNORE_SHEET_NAMES: continue
                df = xl.parse(sheet, header=None).astype(str)
                res = process_sheet(df, f, sheet, master_df, client, deployment, logger)
                if res is not None: file_results.append(res)
            
            if file_results:
                base_name = os.path.splitext(f)[0]
                output_path = os.path.join(output_folder, f"{base_name}_new.xlsx")
                pd.concat(file_results, ignore_index=True).to_excel(output_path, index=False)
                logger.info(f"SAVED: {output_path}")
        except Exception as e: 
            logger.error(f"FAILED {f}: {e}")

if __name__ == "__main__":
    run_pipeline("input/")
