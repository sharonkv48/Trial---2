import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from dateutil import parser
from openai import AzureOpenAI

# =========================================================
# CONFIG
# =========================================================

FINAL_COLUMNS = [
    "account",
    "account description",
    "beginning balance",
    "credit",
    "debit",
    "ending balance",
    "property name",
    "property code",
    "current activity",
    "period_end_date",
    "source_file",
    "source_sheet"
]

IGNORE_SHEET_NAMES = {"upload", "mapping"}
IGNORE_METADATA_KEYWORDS = {"chart of population"}
REQUIRED_HEADER_KEYWORDS = {"account", "balance", "debit", "credit"}
MIN_REQUIRED_HEADERS = 2
MAX_HEADER_ROWS = 3
CONFIDENCE_THRESHOLD = 0.7

PROPERTY_MASTER_PATH = "reference/property_master.xlsx"
PROPERTY_MATCH_COLUMNS = ["B", "F", "N", "O"]  # Excel letters

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("excel_ingestion")

# =========================================================
# ENV + LLM
# =========================================================

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# =========================================================
# HELPERS
# =========================================================

def normalize(text):
    return str(text).lower().strip()

def validate_excel_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".xls", ".xlsx"]:
        raise ValueError("Unsupported file extension")

    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    pd.read_excel(path, engine=engine, nrows=1)
    return engine

def list_excel_files(path):
    if os.path.isfile(path):
        return [path]
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".xls", ".xlsx"))
    ]

def sheet_should_be_ignored(name, df):
    if normalize(name) in IGNORE_SHEET_NAMES:
        return True

    for i in range(min(10, len(df))):
        row_text = " ".join(df.iloc[i].astype(str)).lower()
        if any(k in row_text for k in IGNORE_METADATA_KEYWORDS):
            return True
    return False

def is_header_like_row(row):
    numeric_ratio = pd.to_numeric(row, errors="coerce").notna().mean()
    avg_len = row.astype(str).str.len().mean()
    return numeric_ratio < 0.3 and avg_len < 25

def detect_header_start(df):
    for i, row in df.iterrows():
        text = " ".join(row.astype(str)).lower()
        if sum(k in text for k in REQUIRED_HEADER_KEYWORDS) >= 2:
            return i
    return None

def detect_header_block(df, start):
    rows = [start]
    for i in range(start + 1, min(start + MAX_HEADER_ROWS, len(df))):
        if is_header_like_row(df.iloc[i]):
            rows.append(i)
        else:
            break
    return rows

def merge_headers(df, rows):
    headers = []
    for c in range(df.shape[1]):
        parts = []
        for r in rows:
            val = normalize(df.iloc[r, c])
            if val and val != "nan" and "unnamed" not in val:
                parts.append(val)
        headers.append(" ".join(parts))
    return headers

def find_table_end(df):
    for i, row in df.iterrows():
        if row.astype(str).str.contains("total", case=False).any():
            return i
    return len(df)

# =========================================================
# PROPERTY LOOKUP
# =========================================================

def load_property_master():
    df = pd.read_excel(PROPERTY_MASTER_PATH)
    df.columns = [normalize(c) for c in df.columns]
    return df

PROPERTY_MASTER = load_property_master()

def infer_property(df):
    words = set()

    for col in df.columns:
        sentence = " ".join(df[col].head(10).astype(str))
        words.update(normalize(sentence).split())

    for _, row in PROPERTY_MASTER.iterrows():
        for col_letter in PROPERTY_MATCH_COLUMNS:
            col_idx = ord(col_letter) - ord("A")
            try:
                cell = normalize(row.iloc[col_idx])
                if any(w in cell for w in words):
                    return (
                        row.iloc[1],  # property name (column B)
                        row.iloc[0],  # entity id (column A)
                    )
            except:
                continue
    return None, None

# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_excel_date(df, filename):
    for i in range(min(10, len(df))):
        try:
            return parser.parse(" ".join(df.iloc[i].astype(str)), fuzzy=True)
        except:
            continue
    try:
        return parser.parse(filename, fuzzy=True)
    except:
        return None

# =========================================================
# LLM
# =========================================================

def ask_llm(sample, stats):
    prompt = f"""
Map this financial column.

Sample values: {sample}
Stats: {stats}

Choose one:
["debit", "credit", "beginning balance", "ending balance", "current activity"]

Return JSON only.
"""
    r = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(r.choices[0].message.content)["mapped_to"]

# =========================================================
# CORE PROCESSING
# =========================================================

def process_sheet(df, file_name, sheet_name):
    header_start = detect_header_start(df)
    if header_start is None:
        logger.warning(f"Skipped sheet {sheet_name}: header not found")
        return None

    header_rows = detect_header_block(df, header_start)
    headers = merge_headers(df, header_rows)

    if sum(any(k in h for k in REQUIRED_HEADER_KEYWORDS) for h in headers) < MIN_REQUIRED_HEADERS:
        logger.warning(f"Skipped sheet {sheet_name}: insufficient headers")
        return None

    data = df.iloc[max(header_rows) + 1:].copy()
    data.columns = headers
    data = data.iloc[:find_table_end(data)]

    excel_date = extract_excel_date(df, file_name)
    prop_name, prop_code = infer_property(df)

    clean = pd.DataFrame()
    clean["period_end_date"] = excel_date.strftime("%Y-%m") if excel_date else None
    clean["property name"] = prop_name
    clean["property code"] = prop_code
    clean["source_file"] = file_name
    clean["source_sheet"] = sheet_name

    for c in FINAL_COLUMNS:
        if c not in clean:
            clean[c] = None

    return clean[FINAL_COLUMNS]

# =========================================================
# ENTRY
# =========================================================

def process_input(input_path, output_path):
    all_data = []

    for file in list_excel_files(input_path):
        try:
            engine = validate_excel_file(file)
            xl = pd.ExcelFile(file, engine=engine)
            logger.info(f"Processing file: {file}")

            for sheet in xl.sheet_names:
                df = xl.parse(sheet, header=None).astype(str)

                if sheet_should_be_ignored(sheet, df):
                    logger.info(f"Ignored sheet: {sheet}")
                    continue

                result = process_sheet(df, os.path.basename(file), sheet)
                if result is not None:
                    all_data.append(result)

        except Exception as e:
            logger.error(f"Failed file {file}: {e}")

    if not all_data:
        raise ValueError("No valid data extracted")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_excel(output_path, index=False)
    logger.info(f"Output written to {output_path}")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    process_input("input/", "output/cleaned.xlsx")
