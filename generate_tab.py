#!/usr/bin/env python3
"""
generate_tab.py  —  One-file blueprint PDF -> TAB JSON + TAB Report PDF (local Ollama)

Requirements (in your venv):
  pip install pymupdf requests reportlab

Usage:
  python3 generate_tab.py "/path/to/Blueprint.pdf"

Outputs:
  - master_tab.json
  - tab_report.pdf

Notes:
  - Works best for vector PDFs (born-digital). Scanned PDFs require OCR or vision.
  - The extractor is generic: it reconstructs tables using word x/y positions.
  - LLM (Ollama) is used for semantic mapping into a canonical TAB schema.
"""

import sys
import os
import re
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
import requests

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# -----------------------------
# Config
# -----------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Controls how we group words into "lines"
LINE_Y_TOL = float(os.environ.get("LINE_Y_TOL", "6.0"))

# Controls how we stop table parsing
MAX_TABLE_LINES = int(os.environ.get("MAX_TABLE_LINES", "300"))

# LLM safety/robustness
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "600"))
LLM_NUM_PREDICT = int(os.environ.get("LLM_NUM_PREDICT", "1400"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))

# Chunk sizes for schedule rows sent to LLM
ROWS_PER_CHUNK = int(os.environ.get("ROWS_PER_CHUNK", "25"))

# Schedule keywords (header detection)
SCHEDULE_KEYWORDS = [
    ("FAN_SCHEDULE", ["FAN SCHEDULE"]),
    ("VAV_SCHEDULE", ["VAV SCHEDULE", "AIR TERMINAL SCHEDULE"]),
    ("EQUIPMENT_SCHEDULE", ["EQUIPMENT SCHEDULE"]),
    ("AIRFLOW_SCHEDULE", ["AIRFLOW SCHEDULE", "AIR FLOW SCHEDULE"]),
    ("DIFFUSER_SCHEDULE", ["DIFFUSER SCHEDULE", "GRILLE SCHEDULE", "REGISTER SCHEDULE"]),
]

# Canonical output schema (extend as needed)
def empty_master() -> Dict[str, Any]:
    return {
        "project": {
            "project_name": "",
            "client": "",
            "project_address": "",
            "project_number": "",
            "issue_date": "",
            "permit": "",
            "architect": "",
            "mep_engineer": "",
            "structural_engineer": ""
        },
        "schedules": {
            "fans": [],
            "vavs": [],
            "equipment": [],
            "airflow": [],
            "diffusers": [],
            "unknown": []  # store raw tables that we couldn't map
        },
        "lineage": {
            "source_pdf": "",
            "generated_at": "",
            "model": OLLAMA_MODEL
        }
    }


# -----------------------------
# Utilities
# -----------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # pull first numeric token
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))

def extract_first_json_object(raw: str) -> Dict[str, Any]:
    """
    Robust JSON extraction: model sometimes returns extra text or truncation attempts.
    We extract the first balanced {...} object.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i+1]
                return json.loads(candidate)

    raise ValueError("Unclosed JSON object in LLM output (likely truncated)")


def ollama_json(prompt: str, *, num_predict: int = LLM_NUM_PREDICT) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": num_predict,
            "repeat_penalty": 1.1
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_S)
    r.raise_for_status()
    raw = r.json().get("response", "")
    if not raw:
        raise ValueError("Empty Ollama response")
    try:
        return extract_first_json_object(raw)
    except Exception as e:
        # Save debug
        with open("ollama_bad_json.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise ValueError(f"Failed to parse JSON from LLM (wrote ollama_bad_json.txt): {e}") from e


# -----------------------------
# Table reconstruction from PDF coordinates (generic)
# -----------------------------
@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

def page_words(page: fitz.Page) -> List[Word]:
    words = []
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        t = (w or "").strip()
        if t:
            words.append(Word(x0, y0, x1, y1, t))
    return words

def group_words_into_lines(words: List[Word], y_tol: float = LINE_Y_TOL) -> List[List[Word]]:
    """
    Groups words by baseline-ish y coordinate.
    """
    buckets: Dict[float, List[Word]] = {}
    for wd in words:
        y_key = round(wd.y0 / y_tol) * y_tol
        buckets.setdefault(y_key, []).append(wd)

    lines = []
    for y_key in sorted(buckets.keys()):
        line = sorted(buckets[y_key], key=lambda w: w.x0)
        lines.append(line)
    return lines

def line_text(line: List[Word]) -> str:
    return " ".join(w.text for w in line)

def detect_schedule_type(page_text_top: str) -> Optional[str]:
    t = page_text_top.upper()
    for sched_type, phrases in SCHEDULE_KEYWORDS:
        for p in phrases:
            if p.upper() in t:
                return sched_type
    return None

def find_header_line_index(lines: List[List[Word]], sched_type: str) -> Optional[int]:
    """
    Heuristic: find the line that looks like a header row.
    For schedules, it often contains tokens like DESIGNATION/LOCATION/TYPE/CFM/ESP/etc.
    """
    header_tokens = {
        "FAN_SCHEDULE": ["DESIGNATION", "LOCATION", "TYPE", "AIRFLOW", "CFM", "ESP", "WG", "MOTOR", "POWER", "SPEED", "RPM"],
        "VAV_SCHEDULE": ["TAG", "VAV", "CFM", "NECK", "SIZE", "THROW", "APD", "MAX", "MIN", "MODEL"],
        "EQUIPMENT_SCHEDULE": ["MARK", "TAG", "EQUIPMENT", "MODEL", "MANUFACTURER", "CFM", "HP", "VOLT", "PH", "HZ"],
        "AIRFLOW_SCHEDULE": ["ROOM", "SUPPLY", "RETURN", "EXHAUST", "CFM"],
        "DIFFUSER_SCHEDULE": ["TYPE", "SIZE", "NECK", "CFM", "THROW", "MODEL", "MANUFACTURER"]
    }
    tokens = header_tokens.get(sched_type, ["SCHEDULE", "CFM", "TYPE", "MODEL"])
    tokens = [t.upper() for t in tokens]

    best_i = None
    best_score = 0
    for i, ln in enumerate(lines[:80]):  # headers near top
        txt = line_text(ln).upper()
        score = sum(1 for tok in tokens if tok in txt)
        # also favor lines with many words
        score += min(len(ln), 10) * 0.1
        if score > best_score:
            best_score = score
            best_i = i

    # Require minimal confidence
    if best_i is not None and best_score >= 2.5:
        return best_i
    return None

def cluster_x_positions(xs: List[float], merge_tol: float = 18.0) -> List[float]:
    """
    Cluster x positions into column anchors (generic).
    merge_tol in points; increase if columns are wider/spaced.
    """
    xs = sorted(xs)
    if not xs:
        return []
    clusters = [xs[0]]
    for x in xs[1:]:
        if abs(x - clusters[-1]) <= merge_tol:
            # merge by averaging
            clusters[-1] = (clusters[-1] + x) / 2.0
        else:
            clusters.append(x)
    return clusters

def line_to_cells(line: List[Word], col_anchors: List[float]) -> List[str]:
    """
    Assign each word to nearest column anchor by x0.
    Returns list of cell strings aligned to col_anchors.
    """
    if not col_anchors:
        return [line_text(line)]
    cells: List[List[Word]] = [[] for _ in col_anchors]
    for w in line:
        # find nearest anchor by x0
        j = min(range(len(col_anchors)), key=lambda k: abs(w.x0 - col_anchors[k]))
        cells[j].append(w)
    out = []
    for cell_words in cells:
        cell_words = sorted(cell_words, key=lambda ww: ww.x0)
        out.append(normalize_space(" ".join(ww.text for ww in cell_words)))
    return out

def extract_table(page: fitz.Page, sched_type: str) -> Optional[Dict[str, Any]]:
    """
    Generic schedule table extractor:
      - find header line
      - build column anchors from header word x positions
      - parse subsequent lines into rows (until a stop condition)
    Returns dict: {schedule_type, headers, rows}
    """
    words = page_words(page)
    if not words:
        return None
    lines = group_words_into_lines(words, y_tol=LINE_Y_TOL)
    if not lines:
        return None

    header_i = find_header_line_index(lines, sched_type)
    if header_i is None:
        return None

    header_line = lines[header_i]
    header_txt = normalize_space(line_text(header_line))
    # Build column anchors from header word x positions
    xs = [w.x0 for w in header_line]
    col_anchors = cluster_x_positions(xs, merge_tol=18.0)

    # Headers: use cells from header line
    header_cells = line_to_cells(header_line, col_anchors)
    # Clean empty headers
    header_cells = [hc if hc else f"COL_{idx+1}" for idx, hc in enumerate(header_cells)]

    # Parse rows from subsequent lines
    rows = []
    stop_hits = 0
    for li in range(header_i + 1, min(header_i + 1 + MAX_TABLE_LINES, len(lines))):
        ln = lines[li]
        txt = normalize_space(line_text(ln))
        if not txt:
            stop_hits += 1
            if stop_hits >= 3:
                break
            continue

        # Stop if we hit another schedule title
        up = txt.upper()
        if "SCHEDULE" in up and li < header_i + 20:
            # sometimes schedule title wraps; ignore near header
            pass
        elif "SCHEDULE" in up and li > header_i + 20:
            break

        stop_hits = 0
        cells = line_to_cells(ln, col_anchors)
        # If the line is mostly empty, skip
        nonempty = sum(1 for c in cells if c)
        if nonempty <= 1:
            continue

        # Build row dict keyed by header
        row = {}
        for h, c in zip(header_cells, cells):
            row[h] = c
        rows.append(row)

    if not rows:
        return None

    return {
        "schedule_type": sched_type,
        "headers": header_cells,
        "rows": rows,
        "header_text": header_txt
    }


# -----------------------------
# LLM mapping: raw table -> canonical TAB schedule items
# -----------------------------
def llm_map_schedule_to_canonical(schedule: Dict[str, Any], *, page_num: int) -> Dict[str, Any]:
    """
    Send headers + rows (chunked) to LLM. LLM returns canonical items for that schedule type.
    We keep it flexible: headers and names can vary; the LLM maps them.
    """
    sched_type = schedule["schedule_type"]
    headers = schedule["headers"]
    rows = schedule["rows"]

    # We chunk rows to avoid truncation / bad JSON
    mapped_items_all = []

    for chunk_start in range(0, len(rows), ROWS_PER_CHUNK):
        chunk = rows[chunk_start:chunk_start + ROWS_PER_CHUNK]

        prompt = f"""
You are mapping a mechanical drawing schedule table into a canonical TAB schema.

Schedule type: {sched_type}

Table headers (as seen in PDF):
{json.dumps(headers, indent=2)}

Table rows (each is header->cell string):
{json.dumps(chunk, indent=2)}

Return STRICT JSON ONLY with this shape:

{{
  "mapped": {{
    "fans": [],
    "vavs": [],
    "equipment": [],
    "airflow": [],
    "diffusers": [],
    "unknown": []
  }}
}}

Mapping rules:
- Use the schedule type to decide which list to populate.
- Map whatever headers exist to the closest canonical fields.
- If a value is missing, use "" for strings and null for numbers.
- Convert numeric strings to numbers where appropriate.
- Do NOT invent values.

Canonical schemas:

fans item example:
{{
  "designation": "",
  "location": "",
  "type": "",
  "drive": "",
  "airflow_cfm": null,
  "esp_in_wg": null,
  "motor_power_w": null,
  "speed_rpm": null,
  "electrical_data": "",
  "weight_lbs": null,
  "control": "",
  "basis_of_design": "",
  "notes": "",
  "source": {{"page": {page_num}}}
}}

vavs item example:
{{
  "tag": "",
  "location": "",
  "cfm_min": null,
  "cfm_max": null,
  "neck_size": "",
  "model": "",
  "manufacturer": "",
  "notes": "",
  "source": {{"page": {page_num}}}
}}

equipment item example:
{{
  "tag": "",
  "type": "",
  "location": "",
  "manufacturer": "",
  "model": "",
  "capacity": "",
  "power": "",
  "notes": "",
  "source": {{"page": {page_num}}}
}}

airflow item example:
{{
  "room": "",
  "supply_cfm": null,
  "return_cfm": null,
  "exhaust_cfm": null,
  "notes": "",
  "source": {{"page": {page_num}}}
}}

diffusers item example:
{{
  "tag": "",
  "type": "",
  "size": "",
  "neck": "",
  "cfm": null,
  "throw": "",
  "model": "",
  "manufacturer": "",
  "notes": "",
  "source": {{"page": {page_num}}}
}}

If you cannot confidently map this schedule, put a compact representation into "unknown" like:
{{
  "schedule_type": "{sched_type}",
  "headers": [...],
  "rows": [...],
  "source": {{"page": {page_num}}}
}}

Return valid JSON. No commentary.
""".strip()

        out = ollama_json(prompt, num_predict=LLM_NUM_PREDICT)
        mapped = out.get("mapped", {})
        mapped_items_all.append(mapped)

    # Merge chunks
    merged = {"fans": [], "vavs": [], "equipment": [], "airflow": [], "diffusers": [], "unknown": []}
    for m in mapped_items_all:
        for k in merged.keys():
            if isinstance(m.get(k), list):
                merged[k].extend(m[k])

    # Post-coercion: ensure numeric fields are numeric
    for fan in merged["fans"]:
        for k in ["airflow_cfm", "esp_in_wg", "motor_power_w", "speed_rpm", "weight_lbs"]:
            if k in fan:
                fan[k] = safe_float(fan[k])
    for v in merged["vavs"]:
        for k in ["cfm_min", "cfm_max"]:
            if k in v:
                v[k] = safe_float(v[k])
    for a in merged["airflow"]:
        for k in ["supply_cfm", "return_cfm", "exhaust_cfm"]:
            if k in a:
                a[k] = safe_float(a[k])
    for d in merged["diffusers"]:
        if "cfm" in d:
            d["cfm"] = safe_float(d["cfm"])

    return merged


# -----------------------------
# Project info extraction (title block / cover)
# -----------------------------
def llm_extract_project_info(first_pages_text: str) -> Dict[str, Any]:
    prompt = f"""
Extract project metadata from this blueprint PDF text.

Return STRICT JSON ONLY:
{{
  "project_name": "",
  "client": "",
  "project_address": "",
  "project_number": "",
  "issue_date": "",
  "permit": "",
  "architect": "",
  "mep_engineer": "",
  "structural_engineer": ""
}}

Rules:
- Do not invent values. If missing, use "".
- Prefer values that appear explicitly.

Text:
{first_pages_text[:6000]}
""".strip()
    out = ollama_json(prompt, num_predict=900)
    # normalize keys
    keys = ["project_name","client","project_address","project_number","issue_date","permit","architect","mep_engineer","structural_engineer"]
    return {k: normalize_space(out.get(k, "")) for k in keys}


# -----------------------------
# Simple TAB report PDF generator
# -----------------------------
def render_tab_report_pdf(master: Dict[str, Any], out_path: str = "tab_report.pdf"):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.75 * inch
    line_h = 12

    def draw_line(txt: str, bold: bool = False):
        nonlocal y
        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(x, y, txt[:120])
        y -= line_h

    proj = master["project"]
    draw_line("TAB REPORT (Auto-generated)", bold=True)
    draw_line(f"Source PDF: {master['lineage'].get('source_pdf','')}")
    draw_line(f"Generated: {master['lineage'].get('generated_at','')}")
    draw_line(f"Model: {master['lineage'].get('model','')}")
    y -= 6

    draw_line("PROJECT INFO", bold=True)
    for k, label in [
        ("project_name","Project Name"),
        ("client","Client"),
        ("project_address","Address"),
        ("project_number","Project Number"),
        ("issue_date","Issue Date"),
        ("permit","Permit"),
        ("architect","Architect"),
        ("mep_engineer","MEP Engineer"),
        ("structural_engineer","Structural Engineer"),
    ]:
        v = proj.get(k, "")
        if v:
            draw_line(f"{label}: {v}")

    y -= 8

    def section(title: str, items: List[Dict[str, Any]], fields: List[str]):
        nonlocal y
        draw_line(title, bold=True)
        if not items:
            draw_line("(none)")
            y -= 6
            return
        for it in items:
            # compact one-line representation
            parts = []
            for f in fields:
                val = it.get(f, "")
                if val is None or val == "":
                    continue
                parts.append(f"{f}={val}")
            src = it.get("source", {})
            if src:
                parts.append(f"source_page={src.get('page','')}")
            draw_line(" - " + ", ".join(parts))
        y -= 6

    sch = master["schedules"]
    section("FANS", sch["fans"], ["designation","location","type","drive","airflow_cfm","esp_in_wg","motor_power_w","speed_rpm","electrical_data","weight_lbs","control","basis_of_design","notes"])
    section("VAVS", sch["vavs"], ["tag","location","cfm_min","cfm_max","neck_size","manufacturer","model","notes"])
    section("EQUIPMENT", sch["equipment"], ["tag","type","location","manufacturer","model","capacity","power","notes"])
    section("AIRFLOW", sch["airflow"], ["room","supply_cfm","return_cfm","exhaust_cfm","notes"])
    section("DIFFUSERS", sch["diffusers"], ["tag","type","size","neck","cfm","throw","manufacturer","model","notes"])

    # Unknown schedules
    draw_line("UNKNOWN / UNMAPPED SCHEDULES", bold=True)
    if not sch["unknown"]:
        draw_line("(none)")
    else:
        for u in sch["unknown"][:20]:
            draw_line(f"- {u.get('schedule_type','UNKNOWN')} (page {u.get('source',{}).get('page','')})")
    c.save()


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    if len(sys.argv) < 2:
        die("Usage: python3 generate_tab.py /path/to/Blueprint.pdf")

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        die(f"File not found: {pdf_path}")

    master = empty_master()
    master["lineage"]["source_pdf"] = pdf_path
    master["lineage"]["generated_at"] = now_iso()

    doc = fitz.open(pdf_path)

    # Project info from first 1-2 pages (generic)
    first_text = ""
    for i in range(min(2, len(doc))):
        first_text += doc[i].get_text("text") + "\n"
    try:
        master["project"] = llm_extract_project_info(first_text)
    except Exception as e:
        # Non-fatal; keep going
        master["project"]["project_name"] = ""
        master["project"]["client"] = ""
        master["project"]["project_address"] = ""
        master["project"]["project_number"] = ""

    # Iterate pages: detect schedules and extract tables
    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1
        txt = page.get_text("text")
        top = "\n".join(txt.splitlines()[:40])
        sched_type = detect_schedule_type(top)

        if not sched_type:
            continue

        table = extract_table(page, sched_type)
        if not table:
            # Store fallback raw page text under unknown
            master["schedules"]["unknown"].append({
                "schedule_type": sched_type,
                "headers": [],
                "rows": [],
                "raw_text_excerpt": top[:1200],
                "source": {"page": page_num}
            })
            continue

        # Map to canonical using LLM
        try:
            mapped = llm_map_schedule_to_canonical(table, page_num=page_num)
        except Exception as e:
            # If LLM fails, store raw table
            master["schedules"]["unknown"].append({
                "schedule_type": sched_type,
                "headers": table.get("headers", []),
                "rows": table.get("rows", [])[:50],
                "source": {"page": page_num},
                "error": str(e)
            })
            continue

        # Merge into master
        master["schedules"]["fans"].extend(mapped["fans"])
        master["schedules"]["vavs"].extend(mapped["vavs"])
        master["schedules"]["equipment"].extend(mapped["equipment"])
        master["schedules"]["airflow"].extend(mapped["airflow"])
        master["schedules"]["diffusers"].extend(mapped["diffusers"])
        master["schedules"]["unknown"].extend(mapped["unknown"])

    # Write JSON
    with open("master_tab.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    # Render PDF
    render_tab_report_pdf(master, out_path="tab_report.pdf")

    print("Wrote master_tab.json")
    print("Wrote tab_report.pdf")


if __name__ == "__main__":
    main()
