#!/usr/bin/env python3
import json
import fitz
import requests
from collections import defaultdict

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Column names we expect for a fan schedule (flexible)
CANON_COLS = [
    "designation", "location", "type", "drive",
    "airflow_cfm", "esp_in_wg", "motor_power_w", "speed_rpm",
    "electrical_data", "weight_lbs", "control", "basis_of_design", "notes"
]

def run_llm_json(prompt: str, timeout_s: int = 600) -> dict:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1200},
        },
        timeout=timeout_s,
    )
    r.raise_for_status()
    return json.loads(r.json()["response"])


def nearest_col(x, col_xs):
    # col_xs is list of (col_name, x_center)
    best = None
    best_d = 1e9
    for name, cx in col_xs:
        d = abs(x - cx)
        if d < best_d:
            best = name
            best_d = d
    return best


def build_column_map_from_headers(words):
    """
    Find column x-centers by looking for header tokens and taking their x positions.
    This is heuristic but works well for schedules.
    """
    header_tokens = {
        "DESIGNATION": "designation",
        "LOCATION": "location",
        "TYPE": "type",
        "DRIVE": "drive",
        "AIRFLOW": "airflow_cfm",
        "(CFM)": "airflow_cfm",
        "ESP": "esp_in_wg",
        "(IN": "esp_in_wg",
        "WG)": "esp_in_wg",
        "MOTOR": "motor_power_w",
        "POWER": "motor_power_w",
        "SPEED": "speed_rpm",
        "(RPM)": "speed_rpm",
        "ELECTRICAL": "electrical_data",
        "DATA": "electrical_data",
        "WEIGHT": "weight_lbs",
        "(LBS)": "weight_lbs",
        "CONTROL": "control",
        "BASIS": "basis_of_design",
        "OF": "basis_of_design",
        "DESIGN": "basis_of_design",
        "NOTES": "notes",
    }

    hits = defaultdict(list)
    for x0, y0, x1, y1, w, *_ in words:
        w2 = w.strip().upper()
        if w2 in header_tokens:
            col = header_tokens[w2]
            hits[col].append((x0 + x1) / 2)

    # Reduce to one x-center per canonical column
    col_xs = []
    for col in CANON_COLS:
        if col in hits:
            xs = sorted(hits[col])
            # median is robust
            cx = xs[len(xs)//2]
            col_xs.append((col, cx))

    # If we didn’t detect enough columns, fall back to a simple evenly-spaced map
    # (still better than nothing)
    col_xs = sorted(col_xs, key=lambda t: t[1])
    return col_xs


def extract_rows(words, col_xs, y_tol=4.0):
    """
    Assign each word to nearest column, group by row (y), and return row dicts.
    """
    rows = defaultdict(lambda: defaultdict(list))

    for x0, y0, x1, y1, w, *_ in words:
        text = w.strip()
        if not text:
            continue

        # Skip big title
        if text.upper() == "FAN" or text.upper() == "SCHEDULE":
            continue

        y_key = round(y0 / y_tol) * y_tol

        # assign to nearest detected column
        x_center = (x0 + x1) / 2
        if col_xs:
            col = nearest_col(x_center, col_xs)
        else:
            col = "notes"

        rows[y_key][col].append((x0, text))

    # Build row dicts in top-to-bottom order, joining tokens left-to-right per column
    out = []
    for y_key in sorted(rows.keys()):
        row = {}
        for col, toks in rows[y_key].items():
            toks_sorted = [t for _, t in sorted(toks, key=lambda t: t[0])]
            row[col] = " ".join(toks_sorted).strip()

        # Keep only rows that look like fan records (designation like EF-1 / EF-2 etc.)
        des = row.get("designation", "")
        if des.startswith(("EF-", "SF-", "RF-", "AF-", "BF-")):
            # Ensure all canonical keys exist
            final = {k: row.get(k, "") for k in CANON_COLS}
            out.append(final)

    return out


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[7]  # fan schedule page

    words = page.get_text("words")
    col_xs = build_column_map_from_headers(words)

    # Save detected column positions for debugging
    with open("fan_cols_detected.json", "w", encoding="utf-8") as f:
        json.dump(col_xs, f, indent=2)

    rows = extract_rows(words, col_xs)

    # Save reconstructed rows for visibility
    with open("fan_rows_reconstructed.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Now ask LLM to normalize / coerce numeric fields
    prompt = f"""
Normalize these fan schedule rows into strict JSON with correct types.

Input rows are dictionaries keyed by:
{CANON_COLS}

Return STRICT JSON:
{{
  "fans": [
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
      "notes": ""
    }}
  ]
}}

Rules:
- Convert numeric-like strings to numbers (airflow_cfm, esp_in_wg, motor_power_w, speed_rpm, weight_lbs).
- If value is missing, use null for numbers and "" for strings.
- Do not invent values.

Rows:
{json.dumps(rows, indent=2)}
""".strip()

    result = run_llm_json(prompt)

    with open("fan_schedule.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nSaved fan_schedule.json")
    print("Also wrote fan_cols_detected.json and fan_rows_reconstructed.json for debugging.")


if __name__ == "__main__":
    main()
