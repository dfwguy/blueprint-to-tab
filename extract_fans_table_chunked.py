#!/usr/bin/env python3
import json
import fitz
import requests
from collections import defaultdict

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_first_json_object(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])

    raise ValueError("Unclosed JSON object")


def run_llm_json(prompt: str, timeout_s: int = 600) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            # give it enough room for a small chunk only
            "num_predict": 1200,
            "repeat_penalty": 1.1,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()

    raw = r.json().get("response", "")
    if not raw:
        raise ValueError("Empty Ollama response")

    try:
        return extract_first_json_object(raw)
    except Exception as e:
        with open("ollama_bad_json.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise ValueError(f"Bad JSON from model (wrote ollama_bad_json.txt): {e}") from e


def words_to_table_lines(page, y_tol=3.0):
    words = page.get_text("words")
    if not words:
        return []

    rows = defaultdict(list)
    for x0, y0, x1, y1, w, *_ in words:
        y_key = round(y0 / y_tol) * y_tol
        rows[y_key].append((x0, w))

    lines = []
    for y_key in sorted(rows.keys()):
        row = sorted(rows[y_key], key=lambda t: t[0])
        lines.append(" | ".join(w for _, w in row))

    return lines


def keep_only_fan_rows(lines):
    """
    Keep: header line 'FAN SCHEDULE' + column header-ish area + data rows starting with EF- / SF- / RF- etc.
    This reduces prompt size massively and avoids truncation.
    """
    kept = []
    for ln in lines:
        s = ln.strip()

        if "FAN SCHEDULE" in s:
            kept.append(s)
            continue

        # Keep obvious column-name lines
        if s in {
            "DESIGNATION", "LOCATION", "TYPE", "DRIVE", "AIRFLOW", "(CFM)",
            "ESP", "(IN", "WG)", "MOTOR", "POWER", "SPEED", "(RPM)",
            "ELECTRICAL", "DATA", "WEIGHT", "(LBS)", "CONTROL",
            "BASIS", "OF", "DESIGN", "NOTES", "FLA", "MCA", "MOCP",
            "(AMPS)", "V/PH/HZ", "FACTORY", "DISCONNECT"
        }:
            kept.append(s)
            continue

        # Keep data rows (designation usually like EF-1, EF-2, EF-3...)
        if s.startswith(("EF-", "SF-", "RF-", "AF-", "BF-")):
            kept.append(s)
            continue

    return kept


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[7]  # page 8

    lines = words_to_table_lines(page)
    lines = keep_only_fan_rows(lines)

    # Save filtered table text for visibility
    with open("fan_schedule_table_filtered.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    fans_all = []

    # Chunk to keep prompts small and prevent truncation
    for idx, chunk in enumerate(chunk_list(lines, 40), start=1):
        table_text = "\n".join(chunk)

        prompt = f"""
Extract FAN schedule rows from the table below.

Return STRICT JSON ONLY:
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
      "basis_of_design": "",
      "notes": ""
    }}
  ]
}}

Rules:
- Only parse actual fan records (designation like EF-1, EF-2, etc.).
- If you cannot find a value, use null or "".
- Numbers must be numbers (no quotes).
- Output must be valid JSON.

TABLE CHUNK:
{table_text}
""".strip()

        result = run_llm_json(prompt)
        fans = result.get("fans", [])
        fans_all.extend(fans)
        print(f"Chunk {idx}: extracted {len(fans)} fans")

    out = {"fans": fans_all}

    with open("fan_schedule.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved fan_schedule.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
