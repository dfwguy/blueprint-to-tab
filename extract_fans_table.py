#!/usr/bin/env python3
import json
import fitz
import requests
from collections import defaultdict

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_first_json_object(s: str) -> dict:
    """
    Extract the first top-level JSON object { ... } from a string.
    Handles cases where the model prints extra text before/after JSON.
    """
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found; no JSON object present.")

    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Unclosed JSON object (never reached depth 0).")


def run_llm_json(prompt: str, timeout_s: int = 600) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",     # strongest hint for JSON-only
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1400,
            "repeat_penalty": 1.1,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()

    raw = r.json().get("response", "")
    if not raw:
        # Save full response body for debugging
        with open("ollama_empty_response.txt", "w", encoding="utf-8") as f:
            f.write(r.text)
        raise ValueError("Ollama returned an empty 'response'. Wrote ollama_empty_response.txt")

    # Try to parse robustly
    try:
        return extract_first_json_object(raw)
    except Exception as e:
        # Save the raw model output so you can see what's wrong
        with open("ollama_bad_json.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise ValueError(f"Failed to parse JSON. Wrote ollama_bad_json.txt. Error: {e}") from e


def words_to_table_lines(page, y_tol=3.0):
    """
    Reconstruct table-ish lines from word positions.
    Produces lines like: col1 | col2 | col3 ...
    """
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


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[7]  # Page 8 (0-based) = FAN SCHEDULE

    table_lines = words_to_table_lines(page)
    table_text = "\n".join(table_lines)

    # Optional: also dump the reconstructed table for troubleshooting
    with open("fan_schedule_table_dump.txt", "w", encoding="utf-8") as f:
        f.write(table_text)

    prompt = f"""
You are extracting a FAN SCHEDULE from a mechanical drawing.

Input is a reconstructed table. Columns are separated by " | ".

Return STRICT JSON ONLY in this shape:

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
- Ignore header lines (FAN SCHEDULE / column names).
- airflow_cfm, esp_in_wg, motor_power_w, speed_rpm, weight_lbs must be numbers or null (NO quotes).
- If you see YES/NO columns, map them to "control" or "notes" (NOT numeric fields).
- Do not invent values.

TABLE:
{table_text}
""".strip()

    result = run_llm_json(prompt)

    with open("fan_schedule.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nSaved fan_schedule.json")
    print("Also saved reconstructed table to fan_schedule_table_dump.txt")


if __name__ == "__main__":
    main()
