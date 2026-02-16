#!/usr/bin/env python3
import json
import re
import fitz
import requests
from collections import defaultdict

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

FAN_ID_RE = re.compile(r"\b([A-Z]{1,3}-\d+)\b")  # EF-1, SF-2, RF-10 etc.


def extract_first_json_object(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError("Unclosed JSON object")


def run_llm_json(prompt: str, timeout_s: int = 600) -> dict:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1400},
        },
        timeout=timeout_s,
    )
    r.raise_for_status()
    raw = r.json().get("response", "")
    try:
        return extract_first_json_object(raw)
    except Exception as e:
        with open("ollama_bad_json.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise


def words_grouped_rows(page, y_tol=8.0):
    """
    Group words into rows by y coordinate. Larger y_tol handles PDFs where
    row baselines vary slightly.
    Returns list of rows, each row is list of (x0, word).
    """
    words = page.get_text("words")
    rows = defaultdict(list)

    for x0, y0, x1, y1, w, *_ in words:
        w = w.strip()
        if not w:
            continue
        y_key = round(y0 / y_tol) * y_tol
        rows[y_key].append((x0, w))

    out = []
    for y_key in sorted(rows.keys()):
        out.append(sorted(rows[y_key], key=lambda t: t[0]))
    return out


def row_to_text(row):
    # Keep x-order but separate with " | " to preserve column feel
    return " | ".join(w for _, w in row)


def extract_fan_row_blocks(rows):
    """
    Build 'blocks' starting at a row containing EF-#/SF-#/etc,
    and include subsequent rows until the next fan id row.
    """
    blocks = []
    current = None

    for row in rows:
        line = row_to_text(row)
        m = FAN_ID_RE.search(line)

        if m:
            # Start new block
            if current:
                blocks.append(current)
            current = {"fan_id": m.group(1), "lines": [line]}
        else:
            if current:
                current["lines"].append(line)

    if current:
        blocks.append(current)

    # Keep only EF/SF/RF/AF/BF style fans
    blocks = [b for b in blocks if b["fan_id"].startswith(("EF-", "SF-", "RF-", "AF-", "BF-"))]
    return blocks


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[7]  # fan schedule page

    rows = words_grouped_rows(page, y_tol=8.0)
    blocks = extract_fan_row_blocks(rows)

    with open("fan_blocks_debug.json", "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)

    # Prepare compact input for LLM: each fan block lines joined
    fan_inputs = []
    for b in blocks:
        fan_inputs.append({
            "fan_id": b["fan_id"],
            "text": "\n".join(b["lines"][:12])  # cap lines to avoid huge prompts
        })

    # Chunk prompts to avoid truncation
    all_fans = []
    for idx, chunk in enumerate(chunk_list(fan_inputs, 5), start=1):
        prompt = f"""
You are extracting FAN SCHEDULE records from mechanical drawings.

Each item contains a fan_id and text lines (with " | " as column separators).
Parse each item into a structured record.

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
- designation MUST equal the given fan_id.
- Do not invent values. If missing, use null for numbers and "" for strings.
- airflow_cfm/esp_in_wg/motor_power_w/speed_rpm/weight_lbs must be numbers (no quotes).
- The schedule header may be present; ignore it.

Input items:
{json.dumps(chunk, indent=2)}
""".strip()

        result = run_llm_json(prompt)
        all_fans.extend(result.get("fans", []))
        print(f"Chunk {idx}: extracted {len(result.get('fans', []))} fans")

    out = {"fans": all_fans}
    with open("fan_schedule.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved fan_schedule.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
