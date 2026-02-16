#!/usr/bin/env python3
import json
import re
import fitz  # pymupdf
import requests

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _extract_first_json_object(s: str) -> dict:
    """
    Safety net: if the model returns extra text, extract the first {...} JSON object.
    """
    # Fast path: try direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: find first JSON object using a simple brace match
    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output:\n{s}")

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                return json.loads(candidate)

    raise ValueError(f"Unclosed JSON object in model output:\n{s}")


def run_llm(prompt: str, *, timeout_s: int = 600) -> dict:
    """
    Calls Ollama /api/generate and returns parsed JSON dict.
    Uses Ollama's `format: json` to strongly enforce JSON-only responses.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",  # <-- IMPORTANT: forces JSON output
        "stream": False,
        "options": {
            "temperature": 0,
            # Optional knobs; safe defaults:
            "num_predict": 256
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()

    # Ollama returns: {"response": "...", ...}
    raw = r.json().get("response", "")
    if not raw:
        raise ValueError(f"Empty response from Ollama. Full payload:\n{r.text}")

    return _extract_first_json_object(raw)


def classify_page(page_num: int, text: str) -> dict:
    text = (text or "").strip()
    prompt = f"""
Classify this drawing page.

Return ONLY this JSON structure:
{{"page": {page_num}, "type": ""}}

Valid types (choose one):
COVER
GENERAL_NOTES
EQUIPMENT_SCHEDULE
VAV_SCHEDULE
AIRFLOW_SCHEDULE
FLOOR_PLAN
DETAIL
OTHER

Page text:
{text[:4000]}
""".strip()

    return run_llm(prompt)


def main():
    doc = fitz.open(PDF_PATH)
    results = []

    for i in range(len(doc)):
        page_num = i + 1
        text = doc[i].get_text("text")
        out = classify_page(page_num, text)

        # Normalize/guard
        out_page = out.get("page", page_num)
        out_type = out.get("type", "OTHER")

        results.append({"page": out_page, "type": out_type})
        print(json.dumps(results[-1], indent=2))

    # Save to file for later extraction passes
    with open("page_types.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved page classification to: page_types.json")


if __name__ == "__main__":
    main()
