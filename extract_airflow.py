#!/usr/bin/env python3
import json
import fitz
import requests

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def run_llm(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 800
            }
        },
        timeout=600
    )
    return json.loads(response.json()["response"])


# Load classified pages
with open("page_types.json", "r") as f:
    page_types = json.load(f)

airflow_pages = [p["page"] for p in page_types if p["type"] == "AIRFLOW_SCHEDULE"]

doc = fitz.open(PDF_PATH)

all_airflow_data = []

for page_num in airflow_pages:
    text = doc[page_num - 1].get_text("text")

    prompt = f"""
Extract airflow schedule data from this page.

Return STRICT JSON:

{{
  "airflow_units": [
    {{
      "tag": "",
      "room": "",
      "cfm_supply_design": null,
      "cfm_return_design": null,
      "cfm_exhaust_design": null,
      "notes": ""
    }}
  ]
}}

Page text:
{text}
"""

    result = run_llm(prompt)

    if "airflow_units" in result:
        all_airflow_data.extend(result["airflow_units"])

with open("airflow_data.json", "w") as f:
    json.dump(all_airflow_data, f, indent=2)

print("Saved airflow data to airflow_data.json")
print(json.dumps(all_airflow_data, indent=2))
