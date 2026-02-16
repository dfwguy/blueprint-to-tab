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
                "num_predict": 1000
            }
        },
        timeout=600
    )
    return json.loads(response.json()["response"])


doc = fitz.open(PDF_PATH)

# Page 8 is fan schedule (0-based index = 7)
text = doc[7].get_text("text")

prompt = f"""
Extract FAN schedule data from this page.

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
      "motor_power": "",
      "speed_rpm": null,
      "electrical_data": "",
      "weight_lbs": null,
      "basis_of_design": "",
      "notes": ""
    }}
  ]
}}

Page text:
{text}
"""

result = run_llm(prompt)

with open("fan_schedule.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print("Saved fan_schedule.json")
