import fitz
import requests
import json

PDF_PATH = "/home/dfwharish/Downloads/Boeing R&D Drawings.pdf"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)

def run_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=600
    )
    return response.json()["response"]

text = extract_text(PDF_PATH)

prompt = f"""
You are extracting structured project metadata from engineering R&D drawings.

Return STRICT JSON only. No commentary.

Schema:
{{
  "project_name": "",
  "client": "",
  "project_address": "",
  "architect": "",
  "mep_engineer": "",
  "structural_engineer": "",
  "issue_type": "",
  "issue_date": "",
  "project_number": ""
}}

Text:
{text[:15000]}
"""

output = run_llm(prompt)

print(output)
