import fitz

doc = fitz.open("/home/dfwharish/Downloads/Boeing R&D Drawings.pdf")

for i in range(min(3, len(doc))):
    print(f"\n--- PAGE {i+1} ---")
    print(doc[i].get_text("text")[:2000])
