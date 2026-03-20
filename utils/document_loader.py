import pandas as pd
import os
from pypdf import PdfReader
from typing import List, Dict

# Get absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_documents() -> List[Dict[str, str]]:
    """
    Loads academic documents from Excel, text files, and PDFs into memory.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing 'text' and 'source' keys.
    """
    try:
        docs = []

        # Structured Excel syllabus ingestion
        excel_path = os.path.join(BASE_DIR, "kb", "msc_ds_structure.xlsx")

        if os.path.exists(excel_path):

            xls = pd.ExcelFile(excel_path)

            for sheet in xls.sheet_names:

                df = pd.read_excel(xls, sheet_name=sheet)

                rows_text = []

                for _, row in df.iterrows():

                    text = (
                        f"{sheet} — {row['Course Code']} {row['Subject Name']} — "
                        f"Credits {row['Credits']} — "
                        f"CIA {row['CIA Marks']} — "
                        f"ESE {row['ESE Marks']} — "
                        f"{row['Type']} course."
                    )

                    rows_text.append(text)

                docs.append({
                    "text": "\n".join(rows_text),
                    "source": sheet
                })

        # Optional secondary fallback: curated course detail txt
        course_folder = os.path.join(BASE_DIR, "kb", "course_details")
        if os.path.exists(course_folder):

            for f in os.listdir(course_folder):
                if f.endswith(".txt"):
                    path = os.path.join(course_folder, f)

                    with open(path, "r", encoding="utf-8") as file:
                        text = file.read()

                    docs.append({
                        "text": text,
                        "source": f.replace(".txt", "")
                    })

        # Optional full PDF ingestion as long-tail fallback
        pdf_path = os.path.join(BASE_DIR, "data", "mds_syllabus.pdf")
        if os.path.exists(pdf_path):

            reader = PdfReader(pdf_path)
            full_text = ""

            for p in reader.pages:
                t = p.extract_text()
                if t:
                    full_text += "\n" + t

            docs.append({
                "text": full_text,
                "source": "full_syllabus_pdf"
            })

        return docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []