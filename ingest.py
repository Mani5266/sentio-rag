from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
import requests
from bs4 import BeautifulSoup
import os

def extract_text(file_path=None, url=None):

    elements = []

    if file_path:

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            elements = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res"
            )

        elif ext == ".docx":
            elements = partition_docx(filename=file_path)

        elif ext == ".txt":
            elements = partition_text(filename=file_path)

    elif url:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")

    # Join structured blocks
    return "\n".join([e.text for e in elements if e.text])