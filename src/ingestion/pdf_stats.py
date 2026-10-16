import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.config import pdf_path
    from src.ingestion.extract_text import open_and_read_pdf
except ImportError:
    from config import pdf_path
    from extract_text import open_and_read_pdf


def stats():
    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    df = pd.DataFrame(pages_and_texts)
    print(df.describe().round(2))


if __name__ == "__main__":
    stats()