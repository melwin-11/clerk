from extract_text import open_and_read_pdf
from config import pdf_path
import pandas as pd

def stats():
    pages_and_texts=open_and_read_pdf(pdf_path=pdf_path)
    df=pd.DataFrame(pages_and_texts)
    print(df.describe().round(2))

if __name__=='__main__':
    stats()