import os
import requests
from config import pdf_path

def check():
    if not os.path.exists(pdf_path):
        print(f'[INFO] File does not exit. Import File (Kindly use only .pdfs).\n')
    else:
        print(f'[INFO] File {pdf_path} exists.\n')

if __name__=="__main__":
    check()