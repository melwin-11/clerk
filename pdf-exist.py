import os
import requests

pdf_path='data/monkeybusiness-report.pdf'

if not os.path.exists(pdf_path):
    print(f'[INFO] File does not exit. Import File (Kindly use only .pdfs).')
else:
    print(f'[INFO] File {pdf_path} exists.')