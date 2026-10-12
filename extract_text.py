import fitz
from tqdm.auto import tqdm
from pdf_exist import pdf_path
import random
from config import pdf_path

def text_formatter(text:str)->str:
    cleaned_text=text.replace('\n'," ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path:str)->list[dict]:
    doc=fitz.open(pdf_path)
    pages_and_texts=[]
    for page_number,page in tqdm(enumerate(doc)):
        text = page.get_text()
        text=text_formatter(text=text)
        pages_and_texts.append({"page_number":page_number,
                                "page_char_count":len(text),
                                "page_word_count":len(text.split(" ")),
                                "page_sentence_count_raw":len(text.split(". ")),
                                "page_token_count":len(text)/4,
                                "text":text}) #Because 1 token is roughly 4 characters
    return pages_and_texts

if __name__=='__main__':
    open_and_read_pdf(pdf_path=pdf_path)