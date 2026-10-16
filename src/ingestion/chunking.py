import random
from spacy.lang.en import English
from tqdm.auto import tqdm
import re
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.config import pdf_path, sentence_chunk_size
    from src.ingestion.extract_text import open_and_read_pdf
except ImportError:
    from config import pdf_path, sentence_chunk_size
    from extract_text import open_and_read_pdf

nlp = English()
nlp.add_pipe("sentencizer")

def split_list(input_list: list[str], slice_size: int = sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

def split_sentences(pages_and_text):
    pages_and_chunks = []  
    
    print("Tokenizing sentences with spaCy...")
    for item in tqdm(pages_and_text):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

    print("Splitting sentences into chunks...")
    for item in tqdm(pages_and_text):
        item['sentence_chunks'] = split_list(input_list=item['sentences'], slice_size=sentence_chunk_size)
        item['num_chunks'] = len(item['sentence_chunks'])

    print("Structuring chunks...")
    for item in tqdm(pages_and_text):
        for sentence_chunk in item['sentence_chunks']:
            chunk_dict = {}
            chunk_dict['page_number'] = item['page_number']

            joined_sentence_chunk = ''.join(sentence_chunk).replace(" ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict['sentence_chunk'] = joined_sentence_chunk

            chunk_dict['chunk_char_count'] = len(joined_sentence_chunk)
            chunk_dict['chunk_word_count'] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict['chunk_token_count'] = len(joined_sentence_chunk) / 4

            pages_and_chunks.append(chunk_dict)
            
    df = pd.DataFrame(pages_and_chunks)
    filtered_df = df[df['chunk_token_count'] < 71]
    if not filtered_df.empty:
        sample_size = min(5, len(filtered_df))
        for index, row in filtered_df.sample(sample_size).iterrows():
            print(f'Chunk_token count: {row["chunk_token_count"]} | Text: {row["sentence_chunk"]}')
            
    return pages_and_chunks

if __name__ == "__main__":
    pages_and_text = open_and_read_pdf(pdf_path=pdf_path)
    split_sentences(pages_and_text)
