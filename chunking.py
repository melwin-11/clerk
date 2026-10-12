import random
from spacy.lang.en import English
from tqdm.auto import tqdm
import re
import pandas as pd

from config import pdf_path,sentence_chunk_size
from extract_text import open_and_read_pdf

nlp = English()
nlp.add_pipe("sentencizer")

pages_and_text = open_and_read_pdf(pdf_path=pdf_path)
pages_and_chunks=[]

for item in tqdm(pages_and_text):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

def split_list(input_list:list[str],slice_size:list[int]=sentence_chunk_size)->list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0,len(input_list),slice_size)]

def split_sentences(pages_and_text):
    for item in tqdm(pages_and_text):
        item['sentence_chunks']=split_list(input_list=item['sentences'],slice_size=sentence_chunk_size)
        item['num_chunks']=len(item['sentence_chunks'])

    for item in tqdm(pages_and_text):
        for sentence_chunk in item['sentence_chunks']:
            chunk_dict={}
            chunk_dict['page_number']=item['page_number']

            joined_sentence_chunk=''.join(sentence_chunk).replace(" "," ").strip()
            joined_sentence_chunk=re.sub(r'\.([A-Z])',r'. \1',joined_sentence_chunk)
            chunk_dict['sentence_chunk']=joined_sentence_chunk

            chunk_dict['chunk_char_count']=len(joined_sentence_chunk)
            chunk_dict['chunk_word_count']=len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict['chunk_token_count']=len(joined_sentence_chunk)/4

            pages_and_chunks.append(chunk_dict)
    df=pd.DataFrame(pages_and_chunks)
    filtered_df = df[df['chunk_token_count'] < 71]
    sample_size = min(5, len(filtered_df))
    for index, row in filtered_df.sample(sample_size).iterrows():
        print(f'Chunk_token count: {row["chunk_token_count"]} | Text: {row["sentence_chunk"]}')

if __name__ == "__main__":
    split_sentences(pages_and_text)