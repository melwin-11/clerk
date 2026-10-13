from sentence_transformers import SentenceTransformer
embedding_model=SentenceTransformer(model_name_or_path='sentence-transformers/all-mpnet-base-v2',
                                    device='cpu',token=False)

sentences=['The sentence transformer library provides an easy way to create embeddings',
           'I like horses',
           'Senteces can be embedded one by one or in a list']

embeddings=embedding_model.encode(sentences)
embeddings_dict=dict(zip(sentences,embeddings))

for sentence,embedding in embeddings_dict.items():
    print(f'Sentence: {sentence}')
    print(f'Embedding: {embedding}')
    print("")