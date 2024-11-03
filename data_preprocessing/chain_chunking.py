"""
This module provides functions for segmenting and processing text data by leveraging language model embeddings. 
It primarily focuses on chunking text based on cosine similarity between sentence embeddings.

Functions:
- chunk_text_chain(data, hf, thresh=0.3, split=True): Splits text into sentences, calculates cosine distances 
  between their embeddings, and chunks them based on a similarity threshold.
- split_sentences_chain(data, split): Splits the text into individual sentences using regular expressions.
- calculate_cosine_distances_chain(embedding_current, embedding_next): Computes the cosine distance between two sentence embeddings.
"""

import re
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import Document
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import read_file
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def chunk_text_chain(data, hf, thresh = 0.3, split=True):
    """
    Segments text into chunks based on cosine similarity between sentence embeddings.
    
    Args:
    - data (list of str): The input text data to be chunked.
    - hf (HuggingFaceBgeEmbeddings): Hugging Face embeddings model for generating sentence embeddings.
    - thresh (float): Cosine similarity threshold for determining where to split chunks, default is 0.3.
    - split (bool): Whether to split the text into sentences, default is True.

    Returns:
    - tuple: A tuple containing a list of sentences with their embeddings and distances, and a list of text chunks.
    """
    # Split the input text into individual sentences.
    sentences = split_sentences_chain(data, split)
    if len(sentences) < 2 :
        return None, data
    
    combined_sentence = ""
    chunks = []
    sentences[0]['sentence_embedding'] = hf.embed_documents([sentences[0]['sentence']])
    
    for i in range(len(sentences)-1):

        # Create a string that will hold the sentences which are joined
        combined_sentence += sentences[i]['sentence'] + '. '
        sentences[i]['combined_sentence'] = combined_sentence

        sentences[i]['combined_sentence_embedding'] = hf.embed_documents([sentences[i]['combined_sentence']])[0]
        if i < (len(sentences)-2) : 
            sentences[i+1]['sentence_embedding'] = hf.embed_documents([sentences[i+1]['sentence'] + '. ' + sentences[i+2]['sentence']])[0] 
        else : 
            sentences[i+1]['sentence_embedding'] = hf.embed_documents([sentences[i+1]['sentence']])[0] 
        
        distance = calculate_cosine_distances_chain([sentences[i]['combined_sentence_embedding']], [sentences[i + 1]['sentence_embedding']])
    
        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance
        
        if distance > thresh : 
            
            if len(combined_sentence)<100 :
                if chunks ==[] :
                    continue
                if sentences[i-1]['distance_to_next']<distance :
                    chunks[-1] = chunks[-1] + combined_sentence
                    combined_sentence = ""
            else :
                chunks.append(combined_sentence)
                combined_sentence = ""
    # The last group, if any sentences remain
    last = combined_sentence + sentences[i+1]['sentence'] + '. ' 
    if len(last)<100 :
        chunks[-1] = chunks[-1] + last
    else : 
        chunks.append(last)
            
    return sentences, chunks

def split_sentences_chain(data, split):
    """
    Splits the input text into individual sentences using regular expressions.
    
    Args:
    - data (list of str): The input text data to be split into sentences.
    - split (bool): Whether to split the text into sentences based on punctuation, default is True.

    Returns:
    - list of dict: A list of dictionaries, each containing a sentence and its index in the original text.
    """
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    if split : 
        data = [sentence for d in data for sentence in re.split(r'[.?!]\n+', d) if (sentence != "" and len(sentence) >2)]
    
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(data)]
    return sentences

def calculate_cosine_distances_chain(embedding_current, embedding_next) : 
    """
    Calculates the cosine distance between two sentence embeddings.
    
    Args:
    - embedding_current (list of np.array): The embedding of the current sentence or chunk of sentences.
    - embedding_next (list of np.array): The embedding of the next sentence or chunk of sentences.

    Returns:
    - float: The cosine distance between the two embeddings.
    """
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding_current, embedding_next)[0][0]
    
    # Convert to cosine distance
    distance = 1 - similarity

    return distance

if __name__ == "__main__":
    docs = []
    
    # load data
    def load_wiki_data():
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        data = read_file(ds["train"]['text'])
        return data
    data = load_wiki_data()
    data = [d.replace("= \n", " : ") for d in data] 
    
    # load embedding model
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name="w601sxs/b1ade-embed", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    for d in tqdm(data): #for each articles
        paragraphs = d.split("\n =")
        for para in paragraphs : #for each paragraph
            if len(para)< 1024 :
                docs.append(Document(text = para))
            else : 
                _,chunks = chunk_text_chain([para], hf=hf, thresh = 0.4)
                docs.extend([Document(text = chunk) for chunk in chunks])