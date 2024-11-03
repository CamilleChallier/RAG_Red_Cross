"""
This module provides functions for text processing, topic modeling using LDA, and chunking text based on coherence scores.
It includes steps such as sentence tokenization, topic modeling, and segmenting text into coherent chunks.

Functions:
- create_sentences(data): Splits text into sentences, tokenizes them using SpaCy, and lemmatizes the tokens.
- window(seq, n=3): Generates a sliding window of size `n` over a sequence.
- LDA(tokenized_sents, N_TOPICS=5, N_PASSES=5): Performs Latent Dirichlet Allocation (LDA) on tokenized sentences to extract topics and compute coherence scores between adjacent sentence windows.
- chunks_from_coherence(sentences, coherence, breakpoint_threshold=0.5, min_size=100): Segments the input text into coherent chunks based on the coherence scores of topic distributions.
"""

import re
import spacy
from gensim import corpora, models
from itertools import islice    
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append('../')
from utils import read_file
from datasets import load_dataset
from llama_index.core import Document
from tqdm import tqdm

def create_sentences(data):
    """
    Splits text into sentences and tokenizes them using SpaCy. The sentences are lemmatized, 
    lowercased, and filtered for stop words and punctuation.
    
    Args:
    - data (list of str): The input text data to be processed.

    Returns:
    - tuple: A tuple containing the split sentences and the corresponding tokenized and lemmatized sentences.
    """
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    data_split = [sentence for d in data for sentence in re.split(r'[.?!]\n+', d) if (sentence != "" and len(sentence) >2) ]

    nlp = spacy.load('en_core_web_sm')
    
    sents = []
    for text in data_split:
        doc = nlp(str(text))
        sents.append(doc)
        
    MIN_LENGTH = 3
    tokenized_sents = [[token.lemma_.lower() for token in sent 
                    if not token.is_stop and 
                    not token.is_punct and token.text.strip() and 
                    len(token) >= MIN_LENGTH] 
                  for sent in sents]
    return data_split, tokenized_sents

def window(seq, n=3):
    """
    Generates a sliding window of size `n` over a sequence.
    
    Args:
    - seq (iterable): The sequence to generate a window over.
    - n (int): The size of the sliding window.

    Yields:
    - tuple: The next window of size `n` from the sequence.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n: 
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def LDA(tokenized_sents, N_TOPICS = 5, N_PASSES = 5):
    """
    Performs Latent Dirichlet Allocation (LDA) to identify topics within the tokenized sentences and computes 
    coherence scores between adjacent windows of topics.

    Args:
    - tokenized_sents (list of list of str): The tokenized and lemmatized sentences.
    - N_TOPICS (int): The number of topics to extract, default is 5.
    - N_PASSES (int): The number of passes over the corpus during training, default is 5.

    Returns:
    - list of float: The coherence scores between adjacent windows of topics.
    """
    dictionary = corpora.Dictionary(tokenized_sents)
    bow = [dictionary.doc2bow(sent) for sent in tokenized_sents]
    topic_model = models.LdaModel(corpus=bow, id2word=dictionary, 
                              num_topics=N_TOPICS, passes=N_PASSES)
    doc_topics = list(topic_model.get_document_topics(bow, minimum_probability=0.05))
    
    k = 3
    top_k_topics = [[t[0] for t in sorted(sent_topics, key=lambda x: x[1], reverse=True)][:k] 
                for sent_topics in doc_topics]

    window_topics = window(top_k_topics, n=3)
    window_topics = [list(set(chain.from_iterable(window))) 
                for window in window_topics] + [top_k_topics[-2]] + [top_k_topics[-1]]
    
    binarizer = MultiLabelBinarizer(classes=range(N_TOPICS))
    encoded_topic = binarizer.fit_transform(window_topics)

    coherence_scores = [cosine_similarity([pair[0]], [pair[1]])[0][0] 
                for pair in zip(encoded_topic[:-1], encoded_topic[1:])]
    return coherence_scores

def chunks_from_coherence(sentences, coherence, breakpoint_threshold = 0.5, min_size = 100):
    """
    Segments the input text into coherent chunks based on coherence scores between topic windows.
    
    Args:
    - sentences (list of str): The original sentences to be chunked.
    - coherence (list of float): The coherence scores for each window of sentences.
    - breakpoint_threshold (float): The threshold for coherence below which text should be split, default is 0.5.
    - min_size (int): The minimum size of each chunk in characters, default is 100.

    Returns:
    - list of str: The segmented chunks of text.
    """
    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_under_thresh = [i for i, x in enumerate(coherence) if x < breakpoint_threshold] # The indices of those breakpoints on your list
    
    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_under_thresh:
    
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = '. '.join([d for d in group]) + ". "
        
        if len(combined_text)<min_size :
            if chunks ==[] :
                continue
            if coherence[start_index-1]>coherence[end_index]:
                chunks[-1] = chunks[-1] + combined_text
                start_index = index + 1
        else :         
            chunks.append(combined_text)
            # Update the start index for the next group
            start_index = index + 1
        

    # The last group, if any sentences remain
    if start_index < len(sentences):
        group = sentences[start_index:]
        combined_text = '. '.join([d for d in group])
        
        if len(combined_text)<min_size and chunks != []:
            chunks[-1] = chunks[-1] + combined_text
        else : 
            chunks.append(combined_text)

    return chunks

if __name__ == "__main__":
    docs = []
    
    # load data
    def load_wiki_data():
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        data = read_file(ds["train"]['text'])
        return data
    data = load_wiki_data()
    data = [d.replace("= \n", " : ") for d in data] 
    
    docs = []
    from LDA_chunking import create_sentences, LDA, chunks_from_coherence
    for d in tqdm(data):
        if len(d) < 1024 : #if the paragraph is already smaller than chunk size : stay like this
            docs.append(Document(text = d))
        else : 
            data_split, tokenized_sents = create_sentences([d])
            if len(data_split) > 2 : # if there is many sentences
                coherence_scores = LDA(tokenized_sents)
                chunks = chunks_from_coherence(data_split, coherence_scores, breakpoint_threshold=0.5)
                docs.extend([Document(text = chunk) for chunk in chunks])
            else : 
                docs.append(Document(text = d))

