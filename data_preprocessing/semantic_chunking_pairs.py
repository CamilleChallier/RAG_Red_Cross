"""
Text Chunking with Embedding-Based Cosine Similarity

This script provides a method for splitting text into coherent chunks using sentence embeddings 
and cosine similarity. The primary objective is to identify semantic shifts in the text and split 
it into meaningful sections based on those shifts. The approach involves the following steps:

1. Splitting the text into individual sentences.
2. Creating combined pairs sentence contexts for each sentence (the first sentence is fused with its previous sentence and the 2nd sentence by the next one).
3. Generating embeddings for these combined contexts.
4. Calculating cosine distances between consecutive sentence embeddings.
5. Identifying breakpoints where cosine distances exceed a threshold.
6. Splitting the text into chunks based on these breakpoints.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from llama_index.core import Document
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import read_file
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def chunk_text_pairs(data, hf, buffer_size=1, breakpoint_percentile_threshold=95, split=True, pattern = r'[.?!]\n+'):
    """
    Splits the input text into chunks based on cosine similarity between consecutive sentence embeddings.

    Parameters:
    - data (list): List of text data to be chunked.
    - hf: Pretrained model used for generating sentence embeddings.
    - buffer_size (int): Number of sentences to include before and after each sentence for context.
    - breakpoint_percentile_threshold (int): Percentile threshold for determining chunk breakpoints.
    - split (bool): Whether to split the text into sentences.

    Returns:
    - distances (list): Cosine distances between consecutive combined sentence embeddings.
    - sentences (list): List of sentence dictionaries with combined embeddings.
    - chunks (list): List of text chunks.
    - max_index_split (list): Indices where the text was split based on cosine distance.
    """
    # Split the input text into individual sentences.
    sentences = split_sentences(data, split, pattern)
    if len(sentences) < 2 :
        return None, None, data, None
    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = combine_sentences(sentences, buffer_size=buffer_size)
    
    embedded_sentences = compute_embeddings(combined_sentences, hf)
    
    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances, sentences = calculate_cosine_distances(embedded_sentences)
    
    # Determine the breakpoints based on the cosine distances.
    chunks, max_index_split = chunks_from_distance(sentences, distances, breakpoint_percentile_threshold=breakpoint_percentile_threshold)
    return distances, sentences, chunks, max_index_split

def split_sentences(data, split, pattern):
    """
    Splits the input text into individual sentences.

    Parameters:
    - data (list): List of text data to be split into sentences.
    - split (bool): Whether to split the text into sentences.

    Returns:
    - sentences (list): List of dictionaries, each containing a sentence and its index.
    """
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    if split : 
        data = [sentence for d in data for sentence in re.split(pattern, d) if (sentence != "" and len(sentence) >2)]
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(data)]
    return sentences

def combine_sentences(sentences, buffer_size):
    """
    Combines adjacent sentences to form a context window around each sentence.

    Parameters:
    - sentences (list): List of dictionaries, each containing a sentence and its index.
    - buffer_size (int): Number of sentences to include before and after each sentence for context.

    Returns:
    - sentences (list): Updated list of sentence dictionaries with combined sentences.
    """
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence_before = ''
        combined_sentence_after = sentences[i]['sentence'] + ". "

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence_before += sentences[j]['sentence'] + '. '

        # Add the current sentence
        combined_sentence_before += sentences[i]['sentence'] + '. '

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence_after += sentences[j]['sentence']  + '. '

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence_before'] = combined_sentence_before
        sentences[i]['combined_sentence_after'] = combined_sentence_after

    return sentences

def compute_embeddings(sentences, hf):
    """
    Computes embeddings for combined sentences using a pretrained model.

    Parameters:
    - sentences (list): List of dictionaries, each containing a sentence and its combined sentences.
    - hf: Pretrained model used for generating sentence embeddings.

    Returns:
    - sentences (list): Updated list of sentence dictionaries with combined sentence embeddings.
    """
    embeddings = hf.embed_documents([x['combined_sentence_before'] for x in sentences])
    
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding_before'] = embeddings[i]
        
    embeddings = hf.embed_documents([x['combined_sentence_after'] for x in sentences])
    
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding_after'] = embeddings[i]
    
    
    return sentences

def calculate_cosine_distances(sentences):
    """
    Calculates the cosine distances between consecutive combined sentence embeddings.

    Parameters:
    - sentences (list): List of dictionaries, each containing combined sentence embeddings.

    Returns:
    - distances (list): List of cosine distances between consecutive combined sentence embeddings.
    - sentences (list): Updated list of sentence dictionaries with distance to the next sentence.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding_before']
        embedding_next = sentences[i + 1]['combined_sentence_embedding_after']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences

def chunks_from_distance(sentences, distances, breakpoint_percentile_threshold, min_size = 100):
    """
    Creates text chunks based on cosine distances between sentences.

    Parameters:
    - sentences (list): List of sentence dictionaries.
    - distances (list): Cosine distances between consecutive combined sentence embeddings.
    - breakpoint_percentile_threshold (int): Percentile threshold for determining chunk breakpoints.
    - min_size (int): Minimum size of a chunk in characters.

    Returns:
    - chunks (list): List of text chunks.
    - max_index_split (list): Indices where the text was split based on cosine distance.
    """
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list
    
    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []
    max_index_split = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
    
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = '. '.join([d['sentence'] for d in group]) + ". "
        
        if len(combined_text)<min_size :
            if chunks ==[] :
                continue
            if distances[start_index-1]<distances[end_index]:
                chunks[-1] = chunks[-1] + combined_text
                start_index = index + 1
        else :         
            chunks.append(combined_text)
            # Update the start index for the next group
            start_index = index + 1
        

    # The last group, if any sentences remain
    if start_index < len(sentences):
        group = sentences[start_index:]
        combined_text = '. '.join([d['sentence'] for d in group])
        
        if len(combined_text)<min_size :
            chunks[-1] = chunks[-1] + combined_text
        else : 
            chunks.append(combined_text)

    return chunks, max_index_split

def plot_chunking (distances,breakpoint_percentile_threshold):
    """
    Visualizes the chunking process by plotting cosine distances and breakpoints.

    Parameters:
    - distances (list): Cosine distances between consecutive sentence embeddings.
    - breakpoint_percentile_threshold (int): The percentile threshold used for identifying breakpoints.

    Displays:
    - A matplotlib plot showing the distances, breakpoints, and shaded chunks.
    """

    plt.figure(figsize=(15, 7))
    plt.plot(distances, color = "black");

    y_upper_bound = .2
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-');

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list
    
    plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{len(indices_above_thresh) + 1} Chunks");

    # Start of the shading and text
    colors = sns.color_palette("Paired")#['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, breakpoint_index in enumerate(indices_above_thresh):
        plt.plot(breakpoint_index, distances[breakpoint_index], 'o', color=colors[i % len(colors)])
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.5)
        plt.text(x=np.average([start_index, end_index]),
                y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                s=f"Chunk #{i}", horizontalalignment='center',
                rotation='vertical')

    # Additional step to shade from the last breakpoint to the end of the dataset
    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.5)
            plt.text(x=np.average([last_breakpoint, len(distances)]),
                    y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                    s=f"Chunk #{i+1}",
                    rotation='vertical')

    plt.title("Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in text (Sentence Position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()
    
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
    
    docs_init = []
    for d in tqdm(data):
        if len(d) < 1024 : 
            docs_init.append(Document(text = d))
        else : # if the length of the paragraph is superior to chunk_size split it
            _, _, chunks, _ = chunk_text_pairs([d], hf = hf, breakpoint_percentile_threshold=90, buffer_size=2)
            docs_init.extend([Document(text = chunk) for chunk in chunks])
    
    
    