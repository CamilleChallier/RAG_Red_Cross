"""
Text Chunking with Embedding-Based Cosine Similarity

This script provides a method for splitting text into coherent chunks using sentence embeddings 
and cosine similarity. The primary objective is to identify semantic shifts in the text and split 
it into meaningful sections based on those shifts. The approach involves the following steps:

1. Splitting the text into individual sentences.
2. Creating combined sentence contexts for each sentence.
3. Generating embeddings for these combined contexts.
4. Calculating cosine distances between consecutive sentence embeddings.
5. Identifying breakpoints where cosine distances exceed a threshold.
6. Splitting the text into chunks based on these breakpoints.

Functions:
- `chunk_text`: Main function to chunk the text based on cosine similarity between sentence embeddings.
- `split_sentences`: Splits the input text into individual sentences.
- `combine_sentences`: Combines adjacent sentences to form a context window around each sentence.
- `compute_embeddings`: Generates sentence embeddings using a specified embedding model.
- `calculate_cosine_distances`: Calculates cosine distances between consecutive combined sentence embeddings.
- `split_text`: Recursively splits large chunks of text based on maximum cosine distances.
- `chunks_from_distance`: Identifies breakpoints based on cosine distances and splits the text into chunks.
- `plot_chunking`: Visualizes the chunking process by plotting cosine distances and breakpoints.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import seaborn as sns

from llama_index.core import Document
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import read_file
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def chunk_text(data, model_name_embed="BAAI/bge-small-en", device="cpu",buffer_size=1, breakpoint_percentile_threshold=95, chunk_size=None, split=True, pattern = r'[.?!]\n+'):
    """
    Main function to chunk the text based on cosine similarity between sentence embeddings.

    Parameters:
    - data (list): The input text data split into paragraphs or sentences.
    - model_name_embed (str): The name of the HuggingFace embedding model to use.
    - device (str): The device to run the embeddings model on, e.g., "cpu" or "cuda".
    - buffer_size (int): The number of surrounding sentences to include in the context window.
    - breakpoint_percentile_threshold (int): The percentile threshold to identify breakpoints.
    - chunk_size (int, optional): The maximum size of each chunk in characters.
    - split (bool): Whether to split the input text into sentences.

    Returns:
    - distances (list): Cosine distances between consecutive sentence embeddings.
    - sentences (list): List of sentences with their embeddings.
    - chunks (list): List of text chunks identified.
    - max_index_split (list): Indices where the text was split based on cosine distance.
    """
    # Split the input text into individual sentences.
    sentences = split_sentences(data, split, pattern)
    if len(sentences) < 2 :
        return None, None, data, None
    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = combine_sentences(sentences, buffer_size=buffer_size)
    
    embedded_sentences = compute_embeddings(combined_sentences, model_name=model_name_embed, device=device)
    
    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances, sentences = calculate_cosine_distances(embedded_sentences)
    
    # Determine the breakpoints based on the cosine distances.
    chunks, max_index_split = chunks_from_distance(sentences, distances, breakpoint_percentile_threshold=breakpoint_percentile_threshold,chunk_size=chunk_size)
    return distances, sentences, chunks, max_index_split

def split_sentences(data, split, pattern):
    """
    Splits the input text into individual sentences based on punctuation.

    Parameters:
    - data (list): The input text data as a list of paragraphs or sentences.
    - split (bool): Whether to split the input text into sentences.

    Returns:
    - sentences (list): List of dictionaries containing sentences and their indices.
    """
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    if split : 
        data = [sentence for d in data for sentence in re.split(pattern, d) if sentence != "" ]
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(data)]
    return sentences

def combine_sentences(sentences, buffer_size):
    """
    Combines adjacent sentences to form a context window around each sentence.

    Parameters:
    - sentences (list): List of dictionaries containing sentences and their indices.
    - buffer_size (int): The number of surrounding sentences to include in the context window.

    Returns:
    - sentences (list): List of dictionaries with combined sentences for context.
    """
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def compute_embeddings(sentences, model_name="BAAI/bge-small-en", device="cpu"):
    """
    Generates sentence embeddings using a specified HuggingFace model.

    Parameters:
    - sentences (list): List of dictionaries containing combined sentences for context.
    - model_name (str): The name of the HuggingFace embedding model to use.
    - device (str): The device to run the embeddings model on, e.g., "cpu" or "cuda".

    Returns:
    - sentences (list): List of dictionaries with sentence embeddings added.
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    embeddings = hf.embed_documents([x['combined_sentence'] for x in sentences])
    
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    
    return sentences

def calculate_cosine_distances(sentences):
    """
    Calculates cosine distances between consecutive combined sentence embeddings.

    Parameters:
    - sentences (list): List of dictionaries with sentence embeddings.

    Returns:
    - distances (list): Cosine distances between consecutive sentence embeddings.
    - sentences (list): Updated list of sentences with distance information.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
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

def split_text(group, distances,combined_text, max_cut, chunk_size=1024):
    """
    Recursively splits large chunks of text based on maximum cosine distances.

    Parameters:
    - group (list): List of sentence dictionaries for the current chunk.
    - distances (list): List of cosine distances between sentences in the group.
    - combined_text (str): The combined text of the current chunk.
    - max_cut (list): List to store max cosine distances for each split.
    - chunk_size (int): The maximum size of each chunk in characters.

    Returns:
    - list: The final list of chunks.
    - list: Updated list of max cosine distances.
    """
    def find_max_index(distances):
        return distances.index(max(distances[:-1]))
    
    def split_recursive(group, distances, combined_text, max_cut, chunk_size):
        # combined_text = ' '.join([d['sentence'] for d in group])
        if len(combined_text) <= chunk_size or len(distances) <= 1:
            return combined_text, max_cut
        # max_indices = []
        max_index = find_max_index(distances)
        max_cut.append(max(distances[:-1]))
        
        left_group = group[:max_index+1]
        right_group = group[max_index+1:]
        left_distances = distances[:max_index+1]
        right_distances = distances[max_index+1:]
        
        
        combined_text_left = ' '.join([d['sentence'] for d in left_group])
        if len(combined_text_left) > chunk_size and len(left_distances) > 1:
            left_chunks, max_left = split_recursive(left_group, left_distances, combined_text_left, max_cut, chunk_size)
        else:
            max_left= max_cut
            left_chunks = [combined_text_left]
        
        combined_text_right = ' '.join([d['sentence'] for d in right_group])
        if len(combined_text_right) > chunk_size and len(right_distances) > 1:
            right_chunks, max_right = split_recursive(right_group, right_distances, combined_text_right, max_cut, chunk_size)
        else:
            max_right = max_cut
            right_chunks = [combined_text_right]
        
        return left_chunks + right_chunks, max_left + max_right
    
    return split_recursive(group, distances,combined_text,max_cut, chunk_size)

def chunks_from_distance(sentences, distances, breakpoint_percentile_threshold, chunk_size=None):
    """
    Identifies breakpoints based on cosine distances and splits the text into chunks.

    Parameters:
    - sentences (list): List of dictionaries containing sentences and their embeddings.
    - distances (list): List of cosine distances between consecutive sentence embeddings.
    - breakpoint_percentile_threshold (int): The percentile threshold to identify breakpoints.
    - chunk_size (int, optional): The maximum size of each chunk in characters.

    Returns:
    - chunks (list): List of text chunks identified.
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
        combined_text = ' '.join([d['sentence'] for d in group])
        
        if len(combined_text)<100 :
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
        combined_text = ' '.join([d['sentence'] for d in group])
        
        if chunk_size : 
            if len(combined_text) > chunk_size:
                # If the combined text is larger than the chunk size, split it into smaller chunks
                sub_chunks, max_cut = split_text(group, distances[start_index:], combined_text, [], chunk_size)
                max_indices = [i+start_index for i, val in enumerate(distances[start_index:]) if val in max_cut]            
                max_index_split.extend((sorted(np.unique(max_indices))))
                chunks.extend(sub_chunks)
            else :
                chunks.append(combined_text)
        else : 
            chunks.append(combined_text)

    return chunks, max_index_split

def plot_chunking (distances,breakpoint_percentile_threshold, max_index_split=None):
    """
    Visualizes the chunking process by plotting cosine distances and breakpoints.

    Parameters:
    - distances (list): Cosine distances between consecutive sentence embeddings.
    - breakpoint_percentile_threshold (int): The percentile threshold used for identifying breakpoints.
    - max_index_split (list, optional): Additional indices where the text was split based on chunk size.

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

    if max_index_split : 
        indices_above_thresh.extend(np.unique(max_index_split))
        #reorder list
        indices_above_thresh = sorted(np.unique(indices_above_thresh))
        
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
    
    for d in tqdm(data): #for each articles
        paragraphs = d.split("\n =")
        for para in paragraphs : #for each paragraph
            if len(para)< 1024 :
                docs.append(Document(text = para))
            else : 
                _, _, chunks, _ = chunk_text([para], model_name_embed="w601sxs/b1ade-embed", device="cuda:0")
                docs.extend([Document(text = chunk) for chunk in chunks])