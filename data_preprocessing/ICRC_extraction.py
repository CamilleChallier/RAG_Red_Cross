"""
Extract data from ICRC PDF documents, process the extracted text, and clean it for further analysis.

This script contains multiple utility functions for extracting, processing, and improving the quality of text data obtained from ICRC documents. 
The main tasks include title extraction, word occurrence checks, garbled text detection, and text cleaning. 
It relies on regular expressions, NLTK tokenization, and custom filtering rules to ensure the extracted text is clear and usable.

Functions:
    - extract_titles_and_text(text): Extracts section titles and corresponding content from the provided text.
    - check_word_occurrences(s, word): Checks if a specified word appears more than twice in a given text.
    - is_garbled(text): Identifies garbled sentences that contain excessive one- or two-character words.
    - clean_text(text): Cleans the provided text by filtering out garbled sentences using NLTK's sentence tokenization.

"""

import os
import re
import sys
sys.path.append('../')
from utils import clean_text
import re
import nltk
from nltk.tokenize import sent_tokenize
import pickle

def extract_titles_and_text(text):
    """
    Extracts section titles and corresponding text content from the provided document text.

    Args:
        text (str): The entire document text from which section titles and text need to be extracted.
    
    Returns:
        dict: A dictionary where keys are section titles (as found in the text) and values are the text under each section. 
              If there's any text before the first title, it's stored with an empty string as the key.
    """
    
    # Regular expression pattern for titles with either one or two hashes
    title_pattern = r'\n(#+ .+?)\n'
    # Splitting text into parts using the title pattern
    parts = re.split(title_pattern, text)
    
    content = {}
    # First part is the text before the first title
    pre_title_text = parts[0].strip()
    if pre_title_text:
        content[""] = pre_title_text
    
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        if i + 1 < len(parts):
            content[title] = parts[i + 1].strip()
    
    return content

def check_word_occurrences(s, word):
    """
    Checks if a specific word appears more than twice in the provided text.

    Args:
        s (str): The text to be analyzed.
        word (str): The word to count in the text.
    
    Returns:
        bool: True if the word appears more than twice (case-insensitive), False otherwise.
    """
    return s.lower().split().count(word.lower()) > 2

# Download necessary NLTK data
nltk.download('punkt')

def is_garbled(text):
    """
    Identifies whether the provided text contains garbled content, defined as an excessive number of one- or two-character words in a row.

    Args:
        text (str): The text to be analyzed for garbled content.
    
    Returns:
        bool: True if the text contains garbled content, False otherwise.
    """
    # Define regex pattern for garbled text
    garbled_pattern = r'(?i)(\b\w{1,2}\b\s*){5,}'
    return re.search(garbled_pattern, text) is not None

def clean_text(text):
    """
    Cleans the provided text by removing garbled sentences (as identified by `is_garbled` function) and returning the cleaned version.

    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned version of the text, with garbled sentences removed.
    """
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Filter out garbled sentences
    cleaned_sentences = [sentence for sentence in sentences if not is_garbled(sentence)]
    
    # Join the cleaned sentences
    return ' '.join(cleaned_sentences)


if __name__ == "__main__":
    # Specify the directory containing the files
    folder_path = '/mloscratch/homes/durech/ICRCPublicArchives/CIRC Circulaire/'

    # extract data
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if"ENG" in file_name:
                os.system(f'export EXTRACT_IMAGES=False; marker_single "{file_path}" "/mloscratch/homes/challier/Codex/data_ICRC" --langs English')
            else : 
                os.system(f'export EXTRACT_IMAGES=False; marker_single "{file_path}" "/mloscratch/homes/challier/Codex/data_ICRC" --langs French')
                
    # improve the extracted text
    folder_path = '/mloscratch/homes/challier/Codex/data_ICRC' # Specify the directory containing the files

    data = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Check if the file has .md extension
            if file_name.endswith('.md'):
                print(f"Checking {file_name} in {root}")
                # Construct full file path
                file_path = os.path.join(root, file_name)
                
                # Open and read the file
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    
                    if "CAP" in file_name : 
                        file_contents = file_contents.split('\n.\n')
                        # remove short lines
                        file_contents = [re.sub(r'COMMUNICATION TO THE.*?\n.*? N\'EXISTE PAS\n', '', d, flags=re.DOTALL) for d in file_contents]
                        file_contents = [d.lower() for d in file_contents if len(d) > 100]
                    
                    if "CIRC" in file_name : 
                        file_contents = file_contents.split('\n.\n')
                        # remove short lines
                        file_contents = [d.lower() for d in file_contents if len(d) > 100]
                    
                    if "CDP" in file_name or "RA" in file_name : 
                        # file_contents = file_contents.split('\n#')
                        file_contents = extract_titles_and_text(file_contents)
                        file_contents = {k : v for k,v in file_contents.items() if len(v) > 100}
                        
                        
                    if "annual-report" in file_name : 
                        # file_contents = file_contents.split('\n#')
                        file_contents = extract_titles_and_text(file_contents)
                        file_contents = {k : v for k,v in file_contents.items() if "--------------------------------------------------------" not in v and len(v) > 100}
                    data.append({"title" : file_name, "content" : file_contents})
                        
                        
    list_data = [] 
    #convert to list 
    for d in data :
        if type(d["content"]) == dict : 
            title = d["title"]
            for i, v in enumerate(d["content"].items()) : 
                if i > 0 and not "|----------" in v[1] and clean_text(v[1]) != "" :
                    list_data.append({"title" : title, "content" : v[0] + v[1], "id" : i})
        else :
            title = d["title"]
            for i in range(len(d["content"])) : 
                if i > 0 :
                    if not check_word_occurrences(d["content"][i], "content") and not check_word_occurrences(d["content"][i], "consider") and not "|----------" in d["content"][i] and clean_text(d["content"][i]) != "" :
                        list_data.append({"title" : title, "content" : d["content"][i], "id" : i})

    # cleaned_text = clean_text(list_data[0])
    
    # save data
    with open('data_ICRC.pkl', 'wb') as f:
        pickle.dump(list_data, f)
