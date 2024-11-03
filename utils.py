"""
This script contains functions for processing and cleaning text data, evaluating retrievers, and handling dataset-specific operations.

Functions include:
- `is_garbled(text)`: Checks if a text is garbled based on a regex pattern.
- `clean_text(text)`: Cleans text by removing garbled sentences and joining the remaining sentences.
- `istitle(line)`: Determines if a line is a title in a Wikitext format.
- `read_file(lines)`: Processes lines of a file into articles based on Wikitext structure.
- `create_documents(data, llama_index_doc=True)`: Creates documents from dataset contents, optionally in `Document` format.
- `display_results_retriever(name, eval_results, metrics=["mrr", "hit_rate"])`: Displays evaluation results for a retriever.
- `select_qa(dataset)`: Selects and filters queries and documents from a QA dataset.
- `retriever_evaluation(retriever, node_postprocessor=None, metrics=["hit_rate","mrr"])`: Evaluates a retriever using specified metrics.
- `found_language(title)`: Determines the language of a document based on its title.
- `get_title(nodes, idx)`: Retrieves the title of a node by its index.
- `print_result_lang(base_eval_results, nodes)`: Prints evaluation results separated by language.
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.evaluation import RetrieverEvaluator

import re
import nltk
from nltk.tokenize import sent_tokenize

def is_garbled(text):
    """
    Checks if a given text is garbled based on a regex pattern.

    Args:
        text (str): The text to be checked.

    Returns:
        bool: True if the text matches the garbled pattern, False otherwise.
    """
    # Define regex pattern for garbled text
    garbled_pattern = r'(?i)(\b\w{1,2}\b\s*){5,}'
    return re.search(garbled_pattern, text) is not None

def clean_text(text):
    """
    Cleans the input text by removing garbled sentences and joining the remaining sentences.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text with garbled sentences removed.
    """
    # Tokenize text into sentences
    nltk.download('punkt')
    sentences = sent_tokenize(text)
    
    # Filter out garbled sentences
    cleaned_sentences = [sentence for sentence in sentences if not is_garbled(sentence)]
    
    # Join the cleaned sentences
    return ' '.join(cleaned_sentences)

#### Wikitext dataset processing functions ####

def istitle(line):
    """
    Determines if a given line is a title in Wikitext format.

    Args:
        line (str): The line to be checked.

    Returns:
        bool: True if the line matches the Wikitext title pattern, False otherwise.
    """
    return len(re.findall(r'^\s* = [^=]* = $', line)) != 0

def read_file(lines):
    """
    Processes lines of a file into articles based on Wikitext structure.

    Args:
        lines (list of str): Lines read from a file.

    Returns:
        np.ndarray: An array of articles, where each article is a concatenation of lines.
    """
    articles = []
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and (lines[i+1] == ' \n' or lines[i+1]=='') and istitle(lines[i+2]):
            articles.append(current_article)
            current_article = ''
        if i >= 80000 :
            break
    articles.append(current_article)
    return np.array(articles)

#### ICRC dataset processing functions ####

def create_documents(data, llama_index_doc = True):
    """
    Creates documents from dataset contents, optionally in `Document` format.

    Args:
        data (dict): A dictionary where keys are filenames and values are dictionaries of pages with content.
        llama_index_doc (bool): If True, creates `Document` objects; otherwise, creates dictionaries.

    Returns:
        list: A list of documents created from the data.
    """
    documents = []
    # i = 0
    for filename, pages in data.items():
        print(f"Processing {filename}")
        for page_number, contents in pages.items():
            # print(f"Processing page {page_number}")
            title = ""
            text_set = set()  # Set to store unique text within the same page
            for content in contents:
                for type, info in content.items():
                    # print(type)
                    if type == 'title':
                        title = info
                    if type == 'text':
                        text = info
                        if text not in text_set:  # Check if text is already present in the set
                            if llama_index_doc:
                                documents.append(Document(
                                                    # doc_id=i,
                                                    text=text,
                                                    metadata={
                                                            'filename': filename,
                                                            'page_number': page_number,
                                                            'title': title
                                                            }
                                                        )
                                                )
                            else : 
                                documents.append({"text": text, "metadata":{
                                                        'filename': filename,
                                                        'page_number': page_number,
                                                        'title': title}})
                            # i += 1
                            text_set.add(text)

    return documents


def display_results_retriever(name, eval_results, metrics = ["mrr", "hit_rate"]):
    """
    Displays evaluation results for a retriever.

    Args:
        name (str): The name of the retriever.
        eval_results (list of EvaluatorResult): The results from the retriever evaluation.
        metrics (list of str): The metrics to be displayed.

    Returns:
        pd.DataFrame: A DataFrame containing the average results for each metric.
    """
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    result = {}
    result["Retriever Name"] =  [name]
    for metric in metrics :
        result[metric]= full_df[metric].mean()

    return pd.DataFrame(result)

def select_qa(dataset):
    """
    Selects and filters queries and documents from a QA dataset.

    Args:
        dataset (Dataset): The QA dataset.

    Returns:
        tuple: A tuple containing filtered queries, corpus, and relevant documents.
    """

    new_corpus = dataset.corpus

    keys = [k for k, v in dataset.queries.items() if not (v.startswith("```"))and "uestion" not in v and not (v.startswith("import"))]
    new_queries = {x: dataset.queries[x] for x in keys}
    new_docs = {x: dataset.relevant_docs[x] for x in keys}

    return new_queries, new_corpus, new_docs

def retriever_evaluation (retriever, node_postprocessor = None, metrics = ["hit_rate","mrr"]) :
    """
    Evaluates a retriever using specified metrics.

    Args:
        retriever (Retriever): The retriever to be evaluated.
        node_postprocessor (NodePostprocessor, optional): A postprocessor for nodes.
        metrics (list of str): Metrics for evaluation.

    Returns:
        RetrieverEvaluator: An evaluator configured with the specified metrics and retriever.
    """
    # print(node_postprocessor)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
      metric_names =metrics, retriever=retriever, node_postprocessors = node_postprocessor, device = "cuda"
    )

    return retriever_evaluator

def found_language(title) :
    """
    Determines the language of a document based on its title.

    Args:
        title (str): The title of the document.

    Returns:
        str: The language of the document ("ENG" or "FR").
    """
    if "ENG" in title or "annual-report"in title or "-en-" in title:
        return "ENG"
    elif "FRE" in title or "-fr-" in title: 
        return "FR"
    else :
        return "FR"
    
def get_title(nodes, idx):
    """
    Retrieves the title of a node by its index.

    Args:
        nodes (list of Node): List of nodes.
        idx (str): The index of the node.

    Returns:
        str: The title of the node.
    """
    for node in nodes :
        if node.id_ == idx :
            return node.metadata["title"]

def print_result_lang(base_eval_results, nodes) : 
    """
    Prints evaluation results separated by language.

    Args:
        base_eval_results (list of EvaluatorResult): The results from the evaluation.
        nodes (list of Node): List of nodes to retrieve titles.

    Returns:
        tuple: A tuple containing the overall accuracy, French accuracy, and English accuracy.
    """

    results = {}
    results["FR"] = []
    results["ENG"] = []

    for i in tqdm(range(0, len(base_eval_results))) :
        expected = base_eval_results[i].expected_ids[0]
        retrieved = base_eval_results[i].retrieved_ids
        
        # id = vector_store_dict["text_id_to_ref_doc_id"][expected]
        # print(id)
        title = get_title(nodes, expected)
        # print(title)
        
        language = found_language(title)
        # print(language)
        
        if expected in retrieved :
            results[language].append(1)
        else :
            results[language].append(0)

    print((sum(results["FR"]) + sum(results["ENG"]))/(len(results["FR"]) + len(results["ENG"])))
    print(sum(results["FR"])/len(results["FR"]))
    print(sum(results["ENG"])/len(results["ENG"]))
    return (sum(results["FR"]) + sum(results["ENG"]))/(len(results["FR"]) + len(results["ENG"])), sum(results["FR"])/len(results["FR"]),sum(results["ENG"])/len(results["ENG"])
            
