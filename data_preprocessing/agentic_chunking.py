"""
This module provides utility functions for processing text using a agentic chunking. 

Functions:
- extract_list(output): Extracts a Python list from a JSON-formatted string that may be improperly formatted.
- proposal_indexing(text, pipeline, prompt, history_bot): Prepares a structured prompt for text decomposition, feeds it into a language model pipeline, and returns the model's generated output.
- chunks_and_summarize(text, pipeline): Breaks down a given text into smaller segments, each centered around specific topics, decontextualizes them, and then summarizes the results in JSON format.
"""
import ast
import re
import transformers

def extract_list(output):
    """
    Extracts a Python list from the 'content' field of a dictionary that contains JSON-formatted strings.
    
    Args:
    - output (dict): A dictionary with a 'content' key that holds a string representing a list.

    Returns:
    - list: The extracted list, evaluated as a Python object.
    """
    
    content = output["content"].strip()
    
    if content[0]!= "[" :
        match = re.search(r'\[\s*(".*?"\s*(?:,\s*".*?"\s*)*)\]', content, re.DOTALL)
        if match:
            content = "[" + match.group(1) +"]"
            
    if content[-1] != ']':
        content = content + "]"
        
    actual_list = ast.literal_eval(content)
    return actual_list

def proposal_indexing(text, pipeline, prompt, history_bot):
    """
    Prepares a structured prompt to decompose text, sends it to a language model pipeline, 
    and returns the final generated output.
    
    Args:
    - text (str): The text to be decomposed.
    - pipeline (Pipeline): The language model pipeline used for generating text.
    - prompt (str): The initial prompt for the model.
    - history_bot (str): Previous conversation history with the bot, used as context.

    Returns:
    - str: The last part of the generated output from the language model.
    """
    
    messages = []
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "system", "content": str(history_bot)})
    messages.append({"role": "user", "content": 'Decompose the following:' +str(text)})

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens= 1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]

def chunks_and_summarize(text, pipeline) :
    """
    Breaks down a text into segments of up to 500 words each, decontextualizes them, 
    and summarizes the results as a JSON-formatted list of strings.
    
    Args:
    - text (str): The text to be segmented and summarized.
    - pipeline (Pipeline): The language model pipeline used for processing the text.

    Returns:
    - str: A JSON-formatted list of strings containing the segmented and decontextualized summaries.
    """
    messages = []
    
    history_user = 'Decompose the following text into clear segments of a maximum of 500 words, each centered around specific topics or themes. Each segments should be interpretable out of context. Decontextualize the paragraphs by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to. Present the results as a list of strings, formatted in JSON. \n Exemple : Text : Germany is the seventh-largest country in Europe; bordering Denmark to the north, Poland and the Czech Republic to the east, Austria to the southeast, and Switzerland to the south-southwest. France, Luxembourg and Belgium are situated to the west, with the Netherlands to the northwest. Germany is also bordered by the North Sea and, at the north-northeast, by the Baltic Sea. German territory covers 357,600 km2, consisting of 349,250 km2 of land and 8,350 km2 of water. Elevation ranges from the mountains of the Alps in the south to the shores of the North Sea in the northwest and the Baltic Sea (Ostsee) in the northeast. The forested uplands of central Germany and the lowlands of northern Germany are traversed by major rivers as the Rhine, Danube and Elbe. Significant natural resources include iron ore, coal, potash, timber, lignite, uranium, copper, natural gas, salt, and nickel. \n Output : ["Germany is the seventh-largest country in Europe. Germany is the seventh-largest country in Europe; bordering Denmark to the north, Poland and the Czech Republic to the east, Austria to the southeast, and Switzerland to the south-southwest. France, Luxembourg and Belgium are situated to the west, with the Netherlands to the northwest. Germany is also bordered by the North Sea and, at the north-northeast, by the Baltic Sea.", "content": "German territory covers 357,600 km², consisting of 349,250 km² of land and 8,350 km² of water. Elevation in Germany ranges from the mountains of the Alps in the south to the shores of the North Sea in the northwest and the Baltic Sea in the northeast. The forested uplands of central Germany and the lowlands of northern Germany are traversed by major rivers such as the Rhine, Danube, and Elbe.", "Significant natural resources in Germany include iron ore, coal, potash, timber, lignite, uranium, copper, natural gas, salt, and nickel."]'
    history_bot = 'I will output only the results as one line of list of strings, formatted in JSON.'

    # history_user = ["Could you please divide a given text into segments of maximum 500 words, each centered around specific topics or themes? For each segment, start by a short introduction allowing each part to stand independently."]

    messages.append({"role": "user", "content": str(history_user)})
    messages.append({"role": "system", "content": str(history_bot)})
    messages.append({"role": "user", "content": "Text : "  + str(text) + "\n Output : "})

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]

if __name__ == "__main__":
    
    import sys
    sys.path.append('../')
    from utils import read_file
    from datasets import load_dataset
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    import torch

    # load data
    def load_wiki_data():
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        data = read_file(ds["train"]['text'])
        return data
    data = load_wiki_data()
    data = [d.replace("= \n", " : ") for d in data] 
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
                                        "text-generation",
                                        model=model_id,
                                        model_kwargs={"torch_dtype": torch.bfloat16},
                                        device_map="cuda:0",
                                        max_new_tokens=6000,
                                    )

    chunk_size = 1024
    
    docs=[]
    for i, d in enumerate(data): #for each document
        paragraphs = d.split("\n =")
        print(i)
        
        for para in paragraphs : #for each paragraph
            if len(para)> chunk_size : # if the paragraph is too big, split it a first time before feeding to the llm
                splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
                nodes = splitter.get_nodes_from_documents([Document(text =para, doc_id = str(i))])

                for node in nodes : 
                    chunks= extract_list(chunks_and_summarize(node.text, pipeline)) # extract sentences
                    docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks])
            else :  
                chunks= extract_list(chunks_and_summarize(para, pipeline)) # extract sentences
                docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks]) 