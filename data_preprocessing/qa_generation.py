"""
Notebook to :
- generate chunks of data using different splitting techniques
- generate queries from these chunks
"""

import argparse
import pickle
import sys
sys.path.append('../')

import random
import transformers
import torch
from tqdm import tqdm

from datasets import load_dataset
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from utils import read_file, select_qa
from semantic_chunking_pairs import chunk_text_pairs
from chain_chunking import chunk_text_chain
from semantic_chunking import chunk_text
from agentic_chunking import proposal_indexing, extract_list, chunks_and_summarize


qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be developped and diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""


def load_wiki_data(chunking_type ="char"):
    """
    Loads and processes the WikiText-103 dataset based on a specified chunking type. 
    The dataset is segmented into chunks using various strategies, such as character-based, sentence-based, or semantic chunking. 
    These chunks are stored in `Document` objects, and nodes are extracted from these documents for further use.

    Args:
        chunking_type (str, optional): The type of chunking to apply. Options include:
            - "char": Splits text based on character count with a specified chunk size and overlap.
            - "sentence": Splits text based on sentence boundaries.
            - "recursive": Uses recursive character splitting based on a chunk size and overlap.
            - "semantic_pairs": Splits text semantically using pairwise semantic chunking.
            - "semantic": Splits text semantically based on content understanding.
            - "chain": Splits text using chain-based semantic chunking.
            - "LDA": Uses Latent Dirichlet Allocation (LDA) to split text into chunks based on topic coherence.
            - "combined": A combination of multiple chunking methods, using an LLM (Large Language Model) for decomposition.
            - "llm": Uses an LLM to decompose text into standalone propositions.

    Returns:
        tuple:
            - docs (list): A list of `Document` objects containing the chunked text.
            - nodes (list): A list of nodes derived from the chunked documents.
    """

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    data = read_file(ds["train"]['text'])
    
    if chunking_type == "char":
        docs = [Document(text =row, doc_id=i) for i, row in enumerate(data)] # create documents from each articles
        splitter = TokenTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        separator=" ",
) # initialize splitter based on token number
        
    if chunking_type == "sentence":
        docs = [Document(text =row, doc_id=i) for i, row in enumerate(data)] # create documents from each articles
        splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap) # initialize splitter based on sentences
    
    if chunking_type == "recursive":   
        data = [d.replace("= \n", "") for d in data]  # change title name to help in separation of paragraph
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = args.chunk_size, chunk_overlap=args.overlap) # use recursive splitting from longchain
        docs = [Document(text =doc.page_content, doc_id=i) for i, d in enumerate(data) for doc in text_splitter.create_documents([d])]
        splitter = SentenceSplitter() #no effect just to obtain nodes
        
    if chunking_type == "semantic_pairs":
        from semantic_chunking_pairs import chunk_text_pairs
        docs = []
        data = [d.replace("= \n", " : ") for d in data]  # change title name to help in separation of paragraph
        
        # load embedding model
        model_kwargs = {"device": "cuda:0"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=args.model_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    
        for i,d in enumerate(data):
            print(f"Processing document {i}")
            _, _, chunks, _ = chunk_text_pairs([d], hf = hf, breakpoint_percentile_threshold=90, buffer_size=1)
            docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks])
        splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
        
    if chunking_type == "semantic":
        docs = []
        data = [d.replace("= \n", " : ") for d in data]  # change title name to help in separation of paragraph
        for i,d in enumerate(data):
            print(f"Processing document {i}")
            
            _, _, chunks, _ = chunk_text([d], model_name_embed=args.model_embed, device="cuda", buffer_size=args.buffer_size, breakpoint_percentile_threshold=90, chunk_size=args.chunk_size, split=True)
            docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks if len(chunk) >100])
        splitter = SentenceSplitter() 
    
    if chunking_type == "chain":
        
        # load embedding model
        model_kwargs = {"device": "cuda:0"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=args.model_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        
        docs = []
        from chain_chunking import chunk_text_chain
        for d in tqdm(data):
            paragraphs = d.split("\n =")
            for para in paragraphs :
                if len(para) > 200 : 
                    if len(para)< args.chunk_size :
                        docs.append(Document(text = para))
                    else : 
                        _,chunks =  chunk_text_chain([para], hf=hf, thresh = 0.4)
                        docs.extend([Document(text = chunk) for chunk in chunks])
        splitter = SentenceSplitter() #no effect just to obtain nodes
        
    if chunking_type == "LDA":
        docs = []
        data = [d.replace("= \n", " : ") for d in data]  # change title name to help in separation of paragraph
        from LDA_chunking import create_sentences, LDA, chunks_from_coherence
        
        for i,d in enumerate(data):
            print(f"Processing document {i}")
            data_split, tokenized_sents = create_sentences([d])
            coherence_scores = LDA(tokenized_sents)
            chunks = chunks_from_coherence(data_split, coherence_scores, breakpoint_threshold=0.5)
            docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks])
        splitter = SentenceSplitter()  #no effect just to obtain nodes
            
    if chunking_type == "combined": # agentic + semantic
        # load model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda:0",
            max_new_tokens=4000,
        )
        #prompt 
        prompt = 'System: Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context. \n 1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible. \n 2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition. \n 3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to. \n 4. Present the results as a list of strings, formatted in JSON. \n Example: Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny." Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."]'
        history_bot = 'I will output only the results as one line of list of strings, formatted in JSON.'

        docs = []
        data = [d.replace("= \n", " : ") for d in data]  # change title name to help in separation of paragraph
        for i, d in enumerate(data[0:500]):
            paragraphs = d.split("\n =")
            
            for para in paragraphs :
                if len(para)> args.chunk_size :
                    para= extract_list(proposal_indexing(para, pipeline, prompt, history_bot))
                    _, _, chunks, _ = chunk_text(para, model_name_embed=args.model_embed, breakpoint_percentile_threshold=95, chunk_size=None, split=False)
                    docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks])
                else :   
                    docs.append(Document(text = para, metadata = {"doc_id" : i }))
        splitter = SentenceSplitter() #no effect just to obtain nodes
        
    if chunking_type == "llm" : 
        
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        chunk_size = 2048
        docs=[]
        data = [d.replace("= \n", " : ") for d in data] 
        for i, d in enumerate(data[0:1]):
            paragraphs = d.split("\n =")
            for para in paragraphs :
                if len(para)> chunk_size :
                    chunks= extract_list(chunks_and_summarize(para))
                    docs.extend([Document(text = chunk, metadata = {"doc_id" : i }) for chunk in chunks])
                else :   
                    docs.append(Document(text = para, metadata = {"doc_id" : i }))
    
    nodes = splitter.get_nodes_from_documents(docs)
    return docs, nodes

def load_icrc_data(chunking_type ="char"):
    
    """
    Loads and processes the ICRC (International Committee of the Red Cross) dataset based on the specified chunking type. 
    The dataset is segmented into chunks, with each chunk stored in a `Document` object for further processing, analysis, or indexing. 
    Different chunking strategies can be applied, such as recursive character splitting or semantic chunking.

    Args:
        chunking_type (str, optional): The type of chunking to apply to the dataset. Options include:
            - "recursive": Uses recursive character splitting based on a chunk size and overlap.
            - "semantic_pairs": Splits text semantically using pairwise semantic chunking.
            - "semantic": Splits text semantically based on content understanding.
            - "chain": Splits text using chain-based semantic chunking.
            - "LDA": Uses Latent Dirichlet Allocation (LDA) to split text into chunks based on topic coherence.

    Returns:
        tuple:
            - docs (list): A list of `Document` objects containing the chunked text from the ICRC dataset, along with associated metadata (e.g., title, document ID).
            - nodes (list): A list of nodes derived from the chunked documents, useful for further operations like summarization or knowledge extraction.
    """
    
    with open('../data/data_ICRC.pkl', 'rb') as f:
        data = pickle.load(f)   
    
    if chunking_type == "recursive": 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = args.chunk_size, chunk_overlap=args.overlap) # use recursive splitting from longchain
        docs = [Document(text =doc.page_content, metadata = {"title" : d["title"], "doc_id" : d["id"]}) for d in tqdm(data) for doc in text_splitter.create_documents([d["content"]])]
        splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap) #no effect just to obtain nodes
        
    if chunking_type == "semantic":
        docs = []
        for d in tqdm(data):
            if len(d["content"]) > 200 : # remove bad paragraph that do not form a sentence
                if len(d["content"]) < args.chunk_size :
                    docs.append(Document(text = d["content"], metadata =  {"title" : d["title"], "doc_id" : d["id"]}))
                else :  
                    _, _, chunks, _ = chunk_text([d["content"]], model_name_embed=args.model_embed, device="cuda:0", buffer_size=args.buffer_size, breakpoint_percentile_threshold=95, chunk_size=None, split=True) 
                    docs.extend([Document(text = chunk, metadata =  {"title" : d["title"], "doc_id" : d["id"]}) for chunk in chunks])
        splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
        
    if chunking_type == "semantic_pairs":
        
        #load model
        model_kwargs = {"device": "cuda:0"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=args.model_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        docs_init = []
        for d in tqdm(data):
            if len(d["content"]) > 200 :   # remove bad paragraph that do not form a sentence
                if len(d["content"]) < args.chunk_size : 
                    docs_init.append(Document(text = d["content"], metadata =  {"title" : d["title"], "doc_id" : d["id"]}))
                else :         
                    _, _, chunks, _ = chunk_text_pairs([d["content"]], hf = hf, breakpoint_percentile_threshold=90, buffer_size=args.buffer_size)
                    docs_init.extend([Document(text = chunk, metadata =  {"title" : d["title"], "doc_id" : d["id"]}) for chunk in chunks])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = args.chunk_size, chunk_overlap=args.overlap)
        docs = [Document(text =doc.page_content, metadata = d.metadata) for d in tqdm(docs_init) for doc in text_splitter.create_documents([d.text])] # cut larger chunk
        splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.overlap) #no effect just to obtain nodes
        
    if chunking_type == "chain":
        docs = []

        model_kwargs = {"device": "cuda:0"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=args.model_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        for d in tqdm(data):
            if len(d["content"]) > 200 :  # remove bad paragraph that do not form a sentence
                if len(d["content"]) < args.chunk_size : 
                    docs.append(Document(text = d["content"], metadata =  {"title" : d["title"], "doc_id" : d["id"]}))
                else : 
                    _,chunks = chunk_text_chain([d["content"]],hf, thresh = 0.3,  device="cuda")
                    docs.extend([Document(text = chunk, metadata = {"title" : d["title"], "doc_id" : d["id"]}) for chunk in chunks])
        splitter = SentenceSplitter() # cut large chunks
        
    if chunking_type == "LDA":
        docs = []
        from LDA_chunking import create_sentences, LDA, chunks_from_coherence
        for d in tqdm(data):
            if len(d["content"]) > 200 :  # remove bad paragraph that do not form a sentence
                if len(d["content"]) < args.chunk_size : 
                    docs.append(Document(text = d["content"], metadata =  {"title" : d["title"], "doc_id" : d["id"]}))
                else : 
                    data_split, tokenized_sents = create_sentences([d["content"]])
                    if len(data_split) > 2 : 
                        coherence_scores = LDA(tokenized_sents)
                        chunks = chunks_from_coherence(data_split, coherence_scores, breakpoint_threshold=0.5)
                        docs.extend([Document(text = chunk, metadata =  {"title" : d["title"], "doc_id" : d["id"]}) for chunk in chunks])
                    else : 
                        docs.append(Document(text = d["content"], metadata =  {"title" : d["title"], "doc_id" : d["id"]}))
        splitter = SentenceSplitter()  # cut large chunks
    
    nodes = splitter.get_nodes_from_documents(docs) # form nodes
    print(f"Number of nodes: {len(nodes)}")
    return docs, nodes
    

def main(args):
    
    if args.dataset_name == "wikitext":
        _, nodes = load_wiki_data(args.chunking_type)
    elif args.dataset_name == "icrc":
        _, nodes = load_icrc_data(args.chunking_type)
    else :
        raise(ValueError, f"Dataset {args.dataset_name} not found, should be either 'wikitext' or 'icrc'")
    
    # load queries generator model
    llm_llama3 = LlamaCPP(
        model_url=args.model_url,
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},  # if compiled to use GPU
        verbose=True,
    )

    #generate question context pair from 500 random chunks
    qa_dataset_llama = generate_question_context_pairs(
        random.sample(list(nodes), k = 500),
        llm=llm_llama3,
        num_questions_per_chunk=2,
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
    )
    
    # filter the generated queries, remove the one that start by "question :" or other errors
    new_queries, new_corpus, new_docs = select_qa(qa_dataset_llama)

    # create new Dataset
    qa_dataset_llama = EmbeddingQAFinetuneDataset(
            queries=new_queries, corpus=new_corpus, relevant_docs=new_docs
        )
    
    # save queries
    with open(f'{args.dataset_name}_qa_dataset.pkl', 'wb') as f:
        pickle.dump(qa_dataset_llama, f)
       
    # save nodes 
    with open(f'nodes_{args.dataset_name}_{args.chunking_type}_{args.buffer_size}_{args.chunk_size}.pkl', 'wb') as f:
        pickle.dump(nodes, f)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to generate and save a question-answer dataset with llama_index")
    parser.add_argument('--dataset_name', type=str, default='icrc', help="Dataset name to use for question-answer generation")
    parser.add_argument('--model_url', type=str, default="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", help="Model URL to use for question-answer generation")
    parser.add_argument('--model_embed', type=str, default="w601sxs/b1ade-embed", help="Model embedding to use for chunking")
    parser.add_argument('--chunking_type', type=str, default="semantic2", help="Chunking type to use for dataset")
    parser.add_argument('--chunk_size', type=int, default=2048, help="Chunk size to use for dataset")
    parser.add_argument('--buffer_size', type=int, default=2, help="Buffer size to use for dataset")
    parser.add_argument('--overlap', type=int, default =20, help="Overlap to use for dataset")
    args = parser.parse_args()
    main(args)