"""
Useful functions de generate broad queries and evaluate the performance of the summarizer head in RAG model
"""

from tqdm import tqdm
import re

from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

def preprocess(text, stop_words):
    """
    Tokenizes and preprocesses the input text, removing stopwords and short 
    tokens.

    Parameters:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to be removed from the text.
    Returns:
        list: A list of preprocessed tokens.
    """
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result

def get_topic_lists_from_pdf(nodes, num_topics, words_per_topic):
    """
    Extracts topics and their associated words from a PDF document using the 
    Latent Dirichlet Allocation (LDA) algorithm.

    Parameters:
        nodes: object of class llamaindex containing chunks of text
        num_topics (int): The number of topics to discover.
        words_per_topic (int): The number of words to include per topic.

    Returns:
        list: A list of num_topics sublists, each containing relevant words 
        for a topic.
    """

    # Extract the text from each page into a list. Each page is considered a document
    documents= []
    for chunk in nodes:
        documents.append(chunk.text)

    # Preprocess the documents
    import nltk
    dler = nltk.downloader.Downloader()
    dler._update_index()
    dler._status_cache['panlex_lite'] = 'installed' # Trick the index to treat panlex_lite as it's already installed.
    dler.download('stopwords')
    stop_words = set(stopwords.words(['english','spanish', 'french']))
    processed_documents = [preprocess(doc, stop_words) for doc in documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    # Build the LDA model
    lda_model = LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word=dictionary, 
        passes=15
        )

    # Retrieve the topics and their corresponding words
    topics = lda_model.print_topics(num_words=words_per_topic)

    # Store each list of words from each topic into a list
    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', '').strip() for word in words]
        topics_ls.append(topic_words)

    return topics_ls

DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information about the International Committee of the Red Cross (ICRC) and their interventions is below.

---------------------
{context_str}
---------------------
You are a Teacher/ Professor. Your task is to setup {num_question} open questions for an upcoming quiz/examination. \
Given the context information and not prior knowledge, generate only questions based on the previous keyword themes.
"""

def generate_broad_qa(
    keyword,
    llm,
    num_question=1,
    qa_generate_prompt_tmpl = DEFAULT_QA_GENERATE_PROMPT_TMPL) : 
    """
    Generates a specified number of questions based on input keywords using the provided LLM model.

    Args:
        keyword (list of str): A list of keywords or context strings for question generation.
        llm (LlamaCpp or compatible model): The language model used for generating questions.
        num_question (int, optional): The number of questions to generate per keyword. Defaults to 1.
        qa_generate_prompt_tmpl (str, optional): A template for formatting the question-generation prompt.

    Returns:
        list of str: A list of generated questions.
    """
    queries = []
    for text in tqdm(keyword):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_question=num_question
        )
        
        # Generate the response using the llamacpp model's generate or completion method
        response = llm.complete(
            prompt=query,
            max_tokens=1024,  
            stop=["\n"],     
            echo=False
        )

        # # if using gpt models
        # response = llm.generate(
        #     query,
        #     max_new_tokens=512
        # ) 
        # result = str(response.choices[0].message.content).strip().split("\n")
        # questions = [
        #     re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        # ]
        
        result = str(response).strip().split("\n") # separe each lines
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result if question.endswith("?")
        ] # select only questions
        questions = [question for question in questions if len(question) > 0][
            :num_question] #selct only the required number if needed

        for question in questions:
            queries.append(question)

    # construct dataset
    return queries