"""
Useful functions for training and finetuning embeddings model
"""

import os
import re
import uuid 
import torch
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset
import numpy as np

# LlamaIndex imports
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_tuned_model(name):
    """
    Loads a pre-tuned model from the local cache directory.

    This function constructs the path to a locally stored pre-trained model, validates the input string,
    and loads the model using the HuggingFaceEmbedding class.

    Args:
        name (str): The name of the pre-tuned model to be loaded. The input must be a string prefixed with "local:",
                    followed by the model name.

    Raises:
        ValueError: If the input string does not start with the required prefix "local:".

    Returns:
        HuggingFaceEmbedding: An instance of the HuggingFaceEmbedding class initialized with the specified model
                              and cache directory.
    """
    
    embed_model_str = "local:" + name

    splits = embed_model_str.split(":", 1)
    is_local = splits[0]
    model_name = splits[1] if len(splits) > 1 else None
    if is_local != "local":
        raise ValueError(
            "embed_model must start with str 'local' or of type BaseEmbedding"
        )

    cache_folder = os.path.join(get_cache_dir(), "models")
    os.makedirs(cache_folder, exist_ok=True)

    embed_model = HuggingFaceEmbedding(
        model_name=model_name, cache_folder=cache_folder, trust_remote_code=True
    )
    return embed_model

DEFAULT_QA_GENERATE_PROMPT_TMPL =  """
You are an AI assistant specialized in classifying PDF documents into broad categories. Your task is to analyze the given text and provide a single tag that best describes the content or purpose of the document. Use the most appropriate tag from the following list, or create a new tag if necessary: education, government, science, health, legal, environment, finance, social-sciences, history, human-rights, community-development, humanitarian-aid, advocacy, sustainability, migration-and-refugees, social-justice, crisis, war. \

Respond with only the tag, nothing else. Here are some examples:"

1. "February 7, 1984 lebanon : the icrc calls for immediate ceasefire. Since the deterioration of the situation in lebanon over the last few days, the civilian population has sustained hundreds of victims, dead and wounded, particularly in the south of the capital, beirut."
Humanitarian Aid

2. "M. comelio sommaruga, président du comité international de la croix-rouge (cicr) donnera une conférence de presse le lundi 8 février 1988 à 10 heures 30* m. sommaruga présentera le bilan des principales actions conduites en 1987 ainsi que les objectifs et les grandes options de l'institution pour l'année en cours."
Advocacy

Now, please classify the following text:
{context_str}
"""

def clf_text_topic(
    texts,
    llm,
    qa_generate_prompt_tmpl = DEFAULT_QA_GENERATE_PROMPT_TMPL) : 
    """
    Generate and classify the topic of each chunk of text using a large language model (LLM).

    This function takes a list of texts, formats each text using a specified prompt template, 
    and sends the formatted prompt to the provided LLM for generating its topic. 

    Args:
        texts (list of str): A list of input text strings from which to generate questions.
        llm (object): A large language model (LLM) instance with a `generate` method to produce text completions.
        qa_generate_prompt_tmpl (str, optional): A prompt template for generating questions based on the input text. 
                                                 It must contain a placeholder `{context_str}` that will be replaced 
                                                 by each text. Defaults to `DEFAULT_QA_GENERATE_PROMPT_TMPL`.

    Returns:
        list of str: A list of generated questions, with one question extracted from each text.
    """

    queries = []
    for text in texts:
        query = qa_generate_prompt_tmpl.format(
            context_str=text)
        response = llm.generate(
            query,
            max_new_tokens=512
        )

        result = str(response.choices[0].message.content).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0][
            :1]

        for question in questions:
            queries.append(question)

    # construct dataset
    return queries
    
def group_by_topic(topic_pairs, doc_pairs):
    """
    Groups document pairs by their associated topics and assigns a unique identifier to each topic group.
    
    Args:
        topic_pairs (list of str): A list of topics, where each topic corresponds to a document pair.
        doc_pairs (list): A list of document pairs, where each entry is associated with a topic in `topic_pairs`.

    Returns:
        tuple:
            - queries (dict): A dictionary where each key is a unique question ID (UUID) and the value is the topic string.
            - relevant_docs (dict): A dictionary where each key is a question ID (same as in `queries`), and the value is 
                                    a list of document pairs corresponding to that topic.
    """

    group = {}
    for i, topic in enumerate(topic_pairs):
        if topic not in group:
            group[topic] = [doc_pairs[i]]
        else :
            group[topic].append(doc_pairs[i])

    queries = {}
    relevant_docs = {}

    for topic, doc_ids in group.items():
        question_id = str(uuid.uuid4())
        queries[question_id] = topic
        relevant_docs[question_id] = doc_ids
    return queries, relevant_docs

def build_df (topic_pairs, doc_pairs, corpus, model) :
    """
    Constructs a DataFrame containing topics, document pairs, and their corresponding embeddings,
    while filtering out specific topics based on starting words.


    Args:
        topic_pairs (list of str): A list of topics, where each topic corresponds to a document pair.
        doc_pairs (list): A list of document pairs associated with the topics.
        corpus (dict): A dictionary where each value is a text corresponding to a document.
        model (object): A model instance with an `encode` method to generate embeddings from the corpus texts.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - "topic": Topics corresponding to the documents.
            - "doc": Document pairs.
            - "corpus": Text content associated with the documents.
            - "doc_embeddings": Embeddings generated for the corresponding corpus text.
    """
    df = pd.DataFrame({"topic": topic_pairs, "doc": doc_pairs, "corpus": corpus.values()})
    embeddings = []
    for i in tqdm(range(len(df))):
        corpus = df.iloc[i]["corpus"]
        embeddings.append(model.encode(corpus))
    df["doc_embeddings"] = embeddings
    # remove line in dict if topic has not been found
    df = df[~df["topic"].str.startswith("This")]
    df = df[~df["topic"].str.startswith("Un")]
    df = df[~df["topic"].str.startswith("un")]
    df = df[~df["topic"].str.startswith("No")]
    df = df[~df["topic"].str.startswith("Cryptic")]
    return df

def train_xgb_model(data: pd.DataFrame, labels: pd.Series):
    """
    Trains an XGBoost classifier on the given data and labels, utilizing GPU if available.

    Args:
        data (pd.DataFrame): The input features for training.
        labels (pd.Series): The target labels for binary classification.

    Returns:
        xgb.XGBClassifier: The trained XGBoost classifier model.
    """
    if torch.cuda.is_available():
        boost_device = "cuda"
    else:
        boost_device = "cpu"

    # Initialize the XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                                device=boost_device,
                                random_state=3137,
                                enable_categorical=True)

    xgb_clf.fit(data, labels)

    return xgb_clf

class EmbeddingDataset(Dataset):
    """
    A PyTorch dataset that holds embeddings and their corresponding topics.

    Args:
        embeddings (list or numpy.ndarray): A list or array of precomputed embeddings.
        topics (list or numpy.ndarray): A list or array of topics corresponding to each embedding.

    Methods:
        __len__(): Returns the total number of items in the dataset.
        __getitem__(idx): Returns the embedding and the corresponding topic at the given index.

    """
    def __init__(self, embeddings, topics):
        self.embeddings = embeddings
        self.topics = topics

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.topics[idx]
    
class XGBoostWrapper(nn.Module):
    """
    A PyTorch wrapper for integrating an XGBoost model into a neural network.

    Args:
        xgboost_model (xgb.XGBClassifier): A pre-trained XGBoost classifier model.

    Methods:
        forward(x): Runs the forward pass by converting the input tensor to NumPy, 
                    predicts probabilities using the XGBoost model, and converts the 
                    predictions back to a PyTorch tensor.

    """
    def __init__(self, xgboost_model):
        super(XGBoostWrapper, self).__init__()
        self.xgboost_model = xgboost_model

    def forward(self, x):
        embeddings_np = x.cpu().detach().numpy()  # Ensure detach is only for NumPy operations

        pred_np = self.xgboost_model.predict_proba(pd.DataFrame(list([np.array(embeddings_np)])))

        pred_tensor = torch.tensor(pred_np, dtype=torch.float32).to(x.device)
        return pred_tensor

class FineTuningModel(nn.Module):
    """
    A neural network model that combines an embedding model with an XGBoost classifier for fine-tuning.

    Args:
        embedding_model (nn.Module): A pre-trained embedding model that encodes input data into embeddings.
        xgboost_wrapper (XGBoostWrapper): A wrapper for the pre-trained XGBoost model.

    Methods:
        forward(input_data): Encodes the input data into embeddings and then predicts topics using the XGBoost model.

    """
    def __init__(self, embedding_model, xgboost_wrapper):
        super(FineTuningModel, self).__init__()
        self.embedding_model = embedding_model
        self.xgboost_wrapper = xgboost_wrapper

    def forward(self, input_data):
        embeddings = self.embedding_model.encode(input_data[0], convert_to_tensor=True)
        embeddings.requires_grad_()
        topic_prediction = self.xgboost_wrapper(embeddings)
        return topic_prediction

def train_model(embedding_model, xgboost_model, dataset, num_epochs=10, learning_rate=1e-4):
    """
    Trains the combined model (embedding model + XGBoost classifier) on the given dataset.

    Args:
        embedding_model (nn.Module): A pre-trained model that converts input text to embeddings.
        xgboost_model (xgb.XGBClassifier): A pre-trained XGBoost classifier for topic prediction.
        dataset (EmbeddingDataset): A PyTorch dataset containing embeddings and topics.
        num_epochs (int, optional): Number of training epochs. Default is 10.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.

    Returns:
        None: Trains the model and prints the loss after each epoch.
    """
    # Wrap the XGBoost model
    xgboost_wrapper = XGBoostWrapper(xgboost_model)

    # Create the full model
    model = FineTuningModel(embedding_model, xgboost_wrapper)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.embedding_model.parameters(), lr=learning_rate)
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for text, topics in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(text)
            
            outputs.requires_grad_()
            # topics.requires_grad_()
            loss = criterion(outputs, topics.long().to(outputs.device))
            
            # Backward pass and optimization
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')