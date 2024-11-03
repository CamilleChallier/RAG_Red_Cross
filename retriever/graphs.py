import json 
import re
import asyncio
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.async_utils import run_jobs
# Transformers and models
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Callable, Optional, Union

import nest_asyncio
# Apply asyncio modifications
nest_asyncio.apply()

### Triplex 
def triplextract(model, tokenizer, text, entity_types, predicates):
    """
    Performs Named Entity Recognition (NER) and extracts knowledge graph triplets from the provided text.
    
    Parameters:
        model (transformers.PreTrainedModel): The pre-trained model used for generating responses.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to preprocess input text and format the output.
        text (str): The input text to process.
        entity_types (list): A list of entity types to identify.
        predicates (list): A list of predicates to use for extracting relationships.

    Returns:
        str: The generated output containing identified entities and triplets in a formatted response.
    """
    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """
    
    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                predicates = json.dumps({"predicates": predicates}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt").to("cuda")
    output = tokenizer.decode(model.generate(input_ids=input_ids, max_length=1500)[0], skip_special_tokens=True)
    return output

def extract_json_dict(input_str):
    """
    Extracts a JSON dictionary from a formatted string containing JSON content.
    
    The function looks for JSON content between specific delimiters and ensures the JSON structure is valid.
    
    Parameters:
        input_str (str): The input string containing JSON content.

    Returns:
        dict: The extracted JSON dictionary, or None if the extraction fails.
    """
    
    # Find the start and end positions of the JSON content
    start_pos = input_str.find("```json")
    
    end_pos = input_str.find("```", start_pos + 6)  # Start searching after the first delimiter
    
    if start_pos == -1 : 
        return None
    
    if end_pos == -1:
        # cut str to last ","
        input_str =  input_str[:input_str.rfind('",')+1] + "]} ```"
        end_pos = -3
        if len(input_str) < start_pos :
            return None

    # Extract the JSON content
    json_str = input_str[start_pos+7:end_pos].strip()
    
    # verify "entities_and_triples" only one time in the json
    if json_str.count("entities_and_triples") > 1:
        json_str =  json_str[:json_str.rfind('"entities_and_triples"')] 
        if not json_str.endwith("}"):
            json_str[:json_str.rfind('",')] + "]}"
    
    
    # Convert the JSON string to a dictionary
    try:
        json_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("oups")
        return None

    return json_dict

def extract_and_format_triplets(lines):
    """
    Extracts and formats knowledge graph triplets from a list of lines.
    
    Each line is checked for a specific format that indicates a triplet, which is then extracted and formatted.
    
    Parameters:
        lines (list of str): A list of lines, each potentially containing a triplet.

    Returns:
        list of tuples: A list of triplets formatted as (entity1, predicate, entity2). The list is truncated to a maximum of 5 triplets.
    """
    triplets = []

    # Extract and format triplets
    for line in lines:
        # Check if the line is a triplet
        match = re.match(r'\[(\d+)\]\s+([A-Z_]+)\s+\[(\d+)\]', line)
        if match:
            entity1_id = match.group(1)
            predicate = match.group(2).replace('_', ' ').title()
            entity2_id = match.group(3)
            
            # Find the entities corresponding to the IDs
            entity1 = next((e.split(':')[1].strip() for e in lines if e.startswith(f'[{entity1_id}],')), None)
            entity2 = next((e.split(':')[1].strip() for e in lines if e.startswith(f'[{entity2_id}],')), None)
            
            if entity1 and entity2:
                triplets.append((entity1, predicate, entity2))

    if len(triplets) > 5 :
        triplets = triplets[:5]
    return triplets

def triplets_extraction(text):
    
    entity_types = [ "LOCATION", "DATE", "ORGANIZATION", "PERSON", "NUMBER", "EVENT", "SERVICE"]
    predicates = [
        "PROVIDED_AID", "LOCATED_IN", "HAPPENED_ON", "SUFFERED" "PARTNERED_WITH", "PROVIDED", "AFFECTED_BY", "ASKED"
    ]
    model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)
    prediction = triplextract(model, tokenizer, text, entity_types, predicates)
    triplets_dict = extract_json_dict(prediction)
    if triplets_dict == None :
        return ()
    triplets = extract_and_format_triplets(triplets_dict["entities_and_triples"])

    return triplets

### GraphRAG
# implementation https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/GraphRAG_v1.ipynb
class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata[
                "entity_description"
            ] = description  # Not used in the current implementation. But will be useful in future work.
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=metadata
            )
            existing_nodes.append(entity_node)

        metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, rel, obj, description = triple
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = OpenAI().chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

def parse_fn(response_str: str) -> Any:
    entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
    relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

### Graph RAG
from openai import OpenAI
import networkx as nx
from cdlib import algorithms
import os
from dotenv import load_dotenv

load_dotenv()

# 2. Text Chunks → Element Instances
def extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        print(f"Chunk index {index} of {len(chunks)}:")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract entities and relationships from the following text."},
                {"role": "user", "content": chunk.text}
            ]
        )
        entities_and_relations = response.choices[0].message.content
        elements.append(entities_and_relations)
    return elements


# 3. Element Instances → Element Summaries
def summarize_elements(elements):
    summaries = []
    for index, element in enumerate(elements):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use \"->\" to represent relationships, after the \"Relationships:\" word."},
                {"role": "user", "content": element}
            ]
        )
        summary = response.choices[0].message.content
        summaries.append(summary)
    return summaries


# 4. Element Summaries → Graph Communities
def build_graph_from_summaries(summaries):
    G = nx.Graph()
    for index, summary in enumerate(summaries):
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                if line[0].isdigit() and line[1] == ".":
                    line = line.split(".", 1)[1].strip()
                entity = line.strip()
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    return G


# 5. Graph Communities → Community Summaries
def detect_communities(graph):
    communities = []
    index = 0
    for component in nx.connected_components(graph):
        print(
            f"Component index {index} of {len(list(nx.connected_components(graph)))}:")
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:  # Leiden algorithm requires at least 2 nodes
            try:
                sub_communities = algorithms.leiden(subgraph)
                for community in sub_communities.communities:
                    communities.append(list(community))
            except Exception as e:
                print(f"Error processing community {index}: {e}")
        else:
            communities.append(list(subgraph.nodes))
        index += 1
    print("Communities from detect_communities:", communities)
    return communities


def summarize_communities(communities, graph):
    community_summaries = []
    for index, community in enumerate(communities):
        print(f"Summarize Community index {index} of {len(communities)}:")
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following community of entities and relationships."},
                {"role": "user", "content": description}
            ]
        )
        summary = response.choices[0].message.content.strip()
        community_summaries.append(summary)
    return community_summaries


# 6. Community Summaries → Community Answers → Global Answer
def generate_answers_from_communities(community_summaries, query):
    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Summary index {index} of {len(community_summaries)}:")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer the following query based on the provided summary."},
                {"role": "user", "content": f"Query: {query} Summary: {summary}"}
            ]
        )
        print("Intermediate answer:", response.choices[0].message.content)
        intermediate_answers.append(
            response.choices[0].message.content)

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
                "content": "Combine these answers into a final, concise response."},
            {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
        ]
    )
    final_answer = final_response.choices[0].message.content
    return final_answer


# Putting It All Together
def graph_rag_pipeline(nodes, query):
    # Step 1: Split documents into chunks

    # Step 2: Extract elements from chunks
    elements = extract_elements_from_chunks(nodes)

    # Step 3: Summarize elements
    summaries = summarize_elements(elements)

    # Step 4: Build graph and detect communities
    graph = build_graph_from_summaries(summaries)
    print("graph:", graph)
    communities = detect_communities(graph)

    print("communities:", communities[0])
    # Step 5: Summarize communities
    community_summaries = summarize_communities(communities, graph)

    # Step 6: Generate answers from community summaries
    final_answer = generate_answers_from_communities(
        community_summaries, query)

    return final_answer
