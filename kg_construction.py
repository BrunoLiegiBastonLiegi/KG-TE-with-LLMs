import os
with open('openai_key.txt', 'r') as f:
    key = f.read()[:-1]
os.environ['OPENAI_API_KEY'] = key

import logging
import sys, json

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from llama_index import SimpleDirectoryReader, LLMPredictor
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI

from llama_index import download_loader

print('> Downloading Wikipedia pages.')
WikipediaReader = download_loader("WikipediaReader")
wiki_docs = WikipediaReader().load_data(pages=['Toronto', 'Berlin', 'Tokyo'])

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

try:
    index = GPTKnowledgeGraphIndex.load_from_disk('index_kg.json', llm_predictor=llm_predictor)
    print('> Loading Knowledge Graph from `index_kg.json`.')
except:
    print('> Creating the Knowldge Graph.')
    # NOTE: can take a while! 
    index = GPTKnowledgeGraphIndex(
        wiki_docs, 
        chunk_size_limit=512, 
        max_triplets_per_chunk=2,
        llm_predictor=llm_predictor
    )
    index.save_to_disk('index_kg.json')

"""
query = "Tell me about Berlin"

print(f'\n\n-------> {query} :\n')
response = index.query(
    query, 
    include_text=False, 
    response_mode="tree_summarize"
)
print(response)
"""

query = "Show me the Knowledge Graph about Berlin"
print(f'\n\n-------> {query} :\n')
response = index.query(
    query, 
    include_text=False, 
    response_mode="tree_summarize"
)

print(response)
