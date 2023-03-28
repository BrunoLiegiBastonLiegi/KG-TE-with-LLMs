import os
with open('openai_key.txt', 'r') as f:
    key = f.read()[:-1]
os.environ['OPENAI_API_KEY'] = key

import logging
import sys, json, argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from llama_index import SimpleDirectoryReader, LLMPredictor
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI

from llama_index import Document

parser = argparse.ArgumentParser(description='KG Construction.')
parser.add_argument('infiles', nargs='+')
parser.add_argument('--load_index')
args = parser.parse_args()

assert len(args.infiles) > 0

docs = []
for infile in args.infiles:
    with open(infile, 'r') as f:
        docs.append(Document(f.read()))

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

if args.load_index is not None:
    index = GPTKnowledgeGraphIndex.load_from_disk(args.load_index, llm_predictor=llm_predictor)
    print(f'> Loading Knowledge Graph from `{args.load_index}`.')
else:
    print('> Creating the Knowldge Graph.')
    # NOTE: can take a while! 
    index = GPTKnowledgeGraphIndex(
        docs, 
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

query = "Show me the triples extracted from the input documents"
print(f'\n\n-------> {query} :\n')
response = index.query(
    query, 
    include_text=False, 
    response_mode="tree_summarize"
)

print(response)
