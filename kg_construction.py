import os
with open('openai_key.txt', 'r') as f:
    key = f.read()[:-1]
os.environ['OPENAI_API_KEY'] = key

import logging
import sys, json, argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from llama_index import LLMPredictor
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index import Document

parser = argparse.ArgumentParser(description='KG Construction.')
parser.add_argument('infiles', nargs='+')
parser.add_argument('--load_index')
args = parser.parse_args()

assert len(args.infiles) > 0

docs = []
nodes = []
for i,infile in enumerate(args.infiles):
    with open(infile, 'r') as f:
        chunks = f.read().split('\n')
        chunks = [ Node(text=c, doc_id=str(i)) for c in chunks if len(c) > 0 ]
        for c in chunks:
            nodes.append(c)
        docs.append(Document(f.read()))

from llm import CustomLLM
import torch
from transformers import pipeline as Pipeline

#model = "facebook/opt-iml-max-30b"
#model = "google/t5-v1_1-base"
#model = "EleutherAI/gpt-j-6B"
model = "gpt2"
pipeline = "text-generation"
#pipeline = "text2text-generation"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pipeline = Pipeline(
    pipeline,
    model=model,
    device=device,
    model_kwargs={"torch_dtype":torch.bfloat16}
)
print(f'> Using device: {device}.')
print(f' # Model: {model}\n # Pipeline: {pipeline}')

# define LLM
llm_predictor = LLMPredictor(
    llm = CustomLLM(
        model = model,
        pipeline = pipeline,
        out_tokens = 256
    )
)

print('---------------')

if args.load_index is not None:
    index = GPTKnowledgeGraphIndex.load_from_disk(args.load_index, llm_predictor=llm_predictor)
    print(f'> Loading Knowledge Graph from `{args.load_index}`.')
else:
    print('> Creating the Knowldge Graph.')
    # NOTE: can take a while! 
    index = GPTKnowledgeGraphIndex(
        #docs,
        nodes=nodes,
        chunk_size_limit=512, 
        max_triplets_per_chunk=7,
        llm_predictor=llm_predictor,
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
