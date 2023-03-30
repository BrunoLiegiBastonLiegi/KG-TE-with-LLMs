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
from llama_index.data_structs.node_v2 import Node


parser = argparse.ArgumentParser(description='KG Construction.')
parser.add_argument('infiles', nargs='+')
parser.add_argument('--load_index')
args = parser.parse_args()

if args.load_index is None:
    assert len(args.infiles) > 0

    #docs = []
    nodes = []
    for i,infile in enumerate(args.infiles):
        with open(infile, 'r') as f:
            chunks = f.read().split('\n')
            chunks = [ Node(text=c, doc_id=str(i)+'-'+str(j)) for j,c in enumerate(chunks) if len(c) > 0 ]
            for c in chunks:
                nodes.append(c)
                #docs.append(Document(f.read()))

#from llm import CustomLLM
import torch
from transformers import pipeline as Pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.indices.service_context import ServiceContext


if torch.cuda.is_available():
    device = torch.device('cuda:0')    
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
print(f'> Using device: {device}.')

#model = "facebook/opt-iml-max-30b"
#model = "google/t5-v1_1-base"
#model = "EleutherAI/gpt-j-6B"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipeline = "text-generation"
#pipeline = "text2text-generation"

pipe = Pipeline(
    pipeline,
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    device=device,
    model_kwargs={"torch_dtype":torch.bfloat16}
)

llm = HuggingFacePipeline(pipeline=pipe)

print(f' # Model: {model_id}\n # Pipeline: {pipeline}')

# define LLM
llm_predictor = LLMPredictor(
    llm = llm
)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

print('---------------')

if args.load_index is not None:
    print(f'> Loading Knowledge Graph from `{args.load_index}`.')
    index = GPTKnowledgeGraphIndex.load_from_disk(args.load_index, service_context=service_context)
else:
    print('> Creating the Knowldge Graph.')
    # NOTE: can take a while! 
    index = GPTKnowledgeGraphIndex(
        #docs,
        nodes=nodes,
        max_triplets_per_chunk=7,
        service_context=service_context
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

query = "where is Karnataka located?"
print(f'\n\n-------> {query} :\n')
response = index.query(
    query, 
    include_text=False, 
    response_mode="tree_summarize"
)

print(response)
