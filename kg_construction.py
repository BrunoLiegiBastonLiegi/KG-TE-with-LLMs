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
parser.add_argument('--kb', default='./webnlg-dataset_v3.0/corpus-reader/train.json')
parser.add_argument('--load_index')
parser.add_argument('--prompt')
args = parser.parse_args()

# load documents
if args.load_index is None:
    assert len(args.infiles) > 0

    docs = []
    nodes = []
    for i,infile in enumerate(args.infiles):
        with open(infile, 'r') as f:
            chunks = f.read().split('\n')
            chunks = [ Node(text=c, doc_id=str(i)+'-'+str(j)) for j,c in enumerate(chunks) if len(c) > 0 ]
            for c in chunks:
                nodes.append(c)
            docs.append(Document(f.read()))

# build the knowledge base
from utils import get_triples_from_json

sent2triples, kb_triples = get_triples_from_json(args.kb)

import torch
from transformers import pipeline as Pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from llama_index.indices.service_context import ServiceContext

#model = "facebook/opt-iml-max-30b"
#model_id, pipeline = "google/t5-v1_1-base", "text2text-generation"
#model_id, pipeline = "EleutherAI/gpt-j-6B", "text-generation"
model_id, pipeline = "EleutherAI/gpt-neo-1.3B", "text-generation"
#model_id, pipeline = "gpt2", "text-generation"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, revision="float16")
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

from llama_index.prompts.prompts import KnowledgeGraphPrompt

if args.prompt is None:
    kg_extraction_template =  (
        "Some text is provided below. Given the text, extract up to "
        "{max_knowledge_triplets} "
        "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
        "---------------------\n"
        "Example:"
        "Text: Alice is Bob's mother."
        "Triplets:\n(Alice, is mother of, Bob)\n"
        "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
        "Triplets:\n"
        "(Philz, is, coffee shop)\n"
        "(Philz, founded in, Berkeley)\n"
        "(Philz, founded in, 1982)\n"
        "---------------------\n"
        "Text: {text}\n"
        "Triplets:\n"
    )
else:
    with open(args.prompt, 'r') as f:
        kg_extraction_template =  f.read()

print(f'> Prompt:\n{kg_extraction_template}')

prompt = KnowledgeGraphPrompt(
    kg_extraction_template
)

max_new_tokens = 64

from accelerate import infer_auto_device_map
dev_map = infer_auto_device_map(model)
print(f'> Device Map:\n{dev_map}')

pipe = Pipeline(
    pipeline,
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_new_tokens,
    device_map='auto', # used for distributed
    model_kwargs={"torch_dtype":torch.bfloat16}
)

llm = HuggingFacePipeline(pipeline=pipe)


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class CustomLLM(LLM):
    model_name = model_id
    pipeline = Pipeline(
        pipeline,
        model=model_name,
        device_map='auto',
        model_kwargs={"torch_dtype":torch.bfloat16}
    )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

    
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


print(f' # Model: {model_id}\n # Pipeline: {pipeline}\n # Max New Tokens: {max_new_tokens}')

# define LLM
llm_predictor = LLMPredictor(
    llm = llm
    #llm = CustomLLM()
)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

# Build the Knowledge Base
# initialize an empty index for now 
kb_index = GPTKnowledgeGraphIndex(
    [],
    service_context=service_context,
)



print('--------------------------------------------------------------')

if args.load_index is not None:
    print(f'> Loading Knowledge Graph from `{args.load_index}`.')
    index = GPTKnowledgeGraphIndex.load_from_disk(
        args.load_index,
        service_context=service_context
    )
else:
    print('> Creating the Knowldge Graph.')
    nodes = nodes[:10]
    print('> Input text:')
    for n in nodes:
        print(n.text)
    
    index = GPTKnowledgeGraphIndex(
        nodes=nodes,
        kg_triple_extract_template=prompt,
        max_triplets_per_chunk=7,
        service_context=service_context,
        #include_embeddings=True
    )
    """
    index = GPTKnowledgeGraphIndex.from_documents(
        docs,
        kg_triple_extract_template=prompt,
        max_triplets_per_chunk=7,
        service_context=service_context
    )
    """
    index.save_to_disk('index_kg.json')

print('\n---------- Visualize the extracted KG ----------\n')
import networkx as nx
#import matplotlib.pyplot as plt
from pyvis.network import Network

g = index.get_networkx_graph()
for e in g.edges(data=True):
    print(e)
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example.html")

query = "Who is the leader of Aarhus Airport?"
print(f'\n\n-------> {query} :\n')
response = index.query(
    query, 
    #include_text=False, 
    #response_mode="tree_summarize"
)

print(response)
