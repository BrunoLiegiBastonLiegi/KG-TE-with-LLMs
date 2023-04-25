import sys
sys.path.append('./webnlg-dataset_v3.0/corpus-reader/')

import torch, argparse
from transformers import pipeline as Pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from llama_index.indices.service_context import ServiceContext
from llama_index.data_structs.node_v2 import Node
from llama_index import LLMPredictor
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.prompts.prompts import KnowledgeGraphPrompt

from tqdm import tqdm


parser = argparse.ArgumentParser(description='WebNLG Benchmark.')
parser.add_argument('--prompt')

if args.prompt is None:
    prompt = None
else:
    with open(args.prompt, 'r') as f:
        kg_extraction_template =  f.read()
        prompt = KnowledgeGraphPrompt(
            kg_extraction_template
        )
        
model_id, pipeline = "gpt2", "text-generation"
#model_id, pipeline = "decapoda-research/llama-7b-hf", "text-generation"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

from accelerate import infer_auto_device_map
dev_map = infer_auto_device_map(model)
print(f'> Device Map:\n{dev_map}')

pipe = Pipeline(
    pipeline,
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    device_map='auto', # used for distributed
    model_kwargs={"torch_dtype":torch.bfloat16}
)
llm = HuggingFacePipeline(pipeline=pipe)
llm_predictor = LLMPredictor(llm = llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)


def main():

    from benchmark_reader import Benchmark
    from benchmark_reader import select_files

    path_to_corpus = 'webnlg-dataset_v3.0/en/dev/'

    # initialise Benchmark object
    b = Benchmark()

    # collect xml files
    files = select_files(path_to_corpus)

    # load files to Benchmark
    b.fill_benchmark(files)

    # output some statistics
    print("Number of entries: ", b.entry_count())
    print("Number of texts: ", b.total_lexcount())
    print("Number of distinct properties: ", len(list(b.unique_p_mtriples())))

    import xml.etree.cElementTree as ET
    root = ET.Element("benchmark")
    entries = ET.SubElement(root, "entries")

    # get access to each entry info
    for entry in tqdm(b.entries[:10]):
        e = ET.SubElement(
            entries,
            "entry",
            category=entry.category,
            eid=entry.id
        )
        tripleset = ET.SubElement(e, "generatedtripleset")
        triples = get_triples(entry.lexs)
        for triple in triples:
            ET.SubElement(tripleset, "gtriple").text = triple

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write("generated_triples.xml")
        
def get_triples(lexs):
    index = GPTKnowledgeGraphIndex(
        nodes=[Node(lex) for lex in lexs],
        kg_triple_extract_template=prompt,
        max_triplets_per_chunk=7,
        service_context=service_context,
    )
    g = index.get_networkx_graph()
    triples = [
        '{} | {} | {}'.format(e[0], e[2]['title'], e[1])
        for e in g.edges(data=True)
    ]
    return triples
    


if __name__ == '__main__':
    main()
