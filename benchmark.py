import sys, json, time
sys.path.append('./webnlg-dataset_v3.0/corpus-reader/')

import argparse

from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.prompts.prompts import KnowledgeGraphPrompt
from llama_index.data_structs.node_v2 import Node

from tqdm import tqdm


parser = argparse.ArgumentParser(description='WebNLG Benchmark.')
parser.add_argument('--prompt')
parser.add_argument('--conf', default='llm.conf')

args = parser.parse_args()

if args.prompt is None:
    prompt = None
else:
    with open(args.prompt, 'r') as f:
        kg_extraction_template =  f.read()
        prompt = KnowledgeGraphPrompt(
            kg_extraction_template
        )
    
# prepare the llm
from utils import get_llm

with open(args.conf, 'r') as f:
    conf = json.load(f)

llm_predictor, service_context = get_llm(conf['model'], conf['pipeline'])

def main():

    from benchmark_reader import Benchmark
    from benchmark_reader import select_files

    path_to_corpus = 'webnlg-dataset_v3.0/en/dev/'
    print(f'> Using corpus: {path_to_corpus}')
    
    # initialise Benchmark object
    b = Benchmark()

    # collect xml files
    files = select_files(path_to_corpus)

    # load files to Benchmark
    b.fill_benchmark(files)

    # output some statistics
    #print("Number of entries: ", b.entry_count())
    #print("Number of texts: ", b.total_lexcount())
    #print("Number of distinct properties: ", len(list(b.unique_p_mtriples())))

    import xml.etree.cElementTree as ET
    root = ET.Element("benchmark")
    entries = ET.SubElement(root, "entries")

    print('> Extracting triples from corpus')
    # get access to each entry info
    for entry in tqdm(b.entries):
        e = ET.SubElement(
            entries,
            "entry",
            category=entry.category,
            eid=entry.id
        )
        tripleset = ET.SubElement(e, "generatedtripleset")
        print('> Processing Entry')
        t = time.time()
        triples = get_triples(entry.lexs)
        print(f'  > Processed {len(entry.lexs)} sentences in {time.time()-t:.4f}s ')
        for triple in triples:
            ET.SubElement(tripleset, "gtriple").text = triple

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write("generated_triples.xml")
        
def get_triples(lexs):
    nodes = [Node(lex.lex) for lex in lexs]
    index = GPTKnowledgeGraphIndex(
        nodes=nodes,
        kg_triple_extract_template=prompt,
        max_triplets_per_chunk=10,
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
