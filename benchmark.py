import sys, json, time
sys.path.append('./webnlg-dataset_v3.0/corpus-reader/')

import argparse

from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.prompts.prompts import KnowledgeGraphPrompt
from llama_index.data_structs.node import Node

from tqdm import tqdm

from utils import get_llm, load_kb, get_triplet_extraction_prompt, extract_triples, normalize_triple

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

    import xml.etree.cElementTree as ET
    root = ET.Element("benchmark")
    entries = ET.SubElement(root, "entries")

    # define index for triplet extraction 
    index = GPTKnowledgeGraphIndex(
        nodes=[],
        max_triplets_per_chunk=max_triplets,
        service_context=service_context,
    )
    
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
        #triples = get_triples([entry.lexs[0]])
        sentences = [entry.lexs[0].lex]
        print(f'  > {sentences[0]}')
        if args.groundtruth:
            triples = []
            for triple in entry.list_triples():
                triples.append(normalize_triple(triple))
        else:
            triples = extract_triples(
                sentences=sentences,
                prompt=prompt,
                kg_index=index,
                #max_knowledge_triplets=max_triplets,
                kb_retriever=kb_retriever
            )
            triples = [ f"{t[0]} | {t[1]} | {t[2]}" for t in triples ]
        print(f"----> Extracted Tripets: {triples}")
        print(f'  > Processed sentence in {time.time()-t:.4f}s ')
        for triple in triples:
            ET.SubElement(tripleset, "gtriple").text = triple

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write("generated_triples.xml")
        
def get_triples(lexs):
    global prompt
    nodes = [Node(lex.lex) for lex in lexs]
    if kb_index is not None:
        index = GPTKnowledgeGraphIndex(
            nodes=[],
            max_triplets_per_chunk=10,
            service_context=service_context,
        )
        triples = set()
        for n in nodes:
            print(n.text)
            #prompt = get_triplet_extraction_prompt(body, examples, n.text, kb_retriever)
            #print(prompt.prompt.template)
            index.kg_triple_extract_template = prompt.partial_format(max_knowledge_triplets=max_knowledge_triplets)
            for t in index._extract_triplets(n.text):
                triples.add(t)
    else:
        index = GPTKnowledgeGraphIndex(
            nodes=nodes,
            kg_triple_extract_template=prompt,
            max_triplets_per_chunk=10,
            service_context=service_context,
        )
        g = index.get_networkx_graph()
        triples = list(g.edges(data=True))
        print(index._extract_triplets(nodes[0].text))
    print(triples)
    #assert False
    triples = [
        '{} | {} | {}'.format(e[0], e[1], e[2])
        for e in triples
    ]
    return triples
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='WebNLG Benchmark.')
    parser.add_argument('--prompt')
    parser.add_argument('--conf', default='llm.conf')
    parser.add_argument('--kb')
    parser.add_argument('--groundtruth', action='store_true')
    
    args = parser.parse_args()
    
    if args.prompt is not None:
        with open(args.prompt, 'r') as f:
            prompt = json.load(f)
    else:
        prompt = None
        
    # prepare the llm
    with open(args.conf, 'r') as f:
        conf = json.load(f)
            
    llm_predictor, service_context = get_llm(conf['model'], conf['pipeline'])
    max_triplets = 7
            
    # prepare the kb
    if args.kb is not None:
        kb_index, kb_retriever = load_kb(args.kb, service_context, similarity_top_k=2)
    else:
        kb_index, kb_retriever = None, None
    main()
