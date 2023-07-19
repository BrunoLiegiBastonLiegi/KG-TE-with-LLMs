import sys, json, time
sys.path.append('./webnlg-dataset_v3.0/corpus-reader/')

import argparse

from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.prompts.prompts import KnowledgeGraphPrompt
#from llama_index.data_structs.node import Node
from llama_index.schema import TextNode as Node


from tqdm import tqdm

import xml.etree.cElementTree as ET

from utils import get_llm, load_kb, get_triplet_extraction_prompt, extract_triples, normalize_triple, get_data_loader

def main(path_to_corpus):

    root = ET.Element("benchmark")
    entries = ET.SubElement(root, "entries")

    # define index for triplet extraction 
    index = GPTKnowledgeGraphIndex(
        nodes=[],
        max_triplets_per_chunk=max_triplets,
        service_context=service_context,
    )
    
    print('> Extracting triples from corpus')
    data = get_data_loader(path_to_corpus)
    for i, (sentence, triples) in tqdm(enumerate(data), total=len(data)):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        try:
            cat = entry.category
        except:
            cat = ''#None
            print("> WARNING: couldn't find entry category.")
        try:
            eid = entry.id
        except:
            eid = str(i)
            print("> WARNING: couldn't find entry id.")
            
        e = ET.SubElement(
            entries,
            "entry",
            category=cat,
            eid=eid
        )
        tripleset = ET.SubElement(e, "generatedtripleset")
        print('> Processing Entry')
        t = time.time()
        #triples = get_triples([entry.lexs[0]])
        #sentences = [entry.lexs[0].lex]
        print(f'  > {sentence}\n')
        if args.groundtruth:
            if dataset == 'nyt':
                triples = [ (t[0], t[1].split('/')[-1], t[2]) for t in triples ]
            triples = [ normalize_triple(triple) for triple in triples ]
        else:
            triples = extract_triples(
                sentences=sentence,
                prompt=prompt,
                kg_index=index,
                #max_knowledge_triplets=max_triplets,
                kb_retriever=kb_retriever
            )
        triples = [ f"{t[0]} | {t[1]} | {t[2]}" for t in triples ]
        print(f"\n----> Extracted Tripets: {triples}\n")
        print(f'  > Processed sentence in {time.time()-t:.4f}s\n')
        for triple in triples:
            ET.SubElement(tripleset, "gtriple").text = triple

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    import lxml.etree as etree
    global model_id
    model_id = model_id.replace('/','-')
    if args.groundtruth:
        save_name = f"{dataset}/groundtruth_triples"
    else:
        save_name = f"{dataset}/generated_triples_{model_id}"
        if args.kb is not None:
            save_name += f"_kb-top-{args.top_k}"
    save_name += '.xml'
    tree.write(save_name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmarking triplet extraction.')
    parser.add_argument('--data')
    parser.add_argument('--prompt')
    parser.add_argument('--conf', default='llm.conf')
    parser.add_argument('--kb')
    parser.add_argument('--top_k', default=2, type=int)
    parser.add_argument('--groundtruth', action='store_true')
    
    args = parser.parse_args()

    dataset = args.data.split('/')[0]
    
    if args.prompt is not None:
        with open(args.prompt, 'r') as f:
            prompt = json.load(f)
    else:
        prompt = None
        
    # prepare the llm
    with open(args.conf, 'r') as f:
        conf = json.load(f)

    model_id, pipeline = conf.pop('model'), conf.pop('pipeline')
    llm_predictor, service_context = get_llm(model_id, pipeline, **conf)
    max_triplets = 7
            
    # prepare the kb
    if args.kb is not None:
        kb_index, kb_retriever = load_kb(args.kb, service_context, similarity_top_k=args.top_k)
    else:
        kb_index, kb_retriever = None, None
    main(args.data)
