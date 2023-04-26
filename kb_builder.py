import argparse, json

parser = argparse.ArgumentParser(description='KG Construction.')
parser.add_argument('--data', default='./webnlg-dataset_v3.0/corpus-reader/train.json')
parser.add_argument('--save', default='kb.json')
parser.add_argument('--conf', default='llm.conf')
args = parser.parse_args()

# import the triples
from utils import get_triples_from_json

sent2triples, kb_triples = get_triples_from_json(args.data)
kb_triples = [
    (t[0], t[2], {'title': t[1]})
    for t in kb_triples
]

# build the networkx graph
import networkx as nx

g = nx.DiGraph()
g.add_edges_from(kb_triples)

# build the llm
from utils import get_llm
with open(args.conf, 'r') as f:
    conf = json.load(f)

llm_predictor, service_context = get_llm(conf['model'], conf['pipeline'])

# create the index with nodes consisting of all the
# triples relevant for a specific entity of the graph
from llama_index.data_structs.node_v2 import Node
from llama_index import GPTListIndex

nodes = []
for n in g.nodes():
    edges = list(g.in_edges(n, data=True)) + list(g.out_edges(n, data=True))
    edges = [
        '({}, {}, {})'.format(e[0], e[2]['title'], e[1])
        for e in edges
    ]
    nodes.append(Node(text='\n'.join(edges), doc_id=n))
kb_index = GPTListIndex(nodes=nodes, service_context=service_context)
kb_index.save_to_disk(args.save)

#import matplotlib.pyplot as plt
#nx.draw(g)
#plt.show()
