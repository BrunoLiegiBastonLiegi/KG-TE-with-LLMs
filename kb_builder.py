import argparse, json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='KG Construction.')
parser.add_argument('--data', nargs='+')
parser.add_argument('--save')
parser.add_argument('--conf', default='llm.conf')
args = parser.parse_args()

dataset_dir = args.data[0].split('/')[0]
if args.save is None:
    args.save = f"{dataset_dir}/kb"

# import the triples
from utils import get_triples_from_file

sent2triples, kb_triples = get_triples_from_file(infiles=args.data, dataset=dataset_dir)
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
from llama_index.data_structs.node import Node, DocumentRelationship
from llama_index import GPTListIndex
from llama_index import GPTVectorStoreIndex


nodes, id2embedding = [], {}
for n in tqdm(list(g.nodes())):
    edges = list(g.in_edges(n, data=True)) + list(g.out_edges(n, data=True))
    edges = [
        '({}, {}, {})'.format(e[0], e[2]['title'], e[1])
        for e in edges
    ]
    text = '\n'.join(edges)
    node = Node(
        text=text,
        doc_id=n,
    )
    id2embedding[n] = service_context.embed_model._get_text_embedding(text)
    nodes.append(node)

for i,n in enumerate(nodes):
    n.relationships[DocumentRelationship.SOURCE] = n.get_doc_id()
    #if i < len(nodes) - 1:
    #    n.relationships[DocumentRelationship.NEXT] = nodes[i+1].get_doc_id()
    #if i > 0:
    #    n.relationships[DocumentRelationship.PREVIOUS] = nodes[i-1].get_doc_id()

kb_index = GPTVectorStoreIndex(
    nodes=nodes,
    service_context=service_context,
)




# test retrieval
from llama_index.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=kb_index,
    similarity_top_k=5,
)

r = retriever.retrieve("Linate Airport is located in Milan, Italy.")
triples = [ n.node.text for n in r ]
triples = '\n'.join(triples)
print(triples)

kb_index.storage_context.persist(args.save)

#import matplotlib.pyplot as plt
#nx.draw(g)
#plt.show()
