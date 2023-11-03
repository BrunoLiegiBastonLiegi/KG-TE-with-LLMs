import argparse, json, random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='KB Construction.')
parser.add_argument('--data', nargs='+')
parser.add_argument('--save')
parser.add_argument('--conf', default='model_conf/gpt2.conf')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--no_overlap', action='store_true')
parser.add_argument('--scale', default=1., type=float)
args = parser.parse_args()

dataset_dir = args.data[0].split('/')[0]
if args.save is None:
    args.save = f"{dataset_dir}/kb"

splits = [ d.split('/')[-1][:-5] for d in args.data ]
complete = True if 'test' in splits else False

# import the triples
from utils import get_data_from_files, normalize_triple

_, test_triples = get_data_from_files(infiles=f"{dataset_dir}/test.json")

sent2triples, kb_triples = get_data_from_files(infiles=args.data)
if args.normalize:
    # get rid of all the specifications location/location/contains etc...
    if dataset_dir == 'nyt':
        kb_triples = [ (t[0], t[1].split('/')[-1], t[2]) for t in kb_triples ]
    kb_triples = set([ tuple(normalize_triple(t)) for t in kb_triples ])
    test_triples = set([ tuple(normalize_triple(t)) for t in test_triples ])
    for sent, triples in sent2triples.items():
        if dataset_dir == 'nyt':
            triples = [ (t[0], t[1].split('/')[-1], t[2]) for t in triples ]
        triples = set([ tuple(normalize_triple(t)) for t in triples ])
        sent2triples[sent] = triples

n = int(args.scale * len(kb_triples))
if n != len(kb_triples):
    print(f'> Sampling {n} triples out of the original {len(kb_triples)}')
    kb_triples = random.sample(list(kb_triples), n)

print(len(kb_triples))
if args.no_overlap:
    kb_triples = [t for t in kb_triples if t not in test_triples]
print(len(kb_triples))

edges = [
    (t[0], t[2], {'title': t[1]})
    for t in kb_triples
]

# build the networkx graph
import networkx as nx

g = nx.DiGraph()
g.add_edges_from(edges)

# build the llm
from utils import get_llm
with open(args.conf, 'r') as f:
    conf = json.load(f)

model_id, pipeline = conf.pop('model'), conf.pop('pipeline')
llm_predictor, service_context = get_llm(model_id, pipeline, **conf)

# create the index with nodes consisting of all the
# triples relevant for a specific entity of the graph
#from llama_index.data_structs.node import Node, DocumentRelationship
from llama_index.schema import TextNode as Node
from llama_index import GPTListIndex
from llama_index import GPTVectorStoreIndex


# kb index with each node composed of in+out edges of the graph node

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
    #id2embedding[n] = service_context.embed_model._get_text_embedding(text)
    nodes.append(node)

#for i,n in enumerate(nodes):
    #n.relationships[DocumentRelationship.SOURCE] = n.get_doc_id()
    #if i < len(nodes) - 1:
    #    n.relationships[DocumentRelationship.NEXT] = nodes[i+1].get_doc_id()
    #if i > 0:
    #    n.relationships[DocumentRelationship.PREVIOUS] = nodes[i-1].get_doc_id()

kb_index = GPTVectorStoreIndex(
    nodes=nodes,
    service_context=service_context,
)

# kb index with each node composed by a single edge of the graph

nodes, id2embedding = [], {}
for i, triple in tqdm(enumerate(kb_triples), total=len(kb_triples)):
    text = f"({triple[0]}, {triple[1]}, {triple[2]})"
    print(f"{i}\t{text}")
    node = Node(
        text=text,
        doc_id=str(i),
    )
    #id2embedding[i] = service_context.embed_model._get_text_embedding(text)
    nodes.append(node)

#for i,n in enumerate(nodes):
    #n.relationships[DocumentRelationship.SOURCE] = n.get_doc_id()

kb_index_single_triples = GPTVectorStoreIndex(
    nodes=nodes,
    service_context=service_context,
)


# kb index with each node composed by:
# sentence\n
# triple_1\n
# triple_2\n
# ...

nodes = []
n = int(args.scale * len(sent2triples.items()))
if n != len(sent2triples.items()):
    print(f'> Sampling {n} examples out of the original {len(sent2triples)}')
    sent2triples = dict(random.sample(list(sent2triples.items()), n))

for i, (sentence, triples) in tqdm(enumerate(sent2triples.items()), total=len(sent2triples)):
    triples = '\n'.join([ f"({t[0]}, {t[1]}, {t[2]})" for t in triples ])
    text = f"{sentence}\n{triples}"
    node = Node(
        text=text,
        doc_id=str(i),
    )
    nodes.append(node)

kb_index_few_shots = GPTVectorStoreIndex(
    nodes=nodes,
    service_context=service_context,
)


# test retrieval
from llama_index.retrievers import VectorIndexRetriever

for index in (kb_index, kb_index_single_triples, kb_index_few_shots):
    print(f"\n> Testing index retriever\n")
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    r = retriever.retrieve("Linate Airport is located in Milan, Italy.")
    triples = [ n.node.text for n in r ]
    triples = '\n'.join(triples)
    print(triples)

save_name = f"{args.save}"
save_name_single_triples = f"{args.save}_single_triples"
save_name_few_shots = f"{args.save}_few-shots"
if args.normalize:
    save_name += "_normalized"
    save_name_single_triples += "_normalized"
    save_name_few_shots += "_normalized"
if complete:
    save_name += "_complete"
    save_name_single_triples += "_complete"
    save_name_few_shots += "_complete"
if args.scale != 1.:
    save_name += f'_scale-{args.scale}'
    save_name_single_triples += f'_scale-{args.scale}'
    save_name_few_shots += f'_scale-{args.scale}'
if args.no_overlap:
    save_name += '_no-overlap'
    save_name_single_triples += '_no-overlap'

print(f"> Saving index to {save_name}/.")
kb_index.storage_context.persist(save_name)
print(f"> Saving index to {save_name_single_triples}/.")
kb_index_single_triples.storage_context.persist(save_name_single_triples)
print(f"> Saving index to {save_name_few_shots}/.")
kb_index_few_shots.storage_context.persist(save_name_few_shots)


#import matplotlib.pyplot as plt
#nx.draw(g)
#plt.show()
