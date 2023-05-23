import json, re, torch, time

def normalize(string):
    string = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", string).lower()
    string = re.sub(r'_', ' ', string).lower()
    string = re.sub(r'\s+', ' ', string).lower()
    return string
    
def get_triples_from_json(infile):
    sent2triple = {}
    with open(infile, 'r') as f:
        d = json.load(f)['entries']
        for v in d:
            v = list(v.values())[0]
            triples = [
                (normalize(t['subject']), normalize(t['property']), normalize(t['object']))
                for t in v['modifiedtripleset']
            ]
            for lex in v['lexicalisations']:
                sent2triple.update({
                    lex['lex'] : triples
                })

    triples = []
    for t in sent2triple.values():
        triples += t
    triples = set(triples)

    return sent2triple, triples


def evaluate(p_triples, gt_triples):
    p_triples = set(p_triples)
    gt_triples = set(gt_triples)
    intersection = p_triples.intersection(gt_triples)
    precision = len(intersection) / len(p_triples)
    recall = len(intersection) / len(gt_triples)
    f1 = 2 * ( precision * recall ) / ( precision + recall )
    return precision, recall, f1


from transformers import pipeline as Pipeline
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.indices.service_context import ServiceContext
from llama_index import LLMPredictor, StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.prompts.prompts import KnowledgeGraphPrompt


def get_llm(model_id, pipeline, sentence_transformer='all-MiniLM-L6-v2'):

    print(f"> Using model: {model_id}")
    t = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except:
        assert False, "not implemented yet"
        
    pipe = Pipeline(
        pipeline,
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=128,
        device_map='auto', # used for distributed
        model_kwargs={"torch_dtype":torch.bfloat16}
    )
    
    print(f'> Loaded in {time.time()-t:.4f}s')

    print(f'> Model data type: {pipe.model.dtype}')

    print(f'> Using device: {set(pipe.model.hf_device_map.values())}')

    llm = HuggingFacePipeline(pipeline=pipe)
    llm_predictor = LLMPredictor(llm = llm)
    emb = HuggingFaceEmbeddings(model_name=sentence_transformer)
    emb = LangchainEmbedding(emb)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=emb,
        chunk_size_limit=512
    )

    return llm_predictor, service_context


def load_kb(kb_path, service_context, similarity_top_k=10):
    storage_context = StorageContext.from_defaults(persist_dir=kb_path)
    kb_index = load_index_from_storage(storage_context, service_context=service_context)
    retriever = VectorIndexRetriever(
        index=kb_index,
        similarity_top_k=similarity_top_k,
    )
    return kb_index, retriever


def get_relevant_triples(query, retriever):
    kb_triples = '\n'.join(
        node.node.text
        for node in retriever.retrieve(query)
    )
    return kb_triples


def get_triplet_extraction_prompt(body, examples, sentence=None, kb_retriever=None):
    if sentence is not None:
        assert kb_retriever is not None, "Pass a kb retriever to extract the kb triples relevant to the input sentence."
        triples = get_relevant_triples(sentence, kb_retriever)
        prompt = body + ("Make use of the following relevant triplets:\n{}\n").format(triples)
    else:
        prompt = body + '\n'
    prompt += examples
    prompt += (
        "Text: {text}\n"
        "Triplets:\n"
    )
    prompt = prompt.replace(';',',')
    return KnowledgeGraphPrompt(prompt)


def extract_triples(sentences, prompt, kg_index, max_knowledge_triplets=10):
    triples = set()
    for sent in sentences:
        #prompt = get_triplet_extraction_prompt(body, examples, n.text, kb_retriever)
        #print(prompt.prompt.template)
        index.kg_triple_extract_template = prompt.partial_format(
            max_knowledge_triplets=max_knowledge_triplets
        )
        for t in index._extract_triplets(n.text):
            triples.add(t)
    return list(triples)
    
if __name__ == '__main__':
    import sys
    d, triples = get_triples_from_json(sys.argv[1])
    for k,v in d.items():
        for t in v:
            assert t in triples
        print('---------------------------------------------------------------')
        print(k)
        print(v)
        print('---------------------------------------------------------------')
