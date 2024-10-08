import json, re, torch, time, sys
from tqdm import tqdm

import os, openai
with open('openai_key.txt', 'r') as f:
    api_key = f.read().replace('\n','')
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

def normalize(string):
    string = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", string).lower()
    string = re.sub(r'_', ' ', string).lower()
    string = re.sub(r'\s+', ' ', string).lower()
    return string

#sys.path.append('./webnlg/corpus-reader/')
#from benchmark_reader import Benchmark, select_files

def get_data_from_files(infiles):
    sent2triple, triples = {}, []
    if not isinstance(infiles, list):
        infiles = [infiles]
    for infile in infiles:
        with open(infile, 'r') as f:
            entries = json.load(f)
            for entry in entries.values():
                tmp = [ tuple(t) for t in entry['triplets'] ] # normalization of entity/relation names needed??
                sent2triple[entry['sentence']] = tmp
                triples += tmp
    triples = set(triples)

    return sent2triple, triples

from bs4 import BeautifulSoup

def get_triples_from_xml(*files):
    triples = []
    for file in files:
        with open(file, 'r') as f:
            xml = f.read()
        soup = BeautifulSoup(xml, 'xml')
        entries = soup.find_all('entry')
        triples.append([
            [ e.text for e in entry.find_all('gtriple') ]
            for entry in entries
        ])
    return triples
        

def get_data_loader(datafile: str):
    sent2triples, _ = get_data_from_files(datafile)
    return sent2triples.items()

def triple_equality(triple_1, triple_2):
    for x,y in zip(triple_1, triple_2):
        if normalize(x) != normalize(y):
            return False
    return True

def evaluate(p_triples, gt_triples):
    TP, FP, FN = 0, 0, 0
    to_set = lambda _list: set([tuple(l) for l in _list])
    # Loop over the instances
    for prediction, groundtruth in tqdm(zip(p_triples, gt_triples)):
        prediction = [ normalize_triple(t) for t in prediction]
        groundtruth = [ normalize_triple(t) for t in groundtruth]
        # convert predictions list to set
        prediction = to_set(prediction)
        # loop over the expected triples
        for gt_triple in groundtruth:
            found = False
            # check whether the correct triple was extracted
            for pred_triple in prediction:
                if triple_equality(gt_triple, pred_triple):
                    # found it! 
                    found = True
                    # back up the triple
                    correct_triple = pred_triple
                    break
            if found:
                TP += 1
                # remove the found triples
                prediction.remove(correct_triple)
            else:
                FN += 1
        FP += len(prediction)
    precision = TP / (FP + TP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


from transformers import pipeline as Pipeline
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.indices.service_context import ServiceContext
from llama_index import LLMPredictor, StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.prompts.prompts import KnowledgeGraphPrompt
from llama_index.llms import OpenAI
from functools import partial
from collections import defaultdict


def get_llm(model_id, pipeline, sentence_transformer='all-MiniLM-L6-v2', **model_kwargs):

    print(f"> Preparing model: {model_id}")
    t = time.time()
    is_openai = True if 'openai' in model_kwargs.keys() else False
    if not is_openai:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            print(f'> Loaded tokenizer in {time.time()-t:.4f}s')

        if 'load_in_4bit' not in model_kwargs:
            model_kwargs['load_in_4bit'] = False
        if 'load_in_8bit' not in model_kwargs:        
            model_kwargs['load_in_8bit'] = False
        if model_kwargs['load_in_4bit'] and model_kwargs['load_in_8bit']:
            raise AssertionError("Both 8bit and 4bit model loading specified, pick one.")
        else:
            if not (model_kwargs['load_in_4bit'] or model_kwargs['load_in_8bit']):
                model_kwargs["torch_dtype"] = torch.bfloat16
            
        if 'device_map' not in model_kwargs:
            model_kwargs['device_map'] = 'auto'
        max_new_tokens = model_kwargs.pop('max_new_tokens')
        temperature = model_kwargs.pop('temperature')
    
        t = time.time()
        pipe = Pipeline(
            pipeline,
            model=model_id,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            model_kwargs=model_kwargs,
            temperature=temperature,
            trust_remote_code=True
        )
        print(f'> Created pipeline in {time.time()-t:.4f}s')

        print(f'> Model data type: {pipe.model.dtype}')

        print(f'> Using device: {set(pipe.model.hf_device_map.values())}')

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        print(os.environ["OPENAI_API_KEY"])
        llm = OpenAI(
            model = model_id,
            temperature = model_kwargs['temperature'],
            max_tokens = model_kwargs['max_new_tokens'],
            openai_api_key=api_key
        )
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


def prepare_triples(retrieved_nodes, few_shots=False, n_triplets_per_predicate=None, return_tuple=False):
    if few_shots:
        sents, kb_triples = [], []
        for node in retrieved_nodes:
            node = node.node.text.split('\n')
            sents.append(node[0])
            triples = node[1:]
            if return_tuple:
                triples = [ tuple(t[1:-1].split(', ')) for t in triples ]
            else:
                triples = '\n'.join(triples)
            kb_triples.append(triples)
        return sents, kb_triples
    else:
        kb_triples = [ node.node.text for node in retrieved_nodes ]
        if return_tuple or n_triplets_per_predicate is not None:
            kb_triples = [ tuple(t[1:-1].split(', ')) for t in kb_triples ]
        if n_triplets_per_predicate is not None:
            predicates = set([ t[1] for t in kb_triples ])
            predicates = dict(zip(predicates, [0 for i in range(len(predicates))]))
            new_triples = []
            for t in kb_triples:
                if predicates[t[1]] < n_triplets_per_predicate:
                    new_triples.append(t)
                    predicates[t[1]] += 1
            kb_triples = new_triples
            if not return_tuple:
                kb_triples = ['({}, {}, {})'.format(*t) for t in kb_triples]
        if not return_tuple:
            kb_triples = '\n'.join(kb_triples)
        return None, kb_triples


def get_relevant_triples(query, retriever, return_tuple=False, n_triplets_per_predicate=None, few_shots=False):
    retrieved_nodes = retriever.retrieve(query)
    sents, kb_triples = prepare_triples(
        retrieved_nodes,
        few_shots=few_shots,
        n_triplets_per_predicate=n_triplets_per_predicate,
        return_tuple=return_tuple
    )
    return sents, kb_triples
    

def get_triplet_extraction_prompt(body, examples, answer, sentence=None, kb_retriever=None, few_shots=False):
    #prompt = f"{body}"
    if sentence is not None and kb_retriever is not None:
        sents, triples = get_relevant_triples(sentence, kb_retriever, n_triplets_per_predicate=2, few_shots=few_shots)
        if few_shots:
            sent_triples = '\n'.join([ f"Text: {s}\nTriplets:\n{t}" for s,t in zip(sents,triples) ])
            examples = examples.format(examples=sent_triples)
        else:
            answer = answer.format(text='{text}', context_triplets=triples)
                
    prompt = f"{body}{examples}{answer}"
    #prompt = prompt.replace(';',',')
    return KnowledgeGraphPrompt(prompt)


BODY = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords. "
)
"""
EXAMPLES = (
    "---------------------\n"
    "Example:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets:\n(Alice; is mother of; Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz; is; coffee shop)\n"
    "(Philz; founded in; Berkeley)\n"
    "(Philz; founded in; 1982)\n"
    "---------------------"
)
"""
EXAMPLES = (
    "---------------------\n"
    "Examples:\n"
    "Text: Abilene, Texas is in the United States.\n"
    "Triplets:\n(abilene, texas, country, united states)\n"
    "Text: The United States includes the ethnic group of African Americans and is the birthplace of Abraham A Ribicoff who is married to Casey Ribicoff.\n"
    "Triplets:\n"
    "(abraham a. ribicoff, spouse, casey ribicoff)\n"
    "(abraham a. ribicoff, birth place, united states)\n"
    "(united states, ethnic group, african americans)\n"
    "---------------------\n"
)
ANSWER = (
    "Text: {text}\n"
    "Triplets:\n"
)

def extract_triples(sentences, kg_index, max_knowledge_triplets=None, prompt=None, kb_retriever=None, few_shots=False):
    if not isinstance(sentences, list):
        sentences = [sentences]
    triples = set()
    if max_knowledge_triplets is None:
        max_knowledge_triplets = kg_index.max_triplets_per_chunk
    for sent in sentences:
        if prompt is not None:
            prompt = get_triplet_extraction_prompt(prompt['body'], prompt['examples'], prompt['answer'], sent, kb_retriever, few_shots=few_shots)
        else:
            global BODY, EXAMPLES, ANSWER
            prompt = get_triplet_extraction_prompt(
                body=BODY, examples=EXAMPLES, answer=ANSWER,
                sentence=sent,
                kb_retriever=kb_retriever
            )
        kg_index.kg_triple_extract_template = prompt.partial_format(
            max_knowledge_triplets=max_knowledge_triplets
        )
        print('> Prompt Template:')
        print(kg_index.kg_triple_extract_template.template)
        for t in kg_index._extract_triplets(sent):
            triples.add(t)
    return list(triples)

def normalize_triple(triple):
    if isinstance(triple, str):
        newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
        newtriple = re.sub(r'_', ' ', newtriple).lower()
        newtriple = re.sub(r'\s+', ' ', newtriple).lower()
        adjusttriple = newtriple.split(' | ')
        manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
        if manualmodified:
            adjusttriple[-1] = manualmodified.group(1)
            newtriple = ' | '.join(adjusttriple)
        newtriple = newtriple.replace(',', '')
    elif isinstance(triple, list):
        newtriple = []
        for element in triple:
            new_el = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", element).lower()
            new_el = re.sub(r'_', ' ', new_el).lower()
            new_el = re.sub(r'\s+', ' ', new_el).lower()
            new_el = new_el.replace(',', '')
            if new_el != ' ' and new_el[0] == ' ':
                new_el = new_el[1:]
            if new_el != ' ' and new_el[-1] == ' ':
                new_el = new_el[:-1]
            newtriple.append(new_el)
        manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', newtriple[-1])
        if manualmodified:
            newtriple[-1] = manualmodified.group(1)
    elif isinstance(triple, tuple):
        return normalize_triple(list(triple))    
    else:
        raise TypeError("Expected list or str.")    
    return newtriple
    
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
