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
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from llama_index.indices.service_context import ServiceContext
from llama_index import LLMPredictor

def get_llm(model_id, pipeline):

    print(f"> Using model: {model_id}")
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=torch.cuda.is_available()
    )
    print(f'> Loaded in {time.time()-t:.4f}s')

    print(f'> Model data type: {model.dtype}')

    print(f'> Using device: {set(model.hf_device_map.values())}')
    
    pipe = Pipeline(
        pipeline,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        device_map='auto', # used for distributed
        #model_kwargs={"torch_dtype":torch.bfloat16}
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    llm_predictor = LLMPredictor(llm = llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

    return llm_predictor, service_context


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
