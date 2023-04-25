import json, re

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
