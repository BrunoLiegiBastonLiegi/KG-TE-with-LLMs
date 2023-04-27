import torch, time, argparse, json

from transformers import pipeline as Pipeline
from transformers import  AutoTokenizer

parser = argparse.ArgumentParser(description='Generation speed test.')
parser.add_argument('--conf', default='llm.conf')

args = parser.parse_args()

with open(args.conf, 'r') as f:
    conf = json.load(f)

model_id = conf['model']
    
print(f"> Using model: {model_id}")
t = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = Pipeline(
    conf['pipeline'],
    model=conf['model'],
    tokenizer=tokenizer,
    max_new_tokens=64,
    device_map='auto', # used for distributed
    #model_kwargs={"torch_dtype":torch.bfloat16}
)
print(f'> Loaded in {time.time()-t:.4f}s')

print(f'> Model data type: {pipe.model.dtype}')

print(f'> Using device: {set(pipe.model.hf_device_map.values())}')

prompts = [
    "Italy, officially the Italian Republic or the Republic of Italy, is",
    "Italy, officially the Italian Republic or the Republic of Italy, is a country in Southern and Western Europe.",
    "Italy, officially the Italian Republic or the Republic of Italy, is a country in Southern and Western Europe. Located in the middle of the Mediterranean Sea, it consists of a peninsula delimited by the Alps and surrounded by several islands;",
    "Italy, officially the Italian Republic or the Republic of Italy, is a country in Southern and Western Europe. Located in the middle of the Mediterranean Sea, it consists of a peninsula delimited by the Alps and surrounded by several islands; its territory largely coincides with the homonymous geographical region. Italy shares land borders with France, Switzerland, Austria, Slovenia and the enclaved microstates of Vatican City and San Marino. "
]

for prompt in prompts:
    print('--------------------------------------------------------------------------------------\n')
    print(f'> Prompt:\n  {prompt}')
    print(f"  > {len(tokenizer(prompt)['input_ids'])} input tokens")

    t0 = time.time()
    out = pipe(
        prompt,
        return_full_text = False
    )
    t1 = time.time()
    
    print(f"\n> Generation:\n  {out[0]['generated_text']}")

    print(f"  > {len(tokenizer(out[0]['generated_text'])['input_ids'])} tokens generated")
    print(f"\n> {len(tokenizer(out[0]['generated_text'])['input_ids'])/(t1-t0):.4f} tokens/s")
    print('\n--------------------------------------------------------------------------------------')
