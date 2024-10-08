#  Knowledge Graph Triplet Extraction with LLMs

A pipeline for extracting knoweldge triplets from text using LLMs and Knowledge Bases. Refer to [Zero- and Few-Shots Knowledge Graph Triplet Extraction with Large Language Models](https://aclanthology.org/2024.kallm-1.2/) for the details.
If you find this useful please consider citing us:

```
@inproceedings{papaluca-etal-2024-zero,
    title = "Zero- and Few-Shots Knowledge Graph Triplet Extraction with Large Language Models",
    author = "Papaluca, Andrea  and
      Krefl, Daniel  and
      Rodr{\'\i}guez M{\'e}ndez, Sergio  and
      Lensky, Artem  and
      Suominen, Hanna",
    editor = "Biswas, Russa  and
      Kaffee, Lucie-Aim{\'e}e  and
      Agarwal, Oshin  and
      Minervini, Pasquale  and
      Singh, Sameer  and
      de Melo, Gerard",
    booktitle = "Proceedings of the 1st Workshop on Knowledge Graphs and Large Language Models (KaLLM 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.kallm-1.2",
    pages = "12--23",
    abstract = "In this work, we tested the Triplet Extraction (TE) capabilities of a variety of Large Language Models (LLMs) of different sizes in the Zero- and Few-Shots settings. In detail, we proposed a pipeline that dynamically gathers contextual information from a Knowledge Base (KB), both in the form of context triplets and of (sentence, triplets) pairs as examples, and provides it to the LLM through a prompt. The additional context allowed the LLMs to be competitive with all the older fully trained baselines based on the Bidirectional Long Short-Term Memory (BiLSTM) Network architecture. We further conducted a detailed analysis of the quality of the gathered KB context, finding it to be strongly correlated with the final TE performance of the model. In contrast, the size of the model appeared to only logarithmically improve the TE capabilities of the LLMs. We release the code on GitHub for reproducibility.",
}
```

## Requirements

```
llama-index
langchain
transformers
bs4
nltk
sklearn
tqdm
matplotlib
lxml
networkx
torch
```

## Datasets
### WebNLG
 - `webnlg/`: original data of the 2020 `WebNLG` challenge [https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0).
 - `webnlg_modified/`: adaptation of the `WebNLG` data by [Zheng et al](https://aclanthology.org/P17-1113/).
### NYT
 - `nyt/`: the New York Times dataset by Riedel et al

## Building the Knowledge Base

The Knowledge Base can be built by running the `kb_builder.py` script specifying which data to use. For example to build the KB associated to the train and valid splits of the `WebNLG` dataset, just run:
```
python kb_builder.py --data webnlg_modified/{train,valid}.json
```
This will create two indices `webnlg_modified/kb_triplets/` and `webnlg_modified/kb_sentence-triplets/`. The first one stores separately each triplet found in the provided data (0.5-shots):
```
Node-1
------
    (Egg Harbor Township , New Jersey, isPartOf, Atlantic)
Node-2
------
    (Egg Harbor Township , New Jersey, isPartOf, New Jersey)
Node-3
------
    (Atlantic City International Airport, location, Egg Harbor Township , New Jersey)
```
the second one stores in each node the original sentence followed by the triplets it contains (few-shots):
```
Node-1
------
    Atlantic City International Airport in Egg Harbor Township , New Jersey is in the U.S.A . The airport has a runway that is 1,873 long .
    (Egg Harbor Township , New Jersey, isPartOf, Atlantic)
	(Egg Harbor Township , New Jersey, isPartOf, New Jersey)
	(Atlantic City International Airport, location, Egg Harbor Township , New Jersey)
```
It's recommended to pass also the `--normalize` argument to the script in order to normalize sentences and triplets (e.g. removing unwanted punctuation, moving everything to the lower case, etc...).

## Running the Benchmark

The `benchmark.py` script, instead, runs the triplet extraction pipeline on the specified data with the specified LLM and prompt. For example to test the `gpt2` model on the `WebNLG` dataset with the base prompt:
```
python benchmark.py \
    --data webnlg_modified/test.json \
	--conf model_conf/gpt2.conf \
	--prompt prompts/prompt_base.json
```
The configurations of the different LLMs can be found under `model_conf/`. Note that for the `gpt3-turbo` and `gpt4` models, an OpenAI key is required, please place it in the `openai_key.txt` file as the only line.
Upon completion, the extracted triplets are going to be saved under `webnlg_modified/gpt2/prompt_base/generated_triples_gpt2_temp-0.1.xml`.
By specifying the `--kb` argument it is possible to make the LLM rely on the Knowledge Base built above, remember to also set the appropriate prompt, e.g. `prompt_base_kb.json`:
```
python benchmark.py \
    --data webnlg_modified/test.json \
	--conf model_conf/gpt2.conf \
	--kb webnlg_modified/kb_triplets/ \
	--prompt prompts/prompt_base_kb.json \
	--top_k 5
```
Additionally the number of nodes of the KB index to retrieve can be specified with the `--top_k` argument.

## Performance Evaluation

To evaluate the extracted triplets use the `Evaluation_script.py`, that is adapted from the evaluation script that was provided originally for the `WebNLG` challenge [https://github.com/WebNLG/WebNLG-Text-to-triples](https://github.com/WebNLG/WebNLG-Text-to-triples).
For example, to evaluate the triplets generated by the `gpt2` model shown above, first, generate the groundtruth (`webnlg_modified/groundtruth_triples.xml`) by simply running:
```
python benchmark.py \
    --data webnlg_modified/test.json \
	--groundtruth
```
and then evaluate the `micro`/`macro` averaged performance:
```
python Evaluation_script.py \
    --predictions webnlg_modified/gpt2/prompt_base/generated_triples_gpt2_temp-0.1.xml \
	--groundtruth webnlg_modified/groundtruth_triples.xml \
	--avg micro
```

