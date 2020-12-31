# lm_perplexity

Code for benchmarking language models with the Pile.

## Usage

Evaluating on GPT-2 (uses GPU):

```bash
# Compute intermediate outputs for calculating perplexity (e.g. logprobs)
python lm_perplexity/save_lm_perplexity_data.py \
    --model_config_path preset_configs/gpt2_medium.json \
    --data_path /path/to/mydata.jsonl.zst \
    --output_path /path/to/perplexity_data.p

# Use intermediate outputs to compute perplexity
python lm_perplexity/compute_perplexity.py \
    --perplexity_data_path /path/to/perplexity_data.p \
    --output_path /path/to/perplexity.json
```

Evaluating on GPT-3 (requires OpenAI API key):

```bash
# Compute intermediate outputs for calculating perplexity (e.g. logprobs)
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python lm_perplexity/run_lm_perplexity.py \
    --model_config_path preset_configs/gpt3_curie.json \
    --data_path /path/to/mydata.jsonl.zst \
    --output_path /path/to/perplexity_data.p

# Use intermediate outputs to compute perplexity
python lm_perplexity/compute_perplexity.py \
    --perplexity_data_path /path/to/perplexity_data.p \
    --output_path /path/to/perplexity.json
```

## Assets

JSON files in `assets/${DATASET}/group${GROUP_ID}.json` contain the document indices for the canonical one-tenth split of the test set. Evaluation in the paper were performed on `group0`. 

## Requirements

* numpy
* torch
* transformers
* openai
* lm_dataformat
* tqdm