# lm_perplexity

## Usage

Evaluating on GPT-2 (uses GPU):

```bash
python lm_perplexity/run_lm_perplexity.py \
    --model_config_path preset_configs/gpt2_medium.json \
    --data_path /path/to/mydata.jsonl.zst
```

Evaluating on GPT-3 (requires OpenAI API key):

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python lm_perplexity/run_lm_perplexity.py \
    --model_config_path preset_configs/gpt3_curie.json \
    --data_path /path/to/mydata.jsonl.zst
```

Additional Options:

* `--max_docs n`: Only evaluate the first `n` docs. Use this for testing.
* `--output_path /path/to/results.json`: Write results to JSON file.

## Requirements

* torch
* transformers
* openai
* lm_dataformat
* tqdm