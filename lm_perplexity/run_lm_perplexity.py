import argparse
import json
from tqdm import auto as tqdm_lib

import lm_dataformat
import lm_perplexity.models as models
import lm_perplexity.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', default="")
    parser.add_argument('--output_path', default=None)
    return parser.parse_args()


def compute_perplexity(model, data_path):
    aggregate_logprobs = 0
    aggregate_length = 0

    reader = lm_dataformat.Reader(data_path)
    for doc in tqdm_lib.tqdm(reader.stream_data()):
        output = model.get_perplexity_data(doc)
        aggregate_logprobs += output["avg_logprobs"] * output["length"]
        aggregate_length += output["length"]
    return {
        "perplexity": float(2 ** (-aggregate_logprobs / aggregate_length)),
        "num_tokens": int(aggregate_length),
    }


def main():
    args = parse_args()
    model = models.create_model(args.config_path)
    perplexity_data = compute_perplexity(model=model, data_path=args.data_path)
    formatted_perplexity_data = json.dumps(perplexity_data, indent=4)
    print(formatted_perplexity_data)
    if args.output_path:
        utils.write_json(formatted_perplexity_data, args.output_path)


if __name__ == "__main__":
    main()
