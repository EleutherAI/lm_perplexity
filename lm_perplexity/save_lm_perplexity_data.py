import argparse
import torch
from tqdm import auto as tqdm_lib

import lm_dataformat
import lm_perplexity.models as models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_docs', type=int, default=None)
    return parser.parse_args()


def compute_perplexity_data(model, data_path, max_docs=None):
    # For expedience, we're going to assume everything fits in memory for now
    # Also for expedience we're just going to save lists of arrays
    overall_output = {
        "all_logprobs": [],
        "all_positions": [],
        "aggregate_length": 0,
        "aggregate_utf8_length": 0.
    }

    reader = lm_dataformat.Reader(data_path)
    for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
        output = model.get_perplexity_data(doc)
        if not output:
            continue
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]
        if max_docs is not None and i == max_docs:
            break

    return overall_output


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    perplexity_data = compute_perplexity_data(
        model=model,
        data_path=args.data_path,
        max_docs=args.max_docs,
    )
    torch.save(perplexity_data, args.output_path)


if __name__ == "__main__":
    main()
