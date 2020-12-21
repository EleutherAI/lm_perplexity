import argparse
import torch
from tqdm import auto as tqdm_lib

import lm_dataformat
import lm_perplexity.models as models
import lm_perplexity.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--doc_indices_path', type=str, default=None)
    return parser.parse_args()


def compute_perplexity_data(model, data_path, indices=None):
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
        if indices is not None and i not in indices:
            continue
        output = model.get_perplexity_data(doc)
        if not output:
            continue
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]

    return overall_output


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    if args.doc_indices_path:
        assert args.max_docs is None
        indices = set(utils.read_json(args.doc_indices_path))
    elif args.max_docs:
        assert args.doc_indices_path is None
        indices = set(range(args.max_docs))
    else:
        indices = None
    perplexity_data = compute_perplexity_data(
        model=model,
        data_path=args.data_path,
        indices=indices,
    )
    torch.save(perplexity_data, args.output_path)


if __name__ == "__main__":
    main()
