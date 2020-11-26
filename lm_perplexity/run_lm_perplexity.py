import argparse
import json
import numpy as np
from tqdm import auto as tqdm_lib

import lm_dataformat
import lm_perplexity.models as models
import lm_perplexity.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', default="")
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--max_docs', type=int, default=None)
    return parser.parse_args()


def compute_perplexity(model, data_path, max_docs=None):
    aggregate_logprobs = 0
    aggregate_length = 0
    aggregate_logprobs_position_buckets = utils.LogprobsPositionBuckets(max_seq_len=model.max_seq_len)

    reader = lm_dataformat.Reader(data_path)
    for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
        output = model.get_perplexity_data(doc)
        aggregate_logprobs += output["avg_logprobs"] * output["length"]
        aggregate_length += output["length"]
        aggregate_logprobs_position_buckets.update_with_buckets(buckets=output["logprobs_position_buckets"])
        if max_docs is not None and i == max_docs:
            break
    logprobs_position_buckets_summary = aggregate_logprobs_position_buckets.get_summary()
    return {
        "perplexity": float(np.exp(-aggregate_logprobs / aggregate_length)),
        "num_tokens": int(aggregate_length),
        "per_position": {
            "perplexity": [
                float(np.exp(-logprob))
                for logprob, count in zip(
                    logprobs_position_buckets_summary["logprobs"],
                    logprobs_position_buckets_summary["counts"],
                )
            ],
            "counts": [int(i) for i in logprobs_position_buckets_summary["counts"]],
        },
    }


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    perplexity_data = compute_perplexity(
        model=model,
        data_path=args.data_path,
        max_docs=args.max_docs,
    )
    if args.output_path:
        utils.write_json(perplexity_data, args.output_path)
    del perplexity_data["per_position"]
    print(json.dumps(perplexity_data, indent=4))


if __name__ == "__main__":
    main()
