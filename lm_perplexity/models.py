import numpy as np
import os

import openai
import torch
import torch.nn as nn
import transformers

import lm_perplexity.utils as utils


class LM:
    def get_perplexity_data(self, text) -> dict:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError


class GPT3LM(LM):

    def __init__(self, engine, context_len=1024, max_seq_len=2048, verbose=False):
        import openai
        self.engine = engine
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    def get_perplexity_data(self, text) -> dict:
        # noinspection DuplicatedCode
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        token_windows_and_pred_start = utils.get_rolling_token_windows(
            token_list=input_ids,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        logprobs_position_buckets = utils.LogprobsPositionBuckets(self.max_seq_len)
        # Special handling for first window, which has no context:
        #   - We add a prefix token, and compute logprobs for all tokens in the first window
        block_output = self.get_token_logprobs(
            token_ids=[self.end_of_text_token_id] + token_windows_and_pred_start[0][0],
            pred_start=1,
        )
        all_logprobs.append(block_output["logprobs"])
        logprobs_position_buckets.update_single(
            logprobs=block_output["logprobs"],
            positions=block_output["positions"],
        )

        # Remaining windows
        for token_window, pred_start in token_windows_and_pred_start[1:]:
            block_output = self.get_token_logprobs(
                token_ids=token_window,
                pred_start=pred_start,
            )
            all_logprobs.append(block_output["logprobs"])
            logprobs_position_buckets.update_single(
                logprobs=block_output["logprobs"],
                positions=block_output["positions"],
            )

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        assert len(all_logprobs) == len(input_ids)
        return {
            "avg_logprobs": np.mean(all_logprobs),
            "length": len(all_logprobs),
            "logprobs_position_buckets": logprobs_position_buckets,
            # "token_logprobs": all_logprobs,
        }

    def get_token_logprobs(self, token_ids, pred_start):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=token_ids,
            max_tokens=0,
            temperature=0.0,
            logprobs=0,
            echo=True,
        )
        logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(token_ids))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(token_ids)[pred_start:])
            print("Perplexity:", np.exp(-logprobs.mean()))
            print()

        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))

        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


class GPT2LM(LM):

    def __init__(self, model_name, device="cuda:0", context_len=512, max_seq_len=1024, verbose=False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        torch.set_grad_enabled(False)
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name).eval().to(self.device)
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

    def get_perplexity_data(self, text) -> dict:
        # noinspection DuplicatedCode
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        token_windows_and_pred_start = utils.get_rolling_token_windows(
            token_list=input_ids,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        logprobs_position_buckets = utils.LogprobsPositionBuckets(self.max_seq_len)
        # Special handling for first window, which has no context:
        #   - We add a prefix token, and compute logprobs for all tokens in the first window
        block_output = self.get_token_logprobs(
            token_ids=[self.end_of_text_token_id] + token_windows_and_pred_start[0][0],
            pred_start=1,
        )
        all_logprobs.append(block_output["logprobs"])
        logprobs_position_buckets.update_single(
            logprobs=block_output["logprobs"],
            positions=block_output["positions"],
        )

        # Remaining windows
        for token_window, pred_start in token_windows_and_pred_start[1:]:
            block_output = self.get_token_logprobs(
                token_ids=token_window,
                pred_start=pred_start,
            )
            all_logprobs.append(block_output["logprobs"])
            logprobs_position_buckets.update_single(
                logprobs=block_output["logprobs"],
                positions=block_output["positions"],
            )

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        assert len(all_logprobs) == len(input_ids)
        return {
            "avg_logprobs": np.mean(all_logprobs),
            "length": len(all_logprobs),
            "logprobs_position_buckets": logprobs_position_buckets,
            # "token_logprobs": all_logprobs,
        }

    def get_token_logprobs(self, token_ids, pred_start):
        token_ids = torch.tensor(token_ids).long().to(self.device)
        # We always drop the last token_id, since we only score predictions on it but never
        #   condition on it
        output = self.model(token_ids[:-1], return_dict=True)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        # Reverse the 1-offset from above
        neg_logprobs = loss_fct(output.logits[pred_start-1:], token_ids[pred_start:]).detach().cpu().numpy()
        if self.verbose:
            print("Context:", self.tokenizer.convert_ids_to_tokens(token_ids))
            print("Predicting:", self.tokenizer.convert_ids_to_tokens(token_ids)[pred_start:])
            print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()

        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))
        import pdb; pdb.set_trace()

        return {
            "logprobs": - neg_logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)


def create_model(json_path):
    config = utils.read_json(json_path)
    model_type = config.pop("model_type")
    if model_type == "gpt3":
        model = GPT3LM.create_from_config(config)
    elif model_type == "gpt2":
        model = GPT2LM.create_from_config(config)
    else:
        raise KeyError(model_type)
    return model
