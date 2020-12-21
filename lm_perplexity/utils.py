import contextlib
import json
import time

WB_MAX_IN_TIME_SPAN = 600
WB_TIME_SPAN = 60


def read_json(path, **kwargs):
    with open(path, mode="r", **kwargs) as f:
        return json.loads(f.read())


def write_json(data, path, **kwargs):
    with open(path, mode="w", **kwargs) as f:
        f.write(json.dumps(data))


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield (
        [prefix_token] + token_list[:first_seq_len - 1],
        token_list[:first_seq_len]
    )
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (
            token_list[window_end - max_seq_len - 1:window_end - 1],
            token_list[window_end - window_pred_len:window_end],
        )
        predicted += window_pred_len


class WaitBlocker:
    def __init__(self, backoff=1, verbose=True, max_in_time_span=WB_MAX_IN_TIME_SPAN, time_span=WB_TIME_SPAN):
        self.backoff = backoff
        self.verbose = verbose
        self.max_in_time_span = max_in_time_span
        self.time_span = time_span
        self.record = []

    def wait_until_valid(self):
        i = 0
        now = time.time()
        for i in range(len(self.record)):
            if self.record[i] > now - self.time_span:
                break
        self.record = self.record[i:]
        if len(self.record) >= self.max_in_time_span:
            delta = self.record[i] - now + self.time_span
            print(f"Backing off for {delta:.1f}")
            time.sleep(delta)

    def add_record(self):
        self.record.append(time.time())

    @contextlib.contextmanager
    def check_valid(self):
        self.wait_until_valid()
        yield
        self.add_record()
