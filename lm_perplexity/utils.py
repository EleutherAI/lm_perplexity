import json


def read_json(path, **kwargs):
    with open(path, mode="r", **kwargs) as f:
        return json.loads(f.read())


def write_json(data, path, **kwargs):
    with open(path, mode="w", **kwargs) as f:
        f.write(json.dumps(data))


def get_rolling_token_windows(token_list, max_seq_len, context_len):
    pred_len = max_seq_len - context_len
    rolling_windows_and_pred_start = []
    predicted = 0

    # Special handling for first window: predict all tokens
    first_window = token_list[:max_seq_len]
    rolling_windows_and_pred_start.append((first_window, 0))
    predicted += len(first_window)

    while predicted < len(token_list):
        window_end = predicted + pred_len
        window = token_list[window_end - max_seq_len:window_end]
        rolling_windows_and_pred_start.append((window, context_len))
        predicted += len(window) - context_len

    return rolling_windows_and_pred_start



