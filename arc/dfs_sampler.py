import torch
import torch.nn.functional as F


def dfs_sample(model, input_ids, tokenizer, eos_token_id=None, threshold_logprob=-2.3, max_len=150):
    """
    Run a depth-first sampling from a decoder-only LLM. Keeps only continuations
    where the cumulative log-prob is above a threshold.

    Args:
        model: the causal LM (must return logits)
        input_ids: torch.LongTensor of shape (1, seq_len)
        tokenizer: the model tokenizer
        eos_token_id: token ID for EOS (end-of-sequence)
        threshold_logprob: minimum cumulative log-prob (e.g. log(0.1) ~= -2.3)
        max_len: max total length of output sequence (including prompt)

    Returns:
        candidates: List of (logprob_sum, token_sequence) tuples
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    model.eval()
    candidates = []

    def explore(tokens, logprob_sum):
        if tokens[-1] == eos_token_id or len(tokens) >= max_len:
            candidates.append((logprob_sum, tokens))
            return

        with torch.no_grad():
            out = model(torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0))
            logits = out.logits[0, -1]
            probs = F.log_softmax(logits, dim=-1)

        topk = torch.topk(probs, k=30)  # only explore top-30 tokens
        for i in range(topk.indices.size(0)):
            token = topk.indices[i].item()
            score = topk.values[i].item()
            if logprob_sum + score >= threshold_logprob:
                explore(tokens + [token], logprob_sum + score)

    start = input_ids.squeeze(0).tolist()
    explore(start, 0.0)
    return sorted(candidates, key=lambda x: -x[0])  # sort by descending log-prob
