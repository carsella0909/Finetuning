import torch
import torch.nn.functional as F


def compute_logprob(model, input_ids, candidate_ids):
    """
    Compute total log-probability of generating candidate_ids given input_ids.

    Args:
        model: the causal language model
        input_ids: List[int], flattened input token sequence
        candidate_ids: List[int], generated tokens (target prediction)

    Returns:
        float: summed log-probability of the candidate sequence
    """
    model.eval()
    full_input = torch.tensor(input_ids + candidate_ids[:-1], dtype=torch.long).unsqueeze(0).to(model.device)
    target = torch.tensor(candidate_ids, dtype=torch.long).unsqueeze(0).to(model.device)

    with torch.no_grad():
        output = model(full_input)
        logits = output.logits[:, -target.size(1)-1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)

    gathered = torch.gather(log_probs, dim=2, index=target.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def score_candidates(model, tokenizer, prompt_ids_list, candidate_ids):
    """
    Score a candidate solution using multiple augmentations of the prompt.

    Args:
        model: the causal LLM
        tokenizer: tokenizer used
        prompt_ids_list: List[List[int]], augmented prompts
        candidate_ids: List[int], tokens of one candidate

    Returns:
        float: average log-probability across augmented prompts
    """
    scores = []
    for prompt in prompt_ids_list:
        score = compute_logprob(model, prompt, candidate_ids)
        scores.append(score)

    return sum(scores) / len(scores)
