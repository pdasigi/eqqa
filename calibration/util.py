import torch


def get_k_best_spans(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        k: int
) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask


    # Here we take the span matrix and flatten it, then find the top-k best spans.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, k)
    k_best_spans = torch.topk(valid_span_log_probs.view(batch_size, -1), k, -1).indices

    span_start_indices = k_best_spans // passage_length
    span_end_indices = k_best_spans % passage_length
    # (batch_size, k, 2)
    return torch.stack([span_start_indices, span_end_indices], dim=-1)
