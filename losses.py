import torch
import torch.nn.functional as F
from metrics import es_sentiment
from utils import gather_log_probs, mask_hf_labels, masked_mean


def balanced_bce(log_probs, labels, eps=torch.finfo(torch.float32).eps):
    assert labels.max() <= 1
    assert labels.min() >= 0

    pos_losses = -log_probs[labels == 1]
    neg_probs = 1 - log_probs.exp()
    neg_probs[neg_probs == 0] += eps  # for numerical stability
    neg_losses = -neg_probs.log()[labels == 0]
    pos_loss = pos_losses.mean() if pos_losses.numel() > 0 else 0
    neg_loss = neg_losses.mean() if neg_losses.numel() > 0 else 0

    return pos_loss + neg_loss


def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError


def binary_log_probs(pred, targ, should_reduce=True):
    assert targ.max() <= 1
    assert targ.min() >= 0
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float()
    if should_reduce:
        acc = acc.mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0]
    }


def multiclass_log_probs(
    pred,
    raw_targets,
    shift=True,
    eps=torch.finfo(torch.float32).eps,
    should_reduce=True,
    **kwargs,
):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    mask, targ = mask_hf_labels(raw_targets)
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift to align predictions and targets

    unmasked_log_probs = gather_log_probs(pred, targ)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)  # We want to get the whole sequence right
    acc = correct.float()
    if should_reduce:
        acc = acc.mean()

    if "inner_sent" in kwargs:
        # Only use outer samples with the same sentiment as the inner sample
        same_sent_mask = torch.tensor([i == o for i, o in zip(kwargs["inner_sent"], kwargs["outer_sent"])], device=pred.device)
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        avg_log_prob = good_log_prob

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        nll = -avg_log_prob

    info_dict = {
        "acc": acc,
        "log_prob": avg_log_prob,
        "prob": avg_log_prob.exp(),
        "n_tokens": n_tokens,
        "nll": nll
    }

    if "inner_sent" in kwargs:
        info_dict.update(es_sentiment(kwargs["pre_edit_logits"],
                                      kwargs["post_edit_logits"],
                                      raw_targets,
                                      same_sent_mask))

    return info_dict


def masked_log_probs(pred, targ, shift=True, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        should_reduce = True
        if "should_reduce" in kwargs:
            should_reduce = kwargs["should_reduce"]
        return binary_log_probs(pred, targ, should_reduce=should_reduce)
    else:
        return multiclass_log_probs(pred, targ, shift=shift, **kwargs)


def test_masked_log_probs():
    print()
    N = 10000
    pred = torch.randn(10, 15, N)
    targ = torch.randint(0, N, (10, 15))
    true_pred = pred.clone()
    true_pred.scatter_(2, targ.unsqueeze(-1), 5)
    true_pred = true_pred.roll(-1, 1)

    half_pred = true_pred.clone()
    mask = torch.arange(10) % 2 == 0
    half_pred[mask] = pred[mask]

    pred_ = pred.clone()
    true_pred_ = true_pred.clone()
    half_pred_ = half_pred.clone()
    targ_ = targ.clone()

    print(masked_log_probs(pred, targ, return_acc=True))
    print(masked_log_probs(true_pred, targ, return_acc=True))
    print(masked_log_probs(half_pred, targ, return_acc=True))

    assert (pred == pred_).all()
    assert (targ == targ_).all()
    assert (half_pred == half_pred_).all()
    assert (true_pred == true_pred_).all()

    import pdb; pdb.set_trace()

    pred = torch.randn(1000, 15, 1)
    targ = torch.randint(0, 2, (1000, 15))

    print(masked_log_probs(pred, targ, return_acc=True))


if __name__ == "__main__":
    torch.manual_seed(0)

    test_masked_log_probs()
