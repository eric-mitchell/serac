import torch
from utils import gather_log_probs, mask_hf_labels, masked_mean


def es_sentiment(pre_logits, post_logits, raw_targets, same_sent_mask, NULL_TOKEN=0):
    with torch.no_grad():
        mask, targ = mask_hf_labels(raw_targets)
        pos_mask = same_sent_mask.unsqueeze(-1) * mask
        neg_mask = (~same_sent_mask).unsqueeze(-1) * mask

        # Compute log likelihoods of pos/neg samples
        pre_edit_token_log_probs = gather_log_probs(pre_logits, targ)
        post_edit_token_log_probs = gather_log_probs(post_logits, targ)

        mean_pos_pre = masked_mean(pre_edit_token_log_probs, pos_mask)
        mean_pos_post = masked_mean(post_edit_token_log_probs, pos_mask)
        mean_neg_post = masked_mean(post_edit_token_log_probs, neg_mask)

        z_sent = (mean_pos_post - mean_neg_post).sigmoid()
        z_topic_raw = (mean_pos_post - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_sent = z_sent * z_topic

        return {
            "acc_sent": es_sent,
            "z_sent": z_sent,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_post,
            "wrong_probs": mean_neg_post,
        }


# DEPRECATED
def sent_success(pre_edit_probs, post_edit_probs, pos_mask, eps=torch.finfo(torch.float32).eps, batch_size=20):
    assert False, "No longer used"
    # content_score = post_edit_probs[pos_mask].prod() ** (1/pos_mask.sum()) / (pre_edit_probs[pos_mask]. + eps)
    post_pos_avg = post_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    pre_pos_avg = pre_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    content_score = post_pos_avg / (pre_pos_avg + eps)
    z_content = min(1., content_score)

    # compute z_sent through a weighting objective
    # normalized_probs = post_edit_probs / (post_edit_probs.sum() + eps)
    # balancing_factor = 0.5 * ((~pos_mask).float().sum() / pos_mask.float().sum() + 1)
    # z_sent_weight = balancing_factor * normalized_probs.dot(pos_mask.float())
    post_neg_avg = post_edit_probs[~pos_mask].prod() ** (1 / (~pos_mask).sum())
    neg_over_pos = post_neg_avg / (eps + post_pos_avg)
    z_sent_weight = 1 / (1 + neg_over_pos)

    # compute z_sent through a ranking objective
    batch_mask = pos_mask.view(-1, batch_size).long()
    sort_idxs = post_edit_probs.view(-1, batch_size).sort(-1, descending=True).indices
    ranked_mask = batch_mask.gather(1, sort_idxs)
    true_mask = batch_mask.sort(-1, descending=True).values
    z_sent_rank = (ranked_mask == true_mask).float().mean()

    # compute the final success scores
    weight_success = (z_content * z_sent_weight) ** 0.5
    rank_success = (z_content * z_sent_rank) ** 0.5

    correct_probs = post_edit_probs[pos_mask].mean()
    wrong_probs = post_edit_probs[~pos_mask].mean()

    return {
        "acc_weight": weight_success,
        "acc_rank": rank_success,
        "rank_score": z_sent_rank,
        "weight_score": z_sent_weight,
        "content_score": content_score,
        "post_edit_probs": post_edit_probs.sum(),
        "pre_edit_probs": pre_edit_probs.sum(),
        "correct_probs": correct_probs,
        "wrong_probs": wrong_probs
    }


# def sent_retain(pre_logits, post_logits, sent_mask, batch_size=20, eps=torch.finfo(torch.float32).eps):
#     pre_log_probs = pre_logits.log_softmax(-1).gather(-1, all_targ.unsqueeze(-1)).squeeze(-1)
#     post_log_probs = post_logits.log_softmax(-1).gather(-1, all_targ.unsqueeze(-1)).squeeze(-1)

#     pre_batch = pre_probs.view(-1, batch_size)
#     post_batch = post_probs.view(-1, batch_size)
#     mask_batch = sent_mask.view(-1, batch_size)

#     stats = []
#     for pre, post, mask in zip(pre_batch, post_batch, mask_batch):
#         avg_pre = pre.prod() ** (1 / pre.numel())
#         avg_post = post.prod() ** (1 / post.numel())
#         z_avg = min(avg_pre / avg_post, avg_post / avg_pre)

#         post_neg_avg = post[~mask].prod() ** (1 / (~mask).sum())
#         post_pos_avg = post[mask].prod() ** (1 / mask.sum())

#         pre_neg_avg = pre[~mask].prod() ** (1 / (~mask).sum())
#         pre_pos_avg = pre[mask].prod() ** (1 / mask.sum())

#         post_neg_over_pos = post_neg_avg / (eps + post_pos_avg)
#         pre_neg_over_pos = pre_neg_avg / (eps + pre_pos_avg)
#         z_post = 1 / (1 + post_neg_over_pos)
#         z_pre = 1 / (1 + pre_neg_over_pos)

#         z_sent = min(z_post / z_pre, z_pre / z_post)

#         stats.append((z_avg * z_sent) ** 0.5)

#     return sum(stats) / len(stats)


# For zsRE and F-NLI
def retain_rate(pre_logits, post_logits, mask=None):
    if pre_logits.shape[-1] == 1:
        pre_logits = pre_logits.squeeze(-1)
    if post_logits.shape[-1] == 1:
        post_logits = post_logits.squeeze(-1)

    assert pre_logits.shape == post_logits.shape
    assert pre_logits.shape[0] == mask.shape[0]

    if pre_logits.dim() == 1:
        # binary classification
        pre_preds = pre_logits > 0
        post_preds = post_logits > 0
        retain = (pre_preds == post_preds).float().mean()
    elif pre_logits.dim() == 3:
        # sequence modeling
        pre_preds = pre_logits.argmax(-1)
        post_preds = post_logits.argmax(-1)
        match = (pre_preds == post_preds) * mask
        retain = (match.sum(-1) == mask.sum(-1)).float().mean()
    else:
        raise NotImplementedError

    return retain.item()
