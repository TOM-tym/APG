import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T)
    return T


def cross_entropy_teacher_confidence(similarities, targets, old_confidence, memory_indexes):
    memory_indexes = memory_indexes.byte()

    per_sample_losses = F.cross_entropy(similarities, targets, reduction="none")

    memory_losses = per_sample_losses[memory_indexes]
    new_losses = per_sample_losses[~memory_indexes]

    memory_old_confidence = old_confidence[memory_indexes]
    memory_targets = targets[memory_indexes]

    right_old_confidence = memory_old_confidence[torch.arange(memory_old_confidence.shape[0]),
                                                 memory_targets]
    hard_indexes = right_old_confidence.le(0.5)

    factors = 2 * (1 + (1 - right_old_confidence[hard_indexes]))

    loss = torch.mean(
        torch.cat(
            (new_losses, memory_losses[~hard_indexes], memory_losses[hard_indexes] * factors)
        )
    )

    return loss


def nca(
        similarities,
        targets,
        class_weights=None,
        focal_gamma=None,
        scale=1,
        margin=0.,
        exclude_pos_denominator=True,
        hinge_proxynca=False,
        memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )


def ucir_ranking(logits, targets, n_classes, task_size, nb_negatives=2, margin=0.2):
    """Hinge loss from UCIR.

    Taken from: https://github.com/hshustc/CVPR19_Incremental_Learning

    # References:
        * Learning a Unified Classifier Incrementally via Rebalancing
          Hou et al.
          CVPR 2019
    """
    gt_index = torch.zeros(logits.size()).to(logits.device)
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
    gt_scores = logits.masked_select(gt_index)
    # get top-K scores on novel classes
    num_old_classes = logits.shape[1] - task_size
    max_novel_scores = logits[:, num_old_classes:].topk(nb_negatives, dim=1)[0]
    # the index of hard samples, i.e., samples of old classes
    hard_index = targets.lt(num_old_classes)
    hard_num = torch.nonzero(hard_index).size(0)

    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
                                                   max_novel_scores.view(-1, 1),
                                                   torch.ones(hard_num * nb_negatives).to(logits.device))
        return loss

    return torch.tensor(0).float()


class PromptSimilarity(nn.Module):
    def __init__(self, type='l2'):
        super().__init__()
        self.type = type

    def forward(self, generated_prompt, target_prompt):
        if self.type == 'l2':
            dim = generated_prompt.shape[-1]
            generated_prompt = generated_prompt.reshape(-1, dim)
            target_prompt = target_prompt.reshape(-1, dim)
            return F.mse_loss(generated_prompt, target_prompt)
        elif self.type == 'cos':
            dim = generated_prompt.shape[-1]
            generated_prompt = generated_prompt.reshape(-1, dim)
            target_prompt = target_prompt.reshape(-1, dim)
            return (1 - F.cosine_similarity(generated_prompt, target_prompt)).mean()
        elif self.type == 'l1':
            # return F.l1_loss(generated_prompt, target_prompt, reduction='sum') / len(generated_prompt)
            dim = generated_prompt.shape[-1]
            generated_prompt = generated_prompt.reshape(-1, dim)
            target_prompt = target_prompt.reshape(-1, dim)
            return F.l1_loss(generated_prompt, target_prompt)
        elif self.type == 'euc':
            return torch.sqrt(torch.sum((generated_prompt - target_prompt) ** 2, dim=-1)).mean()
        else:
            raise NotImplementedError


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999.) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    # we change 999999 here to 9999 because of the fp16.
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999.) * (mat_similarity), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


def _batch_hard_softlabel(mat_distance, mat_similarity, indice=False):
    soft_label_mask = (mat_similarity > mat_similarity.mean()).float()
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999.) * (1 - soft_label_mask), dim=1,
                                                       descending=True)
    # we change 999999 here to 9999 because of the fp16.
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999.) * (soft_label_mask), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


import math


class SoftTripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.dist = euclidean_dist

    def forward(self, emb1, label):
        emb2 = emb1
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = self.dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        # label = (label > label.mean()).float()
        # mat_sim = label.mm(label.t())  # for soft laabel
        # mat_sim = (label > label.mean()).float()

        dist_ap, dist_an, ap_idx, an_idx = self._batch_hard_softlabel(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = self.dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss

    @staticmethod
    def _batch_hard_softlabel(mat_distance, mat_similarity, indice=False):
        # soft_label_mask = (mat_similarity > mat_similarity.mean()).float()
        sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999.) * (1 - mat_similarity), dim=1,
                                                           descending=True)
        # we change 999999 here to 9999 because of the fp16.
        hard_p = sorted_mat_distance[:, 0]
        hard_p_indice = positive_indices[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999.) * (mat_similarity), dim=1,
                                                           descending=False)
        hard_n = sorted_mat_distance[:, 0]
        hard_n_indice = negative_indices[:, 0]
        if (indice):
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n


class JSD(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def pairwise_KLDiv(input_q, target_p, log_softmax_input=False, softmax_target=False):
    eps = 1e-7  # for numerical stability
    if not log_softmax_input:  # if the passing input_q is not in the form of log_softmax, ...
        input_q = F.log_softmax(input_q, dim=-1)
    if not softmax_target:  # if the passing target_q is not in the form of softmax, ...
        target_p = F.softmax(target_p, dim=-1)

    batch_klB = (target_p * (target_p + eps).log()).sum(dim=1) - torch.einsum('ik, jk -> ij', input_q, target_p)
    return batch_klB


class SoftTripletLossWithKLDiv(SoftTripletLoss):
    def __init__(self, *args, **kwargs):
        super(SoftTripletLossWithKLDiv, self).__init__(*args, **kwargs)
        self.dist = pairwise_KLDiv


class HardLabelSoftTripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False):
        super(HardLabelSoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature

    def forward(self, emb1, label):
        emb2 = emb1
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = self._batch_hard_softlabel(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss

    @staticmethod
    def _batch_hard_softlabel(mat_distance, mat_similarity, indice=False):
        sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999.) * (1 - mat_similarity), dim=1,
                                                           descending=True)
        # we change 999999 here to 9999 because of the fp16.
        hard_p = sorted_mat_distance[:, 0]
        hard_p_indice = positive_indices[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999.) * (mat_similarity), dim=1,
                                                           descending=False)
        hard_n = sorted_mat_distance[:, 0]
        hard_n_indice = negative_indices[:, 0]
        if (indice):
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n


class TripletLoss(nn.Module):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)
        # mat_dist = cosine_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        # mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        mat_sim = label.mm(label.t())  # for soft label

        dist_ap, dist_an = _batch_hard_softlabel(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


if __name__ == '__main__':
    loss = JSD(reduction='none')
    a = torch.rand(128, 50)
    b = torch.rand(128, 50)
    out = loss(a, a).sum(dim=-1)
    pass
