import torch
import torch.nn as nn
import torch.nn.functional as F
from inclearn.lib.logger import LOGGER
from inclearn.lib import distance as distance_lib
from sklearn.cluster import KMeans
from inclearn.lib import utils

logger = LOGGER.LOGGER


class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
            self,
            features_dim,
            device,
            *,
            proxy_per_class=1,
            distance="cosine",
            merging="softmax",
            scaling=1,
            gamma=1.,
            use_bias=False,
            type=None,
            pre_fc=None,
            negative_weights_bias=None,
            train_negative_weights=False,
            eval_negative_weights=False
    ):
        super().__init__()

        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma

        self.negative_weights_bias = negative_weights_bias
        self.train_negative_weights = train_negative_weights
        self.eval_negative_weights = eval_negative_weights

        self._negative_weights = None
        self.use_neg_weights = True

        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            assert False

        if proxy_per_class > 1:
            logger.info("Using {} proxies per class.".format(proxy_per_class))

        if pre_fc is not None:
            self.pre_fc = nn.Sequential(
                nn.ReLU(inplace=True), nn.BatchNorm1d(self.features_dim),
                nn.Linear(self.features_dim, pre_fc)
            )
            self.features_dim = pre_fc
        else:
            self.pre_fc = None

        self._task_idx = 0

    def on_task_end(self):
        self._task_idx += 1

    def on_epoch_end(self):
        pass

    def forward(self, features):
        if hasattr(self, "pre_fc") and self.pre_fc is not None:
            features = self.pre_fc(features)

        weights = self.weights
        if self._negative_weights is not None and (
                self.training is True or self.eval_negative_weights
        ) and self.use_neg_weights:
            weights = torch.cat((weights, self._negative_weights), 0)

        if self.distance == "cosine":
            raw_similarities = distance_lib.cosine_similarity(features, weights)
        elif self.distance == "stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_neg_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

            if self._negative_weights is not None and self.negative_weights_bias is not None and \
                    self.training is True:
                qt = self._negative_weights.shape[0]
                if isinstance(self.negative_weights_bias, float):
                    similarities[..., -qt:] = torch.clamp(
                        similarities[..., -qt:] - self.negative_weights_bias, min=0
                    )
                elif isinstance(
                        self.negative_weights_bias, str
                ) and self.negative_weights_bias == "min":
                    min_simi = similarities[..., :-qt].min(dim=1, keepdim=True)[0]
                    similarities = torch.min(
                        similarities,
                        torch.cat((similarities[..., :-qt], min_simi.repeat(1, qt)), dim=1)
                    )
                elif isinstance(
                        self.negative_weights_bias, str
                ) and self.negative_weights_bias == "max":
                    max_simi = similarities[..., :-qt].max(dim=1, keepdim=True)[0] - 1e-6
                    similarities = torch.min(
                        similarities,
                        torch.cat((similarities[..., :-qt], max_simi.repeat(1, qt)), dim=1)
                    )
                elif isinstance(self.negative_weights_bias,
                                str) and self.negative_weights_bias.startswith("top_"):
                    topk = int(self.negative_weights_bias.replace("top_", ""))
                    botk = min(qt - topk, qt)

                    indexes = (-similarities[..., -qt:]).topk(botk, dim=1)[1]
                    similarities[..., -qt:].scatter_(1, indexes, 0.)
                else:
                    raise NotImplementedError(f"Unknown {self.negative_weights_bias}.")

        return {"logits": similarities, "raw_logits": raw_similarities}

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

    # ------------------
    # Weights management
    # ------------------

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        if len(self._weights) == 1:
            return

        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_weights_i_to_j(self, indexes_i, indexes_j):
        with torch.no_grad():
            base_weights = self.weights[indexes_i]

            old_norm = torch.mean(base_weights.norm(dim=1))
            new_norm = torch.mean(self.weights[indexes_j].norm(dim=1))

            self.weights[indexes_j] = nn.Parameter((old_norm / new_norm) * self.weights[indexes_j])

    def align_inv_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((new_norm / old_norm) * self._weights[-1])

    @property
    def weights(self):
        return torch.cat([clf for clf in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(self.proxy_per_class * n_classes, self.features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        self._weights.append(new_weights)

        self.to(self.device)
        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
            self, class_indexes, inc_dataset, network, multi_class_diff="normal", type=None
    ):
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)
            else:
                if multi_class_diff == "normal":
                    std = torch.std(features_normalized, dim=0)
                    for _ in range(self.proxy_per_class):
                        new_weights.append(torch.normal(class_embeddings, std) * avg_weights_norm)
                elif multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.numpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(torch.tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".
                        format(multi_class_diff)
                    )

        new_weights = torch.stack(new_weights)
        self._weights.append(nn.Parameter(new_weights))

        self.to(self.device)
        self.n_classes += len(class_indexes)

        return self

    def set_negative_weights(self, negative_weights, ponderate=False):
        """Add weights that are used like the usual weights, but aren't actually
        parameters.

        :param negative_weights: Tensor of shape (n_classes * nb_proxy, features_dim)
        :param ponderate: Reponderate the negative weights by the existing weights norm, as done by
                          "Weights Imprinting".
        """
        logger.info("Add negative weights.")
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                negative_weights = negative_weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_negative_weights_norm
                negative_weights = negative_weights * ratio
            elif ponderate == "inv_align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_negative_weights_norm / avg_weights_norm
                negative_weights = negative_weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        if self.train_negative_weights:
            self._negative_weights = nn.Parameter(negative_weights)
        else:
            self._negative_weights = negative_weights
