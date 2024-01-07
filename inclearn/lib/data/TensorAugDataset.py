import torch
from torch.utils.data import Dataset
from torch.distributions import Distribution
from typing import Dict

_config_num_samples = {
    'cifar100': 500,
    'imagenet100': 1300,
    'imagenetr': 150,
    'Eurosat_rgb': 150,
    'nwpu-resisc45': 150,
}


class TensorAugDataset(Dataset):

    def __init__(self, dist_immediate: Dict[int, Distribution],
                 dist_feat: Dict[int, Distribution],
                 dist_aug_feat: Dict[int, Distribution],
                 dist_aug_feat_all: Dict[int, torch.Tensor],
                 feat_mean: Dict[int, torch.Tensor],
                 all_targets,
                 all_prompts,
                 dataset_name,
                 n=1,
                 new_classes=None):
        self.dist_immediate = dist_immediate
        self.dist_feat = dist_feat
        self.dist_aug_feat = dist_aug_feat
        self.dist_aug_feat_all = dist_aug_feat_all
        self.feat_mean = feat_mean
        self.all_prompts = all_prompts
        self.all_targets = all_targets
        self.n = n
        self.new_classes = new_classes
        self.repeat_new_classes = False
        self.num_sample_per_class = _config_num_samples[dataset_name.lower()]
        assert len(self.all_prompts) == len(self.all_targets)
        self.num_cls = len(self.all_targets.unique())

    def on_repeat_new_classes(self):
        assert self.new_classes is not None
        self.repeat_new_classes = True

    def off_repeat_new_classes(self):
        self.repeat_new_classes = False

    def __len__(self):
        return self.num_sample_per_class * self.num_cls

    def __getitem__(self, idx):
        idx = idx % len(self.all_targets)
        y = self.all_targets[idx]
        # prompts = self.all_prompts[y.item()].sample()
        prompts = torch.tensor([])
        immediate = torch.tensor([])
        feat = torch.tensor([])
        feat_mean = torch.tensor([])
        feat_aug = torch.tensor([])
        feat_aug_all = torch.tensor([])
        y_out = torch.tensor([y for _ in range(self.n)])
        for _ in range(self.n):
            # randidx = torch.randint(len(self.dist_aug_feat_all[y.item()]), size=(1,))
            all_promtps_mean = self.all_prompts[y.item()]
            all_promtps_mean = torch.stack(all_promtps_mean)
            prompts = torch.cat((prompts, all_promtps_mean), dim=0)
            immediate = torch.cat((immediate, self.dist_immediate[y.item()].sample().to(torch.float32).unsqueeze(0)),
                                  dim=0)
            feat = torch.cat((feat, self.dist_feat[y.item()].sample().to(torch.float32).unsqueeze(0)), dim=0)
            feat_mean = torch.cat((feat_mean, self.feat_mean[y.item()].unsqueeze(0)), dim=0)
            feat_aug = torch.cat((feat_aug, self.dist_aug_feat[y.item()].sample().to(torch.float32).unsqueeze(0)),
                                 dim=0)
            # feat_aug_all = torch.cat((feat_aug_all, self.dist_aug_feat_all[y.item()][randidx].unsqueeze(0)), dim=0)
            # feat = torch.cat((feat, self.dist_immediate[y.item()].sample().unsqueeze(0)), dim=0)
            # immediate
        if self.repeat_new_classes:
            new_cls_idx = torch.randperm(len(self.new_classes))[0]
            new_cls_idx = self.new_classes[new_cls_idx]
            y_out = torch.cat((y_out, torch.tensor([new_cls_idx for _ in range(self.n)])))
            for _ in range(self.n):
                all_promtps_mean = self.all_prompts[new_cls_idx]
                all_promtps_mean = torch.stack(all_promtps_mean)
                prompts = torch.cat((prompts, all_promtps_mean), dim=0)
                immediate = torch.cat(
                    (immediate, self.dist_immediate[new_cls_idx].sample().to(torch.float32).unsqueeze(0)),
                    dim=0)
                feat = torch.cat((feat, self.dist_feat[new_cls_idx].sample().to(torch.float32).unsqueeze(0)), dim=0)
                feat_mean = torch.cat((feat_mean, self.feat_mean[new_cls_idx].unsqueeze(0)), dim=0)
                feat_aug = torch.cat(
                    (feat_aug, self.dist_aug_feat[new_cls_idx].sample().to(torch.float32).unsqueeze(0)),
                    dim=0)
        return {"targets": y_out, "feat": feat, "immediate": immediate, 'prompts': prompts, 'feat_mean': feat_mean,
                'feat_aug': feat_aug, }
        # 'feat_aug_all': feat_aug_all}


def my_collect_fn(batch):
    targets = []
    feat = []
    immediate = []
    prompts = []
    feat_mean = []
    feat_aug = []
    # feat_aug_all = []
    for i in batch:
        targets.append(i['targets'])
        feat.append(i['feat'])
        immediate.append(i['immediate'])
        feat_mean.append(i['feat_mean'])
        prompts.append(i['prompts'])
        feat_aug.append(i['feat_aug'])
        # feat_aug_all.append(i['feat_aug_all'])
    targets = torch.cat(targets)
    feat = torch.cat(feat)
    immediate = torch.cat(immediate)
    prompts = torch.cat(prompts)
    feat_mean = torch.cat(feat_mean)
    feat_aug = torch.cat(feat_aug)
    # feat_aug_all = torch.cat(feat_aug_all)
    return {"targets": targets, "feat": feat, "immediate": immediate, 'prompts': prompts, 'feat_mean': feat_mean,
            'feat_aug': feat_aug, }
    # 'feat_aug_all': feat_aug_all}
