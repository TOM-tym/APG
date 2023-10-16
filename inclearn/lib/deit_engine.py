# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import pprint
import sys
from typing import Iterable, Optional

import numpy
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from inclearn.lib.losses import DistillationLoss
from inclearn.lib.losses.base import embeddings_similarity
import inclearn.lib.utils as utils
import logging

# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER
from inclearn.lib.metrics import accuracy_per_task
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from inclearn.lib.losses.base import SoftTripletLossWithKLDiv

logger = LOGGER.LOGGER


class Mycontex(object):
    def __init__(self, ):
        pass

    def __enter__(self):
        pass

    def do_self(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def train_one_epoch_pretrain(model: torch.nn.Module,  # ddp for classifier
                             data_loader: Iterable,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device, epoch: int,
                             loss_scaler: utils.MyScaler,
                             max_norm: float = 0,
                             mixup_fn: Optional[Mixup] = None,
                             teacher_distillation_loss=None,
                             teacher_model=None,
                             ):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    itx = 0
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    for data in metric_logger.log_every(data_loader, print_freq, header):
        itx += 1
        samples = data['inputs']
        targets = data['targets']
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        model.train()

        with torch.cuda.amp.autocast():
            outputs, features_new = model(samples, feat=True)
            teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = teacher_model(samples)
                teacher_output_logits = outputs_teacher['logits']
                _teacher_targets = teacher_output_logits.argmax(1)
            _tea_dis_loss = F.cross_entropy(outputs, _teacher_targets)
            if teacher_distillation_loss is not None:
                with torch.no_grad():
                    outputs_teacher = teacher_model(samples)
                    teacher_output_logits = outputs_teacher['logits']
                _tea_dis_loss += teacher_distillation_loss(outputs, teacher_output_logits * 3) * 70
                # loss_new_ce = criterion(outputs, targets)

            loss = _tea_dis_loss

        loss_value = loss.item()

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        loss_scaler.loss_backward(loss, create_graph=is_second_order)
        loss_scaler(optimizer, clip_grad=max_norm, parameters=None)

        torch.distributed.barrier()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{str(metric_logger)}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_adaptive_prompt_generator(model: torch.nn.Module,  # ddp for classifier
                                              criterion,
                                              adaptive_prompt_generator: torch.nn.Module,  # it should be in ddp
                                              data_loader: Iterable,
                                              optimizer: torch.optim.Optimizer,
                                              device: torch.device, epoch: int,
                                              loss_scaler: utils.MyScaler,
                                              max_norm: float = 0,
                                              mixup_fn: Optional[Mixup] = None,
                                              criterion1=None,
                                              triplet_criterion=None,
                                              feature_triplet_criterion=None,
                                              old_new_feature_triplet_criterion=None,
                                              center_lib_loader=None,
                                              use_center_constraint=True,
                                              train_backbone_model=False,
                                              use_apg=True,
                                              old_model: torch.nn.Module = None,
                                              old_APG=None,
                                              parameter_dict=None,
                                              taskid=0,
                                              norm_constraint=False,
                                              p_attn=False,
                                              prompt_per_cls=5,
                                              ):
    if parameter_dict is None:
        parameter_dict = {}
    attn_type = parameter_dict.get('attn_type', 'hard_ce')
    attnConst_hard_tau = parameter_dict.get('hard_tau', 1)
    r = parameter_dict.get('attnConst_smoothing', 0.1)
    logger.info(f'Augments for current_epoch:\n {pprint.pformat(parameter_dict)}')
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if center_lib_loader is not None:
        center_lib = iter(center_lib_loader)

    criterion2 = torch.nn.CrossEntropyLoss()
    soft_triplet_withKL = SoftTripletLossWithKLDiv()
    itx = 0
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    for data in metric_logger.log_every(data_loader, print_freq, header):
        itx += 1
        samples = data['inputs']
        targets = data['targets']
        samples = samples.to(device, non_blocking=True)
        origin_t = targets = targets.to(device, non_blocking=True)
        prompts_new = None
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # Step 1. generate prompts
        # Step 1.1 Get low level representation of a batch of images.
        if not train_backbone_model:
            model.eval()
        adaptive_prompt_generator.train()

        loss_distill = loss1 = loss2 = loss3 = loss4 = triplet_loss = loss1_distill = loss_norm = prompts_attn_const = 0
        with torch.cuda.amp.autocast():
            if use_apg:
                with torch.no_grad():
                    _, low_level_representation = model(samples, return_immediate_feat=True, only_feat=True,
                                                        break_immediate=True)
                _low_level_representation = low_level_representation[:, 0].unsqueeze(1)
                # Step 1.2 Pass through the generator
                if center_lib_loader is not None and use_center_constraint:
                    with torch.no_grad():
                        try:
                            content = center_lib.__next__()
                        except:
                            center_lib = iter(center_lib_loader)
                            content = center_lib.__next__()
                        centroid = content['immediate']
                        prompts = content['prompts']
                        feat = content['feat']
                        target = content['targets']
                        aug_feat = content['feat_aug']
                        centroid = centroid.unsqueeze(1).cuda()
                        prompts = prompts.cuda()
                        feat = feat.cuda()
                        aug_feat = aug_feat.cuda()
                        target = target.long().cuda()
                    center_prompts, prompts_attn_old = adaptive_prompt_generator(centroid, return_attn=True)

                    p_loss1 = parameter_dict.get('constraint', 1)
                    loss1 = criterion1(center_prompts, prompts) * p_loss1

                    logit_from_cent = model.module.classifier(aug_feat)

                    p_ce_c = parameter_dict.get('cross_entropy_mem', 1)
                    loss3 = criterion2(logit_from_cent, target) * p_ce_c

                prompts_new, prompts_attn_new = adaptive_prompt_generator(_low_level_representation, return_attn=True)
                with torch.no_grad():
                    other_token_norm = low_level_representation.view(-1, low_level_representation.shape[-1]).norm(1,
                                                                                                                  dim=-1)
                p_norm = parameter_dict.get('loss_norm', 1)
                if norm_constraint and p_norm > 0:
                    p_norm_scale = parameter_dict.get('loss_norm_scale', 1)
                    loss_norm = F.relu(
                        prompts_new.norm(p=1, dim=-1).mean() - other_token_norm.mean() * p_norm_scale) * p_norm
                    # metric_logger.update(prompts_norm=prompts_new.norm(p=1, dim=-1).mean().item())
                if p_attn:
                    N, H, _, I = prompts_attn_new.shape
                    N2 = origin_t.shape[0]
                    if not adaptive_prompt_generator.module.cross_attn_wo_x:
                        prompts_attn_new = prompts_attn_new.reshape(N * H, -1)[:, 1:]
                    else:
                        prompts_attn_new = prompts_attn_new.reshape(N * H, -1)

                    if attn_type == 'hard_ce':
                        prompts_constraint_target = origin_t.expand(H * max(1, int(N / N2)), -1).T.reshape(N * H)
                        prompts_attn_const = criterion2(prompts_attn_new / attnConst_hard_tau,
                                                        prompts_constraint_target)
                        assert prompt_per_cls == 1, 'Hard_ce should be with prompt_per_clas == 1.'
                    else:
                        assert False, f'got attn_type: [{attn_type}]'
                else:
                    prompts_attn_const = 0

                prompts_to_triplet = prompts_new
                # for more than one prompts
                B, H, D = prompts_to_triplet.shape
                B2 = origin_t.shape[0]
                prompts_new_flatten = prompts_to_triplet.reshape(-1, D)
                targets_to_triplet = origin_t.expand(H * max(1, int(B / B2)), -1).T.reshape(B * H)

                if center_lib_loader is not None and use_center_constraint:
                    B, H, D = center_prompts.shape
                    B2 = target.shape[0]
                    center_prompts_flatten = center_prompts.reshape(-1, D)
                    center_target_to_triplet = target.expand(H * max(1, int(B / B2)), -1).T.reshape(B * H)
                    prompts_new_flatten = torch.cat((prompts_new_flatten, center_prompts_flatten), dim=0)
                    targets_to_triplet = torch.cat((targets_to_triplet, center_target_to_triplet), dim=0)

                if triplet_criterion is not None:
                    p_triplet = parameter_dict.get('triplet', 1)
                    triplet_loss = triplet_criterion(prompts_new_flatten.squeeze(1).float(),
                                                     targets_to_triplet) * p_triplet
                outputs, features_new = model(low_level_representation, feat=True, extra_tokens=prompts_new,
                                              shortcut=True)
            else:
                outputs, features_new = model(samples, feat=True)
            if old_model is not None:
                old_model.eval()
                outputs_old_model, features_new_old_model = old_model(samples, feat=True, extra_tokens=prompts_new)
                loss_distill = embeddings_similarity(features_new_old_model.detach(), features_new) * 10
                metric_logger.update(loss_distill=loss_distill.item())
            p_ce = parameter_dict.get('cross_entropy', 1)
            tau = parameter_dict.get('tau', [1 for _ in range(taskid + 1)])[taskid]
            loss_new_ce = criterion(outputs / tau, targets) * p_ce

            old_new_feat_triplet = 0
            if old_new_feature_triplet_criterion is not None:
                p_old_new_feat_tri = parameter_dict.get('old_new_feat_triplet', 1)
                if center_lib_loader is not None and use_center_constraint:
                    features_tri = torch.cat((features_new, feat), dim=0)
                    target_tri = torch.cat((origin_t, target), dim=0)
                    old_new_feat_triplet = old_new_feature_triplet_criterion(features_tri,
                                                                             target_tri) * p_old_new_feat_tri

            if feature_triplet_criterion is not None:
                p_feat_triplet = parameter_dict.get('feat_triplet', 1)
                feat_triplet = feature_triplet_criterion(features_new, origin_t) * p_feat_triplet
            else:
                feat_triplet = 0
            loss = loss_new_ce + loss1 + loss2 + loss3 + triplet_loss + feat_triplet + old_new_feat_triplet \
                   + loss1_distill + loss_distill + loss_norm + prompts_attn_const

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.warning("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.distributed.barrier()
        torch.cuda.synchronize()
        optimizer.zero_grad()
        loss_scaler.loss_backward(loss, create_graph=is_second_order)
        loss_scaler(optimizer, clip_grad=max_norm, parameters=None)

        torch.distributed.barrier()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(new_ce=loss_new_ce.item())

        if isinstance(loss_norm, torch.Tensor):
            metric_logger.update(loss_norm=loss_norm.item())

        if center_lib_loader is not None and use_center_constraint:
            metric_logger.update(loss_c=loss1.item())
            if old_APG is not None:
                metric_logger.update(loss_c_dis=loss1_distill.item())
        if loss3 is not None and isinstance(loss3, torch.Tensor):
            metric_logger.update(ce_cent=loss3.item())
        if loss4 is not None and isinstance(loss4, torch.Tensor):
            metric_logger.update(mlp_out=loss4.item())
        if triplet_criterion is not None and isinstance(triplet_loss, torch.Tensor):
            metric_logger.update(tri=triplet_loss.item())
        if feature_triplet_criterion is not None and isinstance(feat_triplet, torch.Tensor):
            metric_logger.update(feat_tri=feat_triplet.item())
        if isinstance(old_new_feat_triplet, torch.Tensor):
            metric_logger.update(ow_feat_tri=old_new_feat_triplet.item())
        if isinstance(prompts_attn_const, torch.Tensor):
            metric_logger.update(p_attn=prompts_attn_const.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    if prompts_new is not None:
        logger.info(f'Current prompts norm {prompts_new.norm(p=1, dim=-1).mean().item()}')
    logger.info(f"Averaged stats:{str(metric_logger)}")
    logger.info(
        f'lr: {[i["lr"] for i in optimizer.param_groups]} wd: {[i["weight_decay"] for i in optimizer.param_groups]}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def fine_tune_adaptive_prompt_generator(model: torch.nn.Module,  # ddp for classifier
                                        center_lib_loader,
                                        adaptive_prompt_generator: torch.nn.Module,  # it should be in ddp
                                        optimizer: torch.optim.Optimizer,
                                        epoch: int,
                                        loss_scaler: utils.MyScaler,
                                        max_norm: float = 0,
                                        criterion1=None,
                                        memory_loader=None,
                                        finetune_whole_fc=False,
                                        old_APG=None,
                                        constraint_APG=True,
                                        parameter_dict=None,
                                        prompt_tri=None,
                                        p_attn=False,
                                        prompt_per_cls=5,
                                        num_classes=0,
                                        ):
    if parameter_dict is None:
        parameter_dict = {}
    logger.info(f'Augments for current_epoch:\n {pprint.pformat(parameter_dict)}')
    attn_type = parameter_dict.get('attn_type', 'hard_ce')
    attnConst_hard_tau = parameter_dict.get('hard_tau', 1)
    r = parameter_dict.get('attnConst_smoothing', 0.1)
    logger.info(f'LSCE at {r}')
    LSCE = LabelSmoothingCrossEntropy(smoothing=r)
    soft_ce = SoftTargetCrossEntropy()
    criterion2 = torch.nn.CrossEntropyLoss()
    soft_target_ce = SoftTargetCrossEntropy()
    more_cls_p_mapping = dict(
        {i: [j for j in range(i * prompt_per_cls, (i + 1) * prompt_per_cls)] for i in range(num_classes)})
    soft_triplet_withKL = SoftTripletLossWithKLDiv()
    if not finetune_whole_fc:
        model.module.classifier.mlp_eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if memory_loader is not None:
        mem_iter = iter(memory_loader)

    criterion2 = torch.nn.CrossEntropyLoss()
    itx = 0
    mem_ce = 0
    loss3 = 0
    loss1_distill = 0
    loss_prompt_triplet = 0
    prompts_attn_const = 0
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    for data in metric_logger.log_every(center_lib_loader, print_freq, header):
        if memory_loader is not None:
            try:
                mem = mem_iter.__next__()
            except:
                mem_iter = iter(memory_loader)
                mem = mem_iter.__next__()
            mem_sample = mem['inputs']
            mem_targets = mem['targets']
        itx += 1
        # model.eval()
        # loss1 = loss2 = triplet_loss = 0
        centroid = data['immediate']
        aug_feat = data['feat_aug']
        prompts = data['prompts']
        feat = data['feat']
        feat_mean = data['feat_mean']
        target = data['targets']
        # Step 1.2 Pass through the generator
        with torch.cuda.amp.autocast():
            # centroid = centroid.unsqueeze(1).cuda()
            prompts = prompts.cuda()
            feat = feat.cuda()
            target = target.long().cuda()
            centroid = centroid.cuda()
            aug_feat = aug_feat.cuda()
            feat_mean = feat_mean.cuda()

            centroid = centroid.unsqueeze(1)
            center_prompts, prompts_attn_new = adaptive_prompt_generator(centroid, return_attn=True)
            if p_attn:
                N, H, _, I = prompts_attn_new.shape
                if not adaptive_prompt_generator.module.cross_attn_wo_x:
                    prompts_attn_new = prompts_attn_new.reshape(N * H, -1)[:, 1:]
                else:
                    prompts_attn_new = prompts_attn_new.reshape(N * H, -1)
                prompts_constraint_target = target.expand(H, -1).T.reshape(N * H)
                if attn_type == 'hard_ce':
                    prompts_constraint_target = target.expand(H, -1).T.reshape(N * H)
                    prompts_attn_const = criterion2(prompts_attn_new / attnConst_hard_tau,
                                                    prompts_constraint_target)
                    assert prompt_per_cls == 1, 'Hard_ce should be with prompt_per_clas == 1.'

            else:
                prompts_attn_const = 0
            prompts_attn_p = parameter_dict.get('prompts_attn_p', 1)
            prompts_attn_const *= prompts_attn_p
            if prompt_tri is not None:
                prompt_tri_p = parameter_dict.get("prompt_tri", 1)
                loss_prompt_triplet = prompt_tri(center_prompts.float().squeeze(1), target) * prompt_tri_p
            if old_APG is not None:
                center_prompts_from_old_model = old_APG(centroid)
                distill_p = parameter_dict.get('distill_p', 1)
                loss1_distill = criterion1(center_prompts, center_prompts_from_old_model) * distill_p
                metric_logger.update(distill=loss1_distill.item())
            if constraint_APG:
                loss1_param = parameter_dict.get('constraint', 1)
                loss1 = criterion1(center_prompts, prompts) * loss1_param
                metric_logger.update(loss_c=loss1.item())
            else:
                loss1 = 0
            if finetune_whole_fc:
                logit, generated_aug_feat = model.module.classifier(aug_feat, return_immediate_feature=True)
            else:
                with torch.no_grad():
                    _, generated_aug_feat = model.module.classifier(aug_feat, return_immediate_feature=True)
                logit = model.module.classifier.forward_last_FC(generated_aug_feat)
            ce_p = parameter_dict.get('cross_entropy', 1)
            loss2 = criterion2(logit, target) * ce_p
            if memory_loader is not None:
                mem_sample = mem_sample.cuda()
                mem_targets = mem_targets.cuda()
                with torch.no_grad():
                    _, low_level_representation = model(mem_sample, only_feat=True, return_immediate_feat=True,
                                                        break_immediate=True)
                    _low_level_representation = low_level_representation[:, 0].unsqueeze(1)
                    prompts_new = adaptive_prompt_generator(_low_level_representation)
                outputs, features_new = model(low_level_representation, feat=True, extra_tokens=prompts_new)
                mem_ce_p = parameter_dict.get('mem_ce', 1)
                mem_ce = criterion2(outputs, mem_targets) * mem_ce_p
            loss = loss1 + loss1_distill + loss2 + mem_ce + loss_prompt_triplet + prompts_attn_const

        loss_value = loss.item()

        is_all_loss_normal = torch.tensor(0).cuda()
        if not math.isfinite(loss_value):
            logger.warning("Loss is {}, skip iter {}".format(loss_value, itx))
            logger.warning(f''
                           f'\rloss1 {loss1} '
                           f'\rloss2 {loss2}'
                           f'\rloss3 {loss3}'
                           f'\rtriplet_loss {mem_ce}'
                           f'\rloss1_distill {loss1_distill}')

            is_all_loss_normal += 1
            optimizer.zero_grad()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            continue
            # sys.exit(1)
        torch.distributed.all_reduce(is_all_loss_normal, op=torch.distributed.ReduceOp.SUM)
        if is_all_loss_normal > 0:
            torch.distributed.barrier()
            torch.cuda.synchronize()
            logger.info(f'One of the loss got NaN, skip iter {itx}')
            continue
        torch.distributed.barrier()
        torch.cuda.synchronize()

        optimizer.zero_grad()
        loss_scaler.loss_backward(loss, create_graph=is_second_order)
        loss_scaler(optimizer, clip_grad=max_norm, parameters=None)

        torch.distributed.barrier()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if center_lib_loader is not None:
            metric_logger.update(loss_ce_c=loss2.item())
        if memory_loader is not None:
            metric_logger.update(mem_ce=mem_ce.item())
        if isinstance(loss_prompt_triplet, torch.Tensor):
            metric_logger.update(prompt_tri=loss_prompt_triplet.item())
        if isinstance(prompts_attn_const, torch.Tensor):
            metric_logger.update(p_attn=prompts_attn_const.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{str(metric_logger)}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, APG=None, extra_tokens=None, output_key=None, init_task=0, task_size=5):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # torch.distributed.barrier()
    all_pred = torch.tensor([])
    all_targets = torch.tensor([])
    for data in metric_logger.log_every(data_loader, 10, header):
        images = data['inputs']
        targets = data['targets']
        images = images.to(device, non_blocking=True)
        target = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if extra_tokens is not None:
                # et = torch.stack([extra_tokens[str(i.item())] for i in target]).unsqueeze(1)
                et = extra_tokens.expand(len(target), -1, -1)
                output = model(images, extra_tokens=et)
            elif APG is not None:
                APG.eval()
                _, low_level_feat = model.backbone.forward_features(images, return_immediate_feat=True,
                                                                  break_immediate=True)
                # low_level_feat = low_level_feat.mean(dim=1).unsqueeze(1)
                _low_level_feat = low_level_feat[:, 0].unsqueeze(1)
                # low_level_feat = low_level_feat.mean()
                prompts = APG(_low_level_feat)
                output = model(low_level_feat, extra_tokens=prompts, shortcut=True)
            else:
                output = model(images)
            if output_key is not None:
                output = output[output_key]
            loss = criterion(output, target)
        all_pred = torch.cat((all_pred, output.cpu()))
        all_targets = torch.cat((all_targets, targets.cpu()))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    all_pred = numpy.asarray(all_pred)
    all_targets = numpy.asarray(all_targets).astype(int)
    all_acc = accuracy_per_task(all_pred, all_targets, init_task_size=init_task, task_size=task_size, topk=1)
    logger.info(f'---' * 50)
    logger.info(f'Acc with APG[{APG is not None}]:\n\t{all_acc}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
