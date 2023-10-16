import collections
import copy
import logging
import os
import pickle
import pprint

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, losses, network, schedulers, utils
from inclearn.lib.losses import PromptSimilarity
from inclearn.models.base import IncrementalLearner
from inclearn.lib.losses.distillation_deit import DistillationLoss
from inclearn.lib.losses.base import SoftTripletLoss, TripletLoss, HardLabelSoftTripletLoss, euclidean_dist
from inclearn.lib.deit_engine import evaluate, train_one_epoch_adaptive_prompt_generator, \
    train_one_epoch_pretrain, fine_tune_adaptive_prompt_generator
from inclearn.lib.data.deit_datasets import build_transform
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma
import argparse
import json
import time, datetime
from inclearn.lib.utils import extract_transform_args
from inclearn.backbones.Self_Prompts import AdaptivePromptGenerator
from inclearn.lib.metrics import accuracy_per_task
from inclearn.lib.utils import MyScaler
from inclearn.lib.data.TensorAugDataset import TensorAugDataset, my_collect_fn
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

EPSILON = 1e-8

# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER

logger = LOGGER.LOGGER


class ContinualTrans(IncrementalLearner):
    """Training transformers in a continual manner.

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args: dict):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["opt"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]
        self._evaluation_type = args.get("eval_type", "icarl")

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]
        self.args = args  # type:dict
        self.clip_grad = args.get("clip_grad", None)
        self._result_folder = None
        self.testing_interval = args.get('testing_interval', 1)
        self.fc_proxy_per_class = args.get("fc_proxy_per_class", 1)

        self._network = network.BasicNet(
            args["backbones"],
            classifier_kwargs=args.get("classifier_config",
                                       {"type": "transclassifier",
                                        "use_bias": True,
                                        "proxy_per_class": self.fc_proxy_per_class}),
            device=self._device, extract_no_act=True, classifier_no_act=False,
            all_args=args,
            ddp=True)

        # teacher network for stage0 training
        if args.get('convnet', None) is not None:
            cnn_classifier_kwargs = args.get("cnn_classifier_config", {})
            self.teacher_network = network.BasicNet(
                args["convnet"],
                convnet_kwargs=args.get("convnet_config", {}),
                classifier_kwargs=cnn_classifier_kwargs,
                postprocessor_kwargs=args.get("cnn_postprocessor_config", {}),
                device=self._device,
                return_features=True,
                extract_no_act=True,
                classifier_no_act=args.get("classifier_no_act", True),
                all_args=args,
            )
            teacher_path = args.get('teacher_path', None)
            assert teacher_path is not None, 'Please specify the teacher path.'
            self.teacher_state_dict_path = teacher_path
        else:
            self.teacher_network = None

        self.smoothing = args.get("smoothing", 0)

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None
        self._old_APG = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._epoch_metrics = collections.defaultdict(list)
        self.Prompt_config = args.get("prompt_config", {})
        self.prompt_use_superclass = self.Prompt_config.get('prompt_use_superclass', {})
        self.prompt_super_class_n_prompts = self.prompt_use_superclass.get('prompt_super_class_n_prompts', 5)
        self.class_n_prompts = args.get('class_n_prompts', 1)
        self.mapping_origin2super = {}
        self.mapping_super2prompt = {}
        self.self_prompt = self.Prompt_config.get('self_prompt', False)
        self.use_finetune = args.get("use_finetune", True)
        self.APG_MLP_num = args.get('APG_MLP_num', 2)
        self.APG_num_heads = args.get('APG_num_heads', 4)
        self.APG_attn_drop = args.get('APG_attn_drop', 0)
        self.APG_proj_drop = args.get('APG_proj_drop', 0)
        self.center_lib_n = args.get('center_lib_n', 4)

        self.more_prompts = self.Prompt_config.get("more_prompts", 1)
        self.attn_depth = args.get("attn_depth", 1)
        self.use_update_tokens = args.get("use_update_tokens", False)
        self.cross_attn_wo_x = args.get("cross_attn_wo_x", False)
        self.APG_residual = args.get("APG_residual", True)
        if self.self_prompt:
            self.adaptive_prompt_generator = AdaptivePromptGenerator(embedding_dim=self._network.backbone.embed_dim,
                                                                     MLP_num=self.APG_MLP_num,
                                                                     num_heads=self.APG_num_heads,
                                                                     proj_drop=self.APG_proj_drop,
                                                                     attn_drop=self.APG_attn_drop,
                                                                     more_prompts=self.more_prompts,
                                                                     attn_depth=self.attn_depth,
                                                                     use_update_tokens=self.use_update_tokens,
                                                                     cross_attn_wo_x=self.cross_attn_wo_x,
                                                                     residual=self.APG_residual).cuda()
        else:
            self.adaptive_prompt_generator = None
        self.parameter_dict = self.args.get('parameter_dict', {})
        self.prompt_center_constraint = self.Prompt_config.get("center_constraint", False)
        self.use_center_constraint_in_APG = self.Prompt_config.get("use_center_constraint_in_APG", True)
        # self.Prompt_number_per_task = self.Prompt_config.get("num_prompt_per_task", 1)
        self.frozen_old_prompt = self.Prompt_config.get("frozen_old_prompt", False)
        # self.prompt_difference = self.Prompt_config.get("prompt_difference", False)
        self.center_library = {}
        self.lib_loader = None
        self.APG_nb_epochs = self.Prompt_config.get('train_MLP_epochs', 10)
        self.APG_lr_configs = self.Prompt_config.get("lr_config", {})
        if self.APG_lr_configs:
            self.APG_nb_epochs_stage0 = self.APG_lr_configs.get('epochs_stage0', 10)

            self.APG_lr_configs['scaled'] = False
            self.APG_lr_configs_args = argparse.Namespace(
                **self.APG_lr_configs)  # to co-operate with timm, convert dict into Namespace.
            self.immediate_layer = args.get('immediate_layer', 9)
            self.frozen_layer = args.get('frozen_layer', 9)
            self.reset_deep_layers = args.get('reset_deep_layers', True)
            self.force_reset_deep_layers = args.get('force_reset_deep_layers', False)
            # for resume
            self.APG_optimizer = create_optimizer(self.APG_lr_configs_args, self.adaptive_prompt_generator)
            self.APG_scheduler, _ = create_scheduler(self.APG_lr_configs_args, self.APG_optimizer)
            self.APG_loss_scaler = NativeScaler()

        self.APG_constraint_loss_type = self.Prompt_config.get('constraint_loss_type', 'l2')
        self.APG_norm_constraint = self.Prompt_config.get('APG_norm_constraint', True)
        self.APG_Prompt_triplet = self.Prompt_config.get('APG_Prompt_triplet', True)

        # finetune
        self.finetune_config = self.Prompt_config.get('finetune_config', {})
        self.resume_fc_after_finetune = self.finetune_config.get('resume_fc_after_finetune', False)
        self.finetune_constraint_APG = self.finetune_config.get('finetune_constraint_APG', True)
        self.finetune_parameter_dict = self.args.get('finetune_parameter_dict', {})
        self.finetune_whole_fc = self.finetune_config.get('finetune_whole_fc', False)
        self.finetune_lr_config = self.finetune_config.get("lr_config", {})
        self.finetune_start_epoch = self.finetune_config.get('start_epoch', 60)
        self.finetune_epochs = self.finetune_lr_config.get('epochs', 10)
        self.scale_finetune_epochs = self.finetune_config.get('scale_finetune_epochs', False)
        self.scale_finetune_epochs_per_time = self.finetune_config.get('scale_finetune_epochs_per_time', 10)
        self.finetune_task = self.finetune_config.get('finetune_task', [])
        self.finetune_repeat_new_cls = self.finetune_config.get('repeat_new_cls', False)
        if self.finetune_config:
            assert not (self.finetune_constraint_APG and not self.finetune_whole_fc)
            logger.info(f'Enable scale finetune: [{self.scale_finetune_epochs}],'
                        f'scale_finetune_epochs_per_time: [{self.scale_finetune_epochs_per_time}]')

            self.finetune_lr_config['scaled'] = False
            self.finetune_lr_config_args = argparse.Namespace(**self.finetune_lr_config)
            self.finetune_APG_optimizer = create_optimizer(self.finetune_lr_config_args, self.adaptive_prompt_generator)
            self.finetune_APG_scheduler, _ = create_scheduler(self.finetune_lr_config_args, self.finetune_APG_optimizer)
            logger.info(f'finetune_repeat_new_cls: [{self.finetune_repeat_new_cls}]')
            logger.info(f'Will finetune the model at the end of these tasks: {self.finetune_task}')
        else:
            self.finetune_APG_optimizer = None
            self.finetune_APG_scheduler = None

        # pretrain
        self.pretrain_config = args.get('pretrain_config', {})
        if not len(self.pretrain_config):
            self.pretrain_optimizer = None
            self.pretrain_scheduler = None
        else:
            self.pretrain_lr_config = self.pretrain_config.get("lr_config", {})
            self.pretrain_epochs = self.pretrain_lr_config.get('epochs', 150)

            self.pretrain_lr_config['scaled'] = False
            self.pretrain_lr_config_args = argparse.Namespace(**self.pretrain_lr_config)
            self.pretrain_optimizer = create_optimizer(self.pretrain_lr_config_args, self._network)
            self.pretrain_scheduler, _ = create_scheduler(self.pretrain_lr_config_args, self.pretrain_optimizer)

        if True:  # if ddp
            self.ddp = True

    def load_teacher_statedict(self, state_dict_path):
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location='cpu')
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                new_k = k
                if 'layer' in k:
                    new_k = k.replace('layer', 'stage_')
                if 'convnet' in k:
                    new_k = new_k.replace('convnet', 'backbone')
                if 'downsample.conv1x1' in k:
                    new_k = new_k.replace('downsample.conv1x1', 'downsample.0')
                if 'downsample.bn' in k:
                    new_k = new_k.replace('downsample.bn', 'downsample.1')
                # if 'downsample.0' in k:
                #     new_k = k.replace('downsample.0', 'downsample.conv1x1')
                # if 'downsample.1' in k:
                #     new_k = k.replace('downsample.1', 'downsample.bn')
                new_state_dict[new_k] = v
            errors = self.teacher_network.load_state_dict(new_state_dict, strict=False)
            logger.warning(f'{errors}')

    def set_old_prompt_frozen(self, frozen=True):
        assert self.adaptive_prompt_generator is not None, 'Not init the APG.'
        for prompt in self.adaptive_prompt_generator.all_tokens.get_old_tokens():
            prompt.requires_grad = not frozen

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def save_parameters(self, results_folder, run_id, epoch=-1, train_type='APG', end='False', prefix=''):
        if utils.is_main_process():
            path = os.path.join(results_folder, f"{prefix}net_{run_id}_task_{self._task}.pth")
            logger.info(f"Saving model at {path}.")
            save_content = {'backbone_fc': self.network.state_dict(),
                            'backbone_optim': self._optimizer.state_dict(),
                            'task_id': self._task, 'run_id': run_id, 'epoch': epoch,
                            'type': train_type,
                            'end': end
                            }
            if self.self_prompt:
                save_content.update({
                    'APG': self.adaptive_prompt_generator.state_dict(),
                    'APG_optim': self.APG_optimizer.state_dict(),
                    'APG_lr_scheduler': self.APG_scheduler.state_dict(),
                    'APG_scaler': self.APG_loss_scaler.state_dict(),
                })
            torch.save(save_content, path)

    def load_parameters(self, directory, run_id, force_from_stage0, path=None):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth") if path is None else path
        if not os.path.exists(path):
            assert FileNotFoundError, f"check your path, {path}"

        logger.info(f"Loading model at {path}.")
        load_content = torch.load(path, map_location=torch.device('cpu'))
        epoch = load_content['epoch']
        task_id = load_content['task_id']
        end = load_content.get('end', True)
        train_type = load_content.get('type', 'finetune')
        logger.info(f'Checkpoint status: \n'
                    f'\t| Checkpoint path | \t {directory}\n'
                    f'\t| Epoch           | \t {epoch}\n'
                    f'\t| Task ID         | \t {task_id}\n'
                    f'\t| End of task?    | \t {end}\n'
                    f'\t| training type   | \t {train_type}')
        if end:
            logger.info(f"The checkpoint is saved after a full stage, will start a new stage {task_id + 1}")
            resume_from_epoch = None
        else:
            logger.info(f"The checkpoint is saved in the middle of a training stage,"
                        f" will resume from epoch {epoch + 1} of stage {task_id}")
            resume_from_epoch = epoch

            # load backbone and fc
        logger.info('loading backbone fc')
        backbone_state_dict = load_content['backbone_fc']
        backbone_state_dict.pop('backbone.pos_embed_new')
        self.load_state_dict_force(self._network, backbone_state_dict)

        # load APG
        logger.info('loading APG')
        APG_state_dict = load_content['APG']

        if not force_from_stage0:
            self.load_state_dict_force(self.adaptive_prompt_generator, APG_state_dict)

        return resume_from_epoch, train_type

    @staticmethod
    def load_state_dict_force(model: nn.Module, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            logger.warning(
                f'Loading state dict with following errors: '
                f'\n\t Missing keys:--------------\n{pprint.pformat(missing_keys)}'
                f'\n\t unexpected_keys:-----------\n{pprint.pformat(unexpected_keys)}')
        else:
            logger.info('Successfully match all keys.')

    def load_prompts(self, directory, run_id, taskid):
        path = os.path.join(directory, f"prompts_{run_id}_task_{taskid}.pkl")

        logger.info("Loading prompt at {}.".format(path))
        with open(path, "rb") as f:
            Prompt = pickle.load(f)
            rank = torch.distributed.get_rank()
            with torch.no_grad():
                for idx, plist in Prompt:
                    if not len(plist):
                        continue
                    for ii in range(len(plist)):
                        Prompt[idx][ii] = Prompt[idx][ii].to(f'cuda:{rank}')
                self._network.backbone.Prompt = Prompt
            torch.cuda.empty_cache()

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    # ----------
    # Public API
    # ----------
    def get_groupwise_parameters(self):
        params = []
        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            groupwise_factor = self._groupwise_factors
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                logger.info(f"Group: {group_name}, lr: {self._lr * factor}.")
        return params

    def _before_task(self, train_loader, val_loader, dataset):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        if self.teacher_network is not None:
            self.teacher_network.add_classes(self._task_size)
        if self.Prompt_config:
            self.adaptive_prompt_generator.all_tokens.add_tokens(num_tokens=self._task_size * self.class_n_prompts,
                                                                 is_new_task=True)
            if self._task > 0:
                self.APG_lr_configs_args = utils.lr_scaler(self.APG_lr_configs_args, train_loader.batch_size,
                                                           convert=False)
                self.APG_optimizer = create_optimizer(self.APG_lr_configs_args, self.adaptive_prompt_generator)
                self.APG_scheduler, _ = create_scheduler(self.APG_lr_configs_args, self.APG_optimizer)

        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        namespace_args = utils.lr_scaler(self.args, train_loader.batch_size)
        if self._groupwise_factors:
            group_wise_params = self.get_groupwise_parameters()
            self._optimizer = create_optimizer(namespace_args, group_wise_params)
        else:
            self._optimizer = create_optimizer(namespace_args, self._network)
        self._scheduler, _ = create_scheduler(namespace_args, self._optimizer)

        # if args.distributed:
        if self.Prompt_config:
            if self._task > 0:
                # freeze the backbone
                self._network.backbone.set_backbone_frozen(num_layer=self.frozen_layer)
                logger.info(f'Set transformer block frozen until block [{self.frozen_layer}]')
                logger.info(
                    f'{pprint.pformat({i[0]: i[1].requires_grad for i in self.network.backbone.named_parameters()}, sort_dicts=False)}')
            if self.frozen_old_prompt and self._task > 0:
                self.set_old_prompt_frozen()
                logger.info(f'Set old prompts frozen.')

        if True:
            self.model_with_ddp = torch.nn.parallel.DistributedDataParallel(self._network,
                                                                            device_ids=[self._multiple_devices],
                                                                            find_unused_parameters=True)
            self.model_without_ddp = self.model_with_ddp.module

        self.n_parameters = sum(p.numel() for p in self.model_with_ddp.parameters() if p.requires_grad)
        logger.info(f'number of params:{self.n_parameters}')
        torch.distributed.barrier()
        self.loss_scaler = NativeScaler()

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        # important!!
        self._training_step(train_loader, val_loader, 0, self._n_epochs)
        self.model_without_ddp = self.model_with_ddp.module
        self._network = self.model_with_ddp.module

    def _after_task_intensive(self, inc_dataset, test_loader):
        if self._memory_size > 0:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                inc_dataset, self._herding_indexes, extract_distributed=True
            )
        else:
            self._class_means = None

    def update_center_lib(self, inc_dataset):
        self.center_library.update(
            # self.build_center_library(inc_dataset, extract_distributed=True, use_apg=self._task > 0))
            self.build_center_library(inc_dataset, extract_distributed=True, use_apg=True))
        self.build_center_lib_loader(self._task, self._base_task_size, self._task_size)

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()
        # self.plot_tsne()

    def save_old_APG(self):
        if self.self_prompt:
            self._old_APG = copy.deepcopy(self.adaptive_prompt_generator.to(self._device))

    def _eval_task(self, data_loader):
        if self._task == 0 and self.self_prompt:
            ypreds, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means, APG=True)
            all_acc = accuracy_per_task(ypreds, ytrue, init_task_size=self._base_task_size, task_size=self._task_size,
                                        topk=1)
            logger.info(f'---' * 50)
            logger.info(f'Acc with APG:\n\t{all_acc}')
        ypreds, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means, APG=self._task > 0)
        all_acc = accuracy_per_task(ypreds, ytrue, init_task_size=self._base_task_size, task_size=self._task_size,
                                    topk=1)
        logger.info(f'---' * 50)
        logger.info(f'Acc with APG [{self._task > 0}]:\n\t{all_acc}')
        return ypreds, ytrue

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def build_center_library(self, dataset, extract_distributed=True, use_apg=True):
        """
        This function is used for storing the feature center of each class (in low level representation), as well as its
        prompts generated by APG.

        Returns:

        """
        center_library = {}
        # Step 1. Get low level representation of the whole dataset.
        transform_args = ('input_size', 'color_jitter', 'aa', 'train_interpolation', 'reprob', 'remode', 'recount')
        _args = extract_transform_args(transform_args, self.args)
        trsf = build_transform(is_train=False, args=_args)
        if extract_distributed:
            inputs, loader = dataset.get_custom_loader(
                None, mode="test", data_source='train', distributed=True, transform_args=_args,
                force_batchsize=dataset._batch_size // utils.get_world_size(), current_task_class=True,
                memory=None
            )
        else:
            inputs, loader = dataset.get_custom_loader(
                None, mode="test", data_source='train', force_transform=trsf, current_task_class=True,
                force_batchsize=dataset._batch_size // utils.get_world_size(), memory=None
            )
        APG = self.adaptive_prompt_generator if use_apg else None
        logger.info(f'use APG to extra features? [{APG is not None}]')
        APG_state = APG.training
        APG.eval()
        features, targets, lowlevel_features, augmented_features = utils.extract_features(self._network, loader,
                                                                                          distributed=True,
                                                                                          additional_parameters={
                                                                                              'only_immediate': False,
                                                                                              'return_immediate_feat': True},
                                                                                          APG_dual_feat=True,
                                                                                          return_numpy=False,
                                                                                          APG=APG)

        torch.distributed.barrier()
        torch.cuda.synchronize()
        logger.info('get rid of replicate data')
        # get rid of replicate data.
        features = features[:len(inputs)]
        targets = targets[:len(inputs)]
        lowlevel_features = lowlevel_features[:len(inputs)]
        augmented_features = augmented_features[:len(inputs)]

        # testing_features = testing_features[:len(testing_inputs)]
        # testing_targets = testing_targets[:len(testing_inputs)]
        # testing_extra_features = testing_extra_features[:len(testing_inputs)]

        torch.distributed.barrier()
        torch.cuda.synchronize()

        # Step 3. Get corresponding prompts of each centroids
        temp_immediate_dataset = torch.utils.data.TensorDataset(lowlevel_features, targets)
        num_tasks = utils.get_world_size()
        batchsize = self.args.get("batch_size", 128)
        logger.info(f'Center lib batchsize {batchsize}')
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(temp_immediate_dataset, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)
        tmp_immediate_loader = torch.utils.data.DataLoader(temp_immediate_dataset, sampler=sampler_train,
                                                           batch_size=batchsize, num_workers=0, drop_last=False, )
        all_prompts = torch.tensor([])

        torch.distributed.barrier()
        torch.cuda.synchronize()
        logger.info(f'-' * 50)
        # Step 2. Get class centroid of each class
        for cls in tqdm(torch.unique(targets)):
            mask = targets == cls
            cls_num = cls.item()
            immedia_feat_mean = lowlevel_features[mask].mean(0)
            augmented_feat_mean = augmented_features[mask].mean(0)
            feat_mean = features[mask].mean(0)
            torch_feat = features[mask]

            center_library[cls_num] = {'centroid_mean': immedia_feat_mean}
            center_library[cls_num]['augmented_feat_mean'] = augmented_feat_mean
            center_library[cls_num]['augmented_feat_cov'] = torch.cov(augmented_features[mask].T)
            center_library[cls_num]['centroid_cov'] = torch.cov(lowlevel_features[mask].T)
            center_library[cls_num]['feat_mean'] = feat_mean
            with torch.no_grad():
                prompts = self.adaptive_prompt_generator(feat_mean.cuda().reshape(1, 1, -1)).cpu().squeeze(0).squeeze(0)
                center_library[cls_num]['feat_mean_prompts'] = prompts
            center_library[cls_num]['feat_cov'] = torch.cov(torch_feat.T)

        torch.distributed.barrier()
        torch.cuda.synchronize()
        logger.info(f'feed into APGs')
        logger.info(f'-' * 50)
        for content in tqdm(tmp_immediate_loader):
            feat = content[0].cuda().unsqueeze(1)
            with torch.no_grad():
                prompts = self.adaptive_prompt_generator(feat)

            # gather all feat from other process
            torch.distributed.barrier()
            torch.cuda.synchronize()
            all_feat_list = [torch.zeros_like(prompts, dtype=torch.float32) for _ in range(num_tasks)]
            torch.distributed.all_gather(all_feat_list, prompts)
            prompts = torch.cat(all_feat_list, dim=0).cpu()

            all_prompts = torch.cat((all_prompts, prompts), dim=0)

        torch.distributed.barrier()
        torch.cuda.synchronize()
        # get rid of replicate data
        all_prompts = all_prompts[:len(inputs)]
        assert len(all_prompts) == len(targets)

        logger.info(f'Extracting prompts mean')
        all_cls_prompts_mean = torch.tensor([])
        for cls, content in center_library.items():
            mask = targets == cls
            center_library[cls]['prompts_mean'] = []
            center_library[cls]['prompts_cov'] = []
            for i in range(self.more_prompts):
                prompts = all_prompts[mask][:, i, ...]
                prompts_mean = prompts.mean(dim=0)
                prompts_cov = torch.cov(prompts.transpose(0, 1))
                center_library[cls]['prompts_mean'].append(prompts_mean)
                center_library[cls]['prompts_cov'].append(prompts_cov)

            #
        APG.train(APG_state)
        logger.info('Done')
        return center_library

    def set_result_folder(self, path):
        self._result_folder = path

    def compute_accuracy(self, model, loader, class_means, APG=False):
        if self._evaluation_type in ("icarl", "nme"):
            features, targets_ = utils.extract_features(model, loader, distributed=True,
                                                        APG=self.adaptive_prompt_generator if APG else None)

            features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

            # Compute score for iCaRL
            sqd = cdist(class_means, features, 'sqeuclidean')
            score_icarl = (-sqd).T
            return score_icarl, targets_
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []
            # if hasattr(self, 'model_with_ddp'):
            #     net = self.model_with_ddp
            # else:
            net = self._network
            net.eval()
            if APG:
                self.adaptive_prompt_generator.eval()
            for input_dict in loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                with torch.no_grad():
                    if APG:
                        _, low_level_feat = net.backbone.forward_features(inputs, return_immediate_feat=True,
                                                                          break_immediate=True)
                        _low_level_feat = low_level_feat[:, 0].unsqueeze(1)
                        prompts = self.adaptive_prompt_generator(_low_level_feat)
                        logits = net(low_level_feat, extra_tokens=prompts, shortcut=True).detach().cpu()
                    else:
                        logits = net(inputs, ).detach().cpu()
                preds = F.softmax(logits, dim=-1)
                del logits
                ypred.append(preds.cpu().numpy())
            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def train_APG(self, train_loader, val_loader, run_id, resume_from_epoch=-1, pretrain=False, finetune=False,
                  stage0=False, task_id=0, ):
        """
        This function is used for training model._network.adaptive_prompt_generator after first stage.
        In this training process, we use the existing prompts to train the mapping from low level image representation
        to prompts.

        training data: train_loader
        testing: use the val_loader as well as the generated prompts (extra token) to go through the model and compare
        this result as the result of existing prompts.
        Args:
            train_loader:
            val_loader:

        Returns:

        """
        if pretrain:
            train_type = 'pretrain'
        elif finetune:
            train_type = 'finetune'
        else:
            train_type = 'APG'

        logger.info(f'train_type: [{train_type}] ')
        if resume_from_epoch == -1:
            self.APG_loss_scaler = NativeScaler()
        # get parameters
        criterion1 = None
        triplet_criterion = None
        mixup = self.args.get("mixup", 0.8)
        cutmix = self.args.get("cutmix", 1.0)
        mixup_prob = self.args.get("mixup_prob", 1)
        mixup_switch_prob = self.args.get("mixup_switch_prob", 0.5)
        cutmix_minmax = self.args.get("cutmix_minmax", None)
        mixup_mode = self.args.get("mixup_mode", "batch")
        smoothing = self.args.get("label_smoothing", 0.1)
        nb_classes = self._n_classes
        mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                label_smoothing=smoothing, num_classes=nb_classes)
        else:
            mixup_fn = None

        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()

        model = self._network
        model.train()
        if pretrain:
            model.freeze(trainable=True)
        else:
            model.backbone.set_backbone_frozen(num_layer=self.frozen_layer)
        finetune_params = []
        ada_params = []
        if pretrain:
            assert self._task == 0
            ada_params.extend(list(self._network.parameters()))
        else:
            self.adaptive_prompt_generator.train()
            ada_params = [
                {'params': list(self.adaptive_prompt_generator.parameters()),
                 'weight_decay': self.APG_lr_configs_args.APG_wd,
                 'lr': self.APG_lr_configs_args.lr},

                {'params': self._network.backbone.get_deep_params(self.frozen_layer),
                 'lr': self.APG_lr_configs_args.deep_layer_lr}]
            if self._task == 0:
                logger.info(f'Added pos_embed_new param to optimizer.')
                ada_params.append({'params': self._network.backbone.pos_embed_new, 'lr': self.APG_lr_configs_args.lr})
                ada_params.append(
                    {'params': list(self._network.classifier.parameters()), 'lr': self.APG_lr_configs_args.lr_cls})
                finetune_params.extend(list(self._network.classifier.parameters()))
            if self._task > 0:
                logger.info(f'Added classifier param to optimizer.')
                ada_params.append(
                    {'params': list(self._network.classifier.parameters()), 'lr': self.APG_lr_configs_args.lr_cls})
                finetune_params.extend(list(self._network.classifier.parameters()))
                finetune_params.extend(list(self.adaptive_prompt_generator.parameters()))
        # add extra tokens in APG
        if self.lib_loader is not None:
            # we add constraint here to make sure the APG does not forget
            logger.info(f'APG_constraint_loss_type:{self.APG_constraint_loss_type}')
            criterion1 = PromptSimilarity(type=self.APG_constraint_loss_type)

        soft_triplet_criterion = SoftTripletLoss()
        triplet_criterion = SoftTripletLoss()
        feature_triplet_criterion = SoftTripletLoss()
        old_new_feature_triplet_criterion = SoftTripletLoss()

        mem_loader = None
        old_fc = copy.deepcopy(self._network.classifier.state_dict())
        prompt_triplet = self.finetune_lr_config.get("prompt_triplet", False)
        feat_triplet = self.APG_lr_configs.get("feat_triplet", False)
        if pretrain:
            logger.info(f'creating optimizer for pre-training')
            self.pretrain_lr_config_args = utils.lr_scaler(self.pretrain_lr_config_args, train_loader.batch_size,
                                                           convert=False)
            self.pretrain_optimizer = create_optimizer(self.pretrain_lr_config_args, ada_params)
            self.pretrain_scheduler, _ = create_scheduler(self.pretrain_lr_config_args, self.pretrain_optimizer)
            lr_config = self.pretrain_lr_config_args
            optimizer = self.pretrain_optimizer
            scheduler = self.pretrain_scheduler
            self.load_teacher_statedict(self.teacher_state_dict_path)

        elif finetune:

            finetune_config = self.finetune_config
            self.finetune_lr_config_args = utils.lr_scaler(self.finetune_lr_config_args, train_loader.batch_size,
                                                           convert=False)
            finetune_lr_config, finetune_lr_config_args = self.finetune_lr_config, self.finetune_lr_config_args
            finetune_constraint_APG, finetune_parameter_dict = self.finetune_constraint_APG, self.finetune_parameter_dict
            scale_finetune_epochs = self.scale_finetune_epochs
            finetune_epochs = self.finetune_epochs

            use_memory = finetune_config.get("use_memory_to_finetune", False)
            # logger.info(f'Use memory for APG finetuning')
            if use_memory:
                logger.info(f'Use memory to finetune [{use_memory}].')
                transform_args = (
                    'input_size', 'color_jitter', 'aa', 'train_interpolation', 'reprob', 'remode', 'recount')
                _args = utils.extract_transform_args(transform_args, self.args)
                mem_loader = self.inc_dataset.get_memory_loader(*self.get_memory(), distributed=True, repeated_aug=True,
                                                                trans_args=_args)
                finetune_params.extend(self._network.backbone.get_deep_params())
            if scale_finetune_epochs:
                finetune_epochs = finetune_epochs + int(task_id * self.scale_finetune_epochs_per_time)
                logger.info(f'scaled finetune epochs: '
                            f'\n\ttask_size [{self._task_size}]'
                            f'\n\tbase epoch [{finetune_epochs}]'
                            f'\n\tscaled epoch [{finetune_epochs}]')
                self.finetune_lr_config_args.epochs = finetune_epochs
            logger.info(f'creating optimizer for APG finetuning')
            self.finetune_APG_optimizer = create_optimizer(finetune_lr_config_args, finetune_params)
            self.finetune_APG_scheduler, _ = create_scheduler(finetune_lr_config_args,
                                                              self.finetune_APG_optimizer)
            lr_config = finetune_lr_config_args
            optimizer = self.finetune_APG_optimizer
            scheduler = self.finetune_APG_scheduler
            feat_triplet_old_new = finetune_lr_config.get("feat_triplet_old_new", False)
            p_attn = finetune_config.get("p_attn", False)


        else:
            logger.info(f'creating optimizer for APG training')
            self.APG_lr_configs_args = utils.lr_scaler(self.APG_lr_configs_args, train_loader.batch_size, convert=False)
            self.APG_optimizer = create_optimizer(self.APG_lr_configs_args, ada_params)
            # self.APG_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.APG_optimizer,
            #                                                           milestones=self.APG_lr_configs_args.milestones,
            #                                                           gamma=0.1)
            self.APG_scheduler, _ = create_scheduler(self.APG_lr_configs_args, self.APG_optimizer)
            lr_config = self.APG_lr_configs_args
            optimizer = self.APG_optimizer
            scheduler = self.APG_scheduler
            feat_triplet_old_new = self.APG_lr_configs.get("feat_triplet_old_new", False)
            p_attn = self.APG_lr_configs.get("p_attn", False)
            if (self.reset_deep_layers and self._task == 0) or self.force_reset_deep_layers:
                logger.info(f'Resetting deep layers to random numbers.')
                self._network.backbone.reset_deep_layers(self.frozen_layer)
            logger.info(f'feat_triplet [{feat_triplet}], feat_triplet_old_new [{feat_triplet_old_new}]')

        self.APG_loss_scaler = NativeScaler()
        logger.info(f'{lr_config}')
        if self.self_prompt:
            ddp_model_APG = torch.nn.parallel.DistributedDataParallel(self.adaptive_prompt_generator,
                                                                      find_unused_parameters=True)
        else:
            ddp_model_APG = None
        ddp_model_backbone = torch.nn.parallel.DistributedDataParallel(self._network, find_unused_parameters=True)

        loss_scaler = MyScaler()
        if pretrain:
            total_epoch = self.pretrain_lr_config_args.epochs
            logger.info("testing teacher.")
            self.testing(self.teacher_network, APG_model=None, val_loader=val_loader, APG=False, output_key='logits')
        elif finetune:
            total_epoch = self.finetune_epochs
            if self.scale_finetune_epochs:
                total_epoch = self.finetune_epochs + int(task_id * self.scale_finetune_epochs_per_time)
                logger.info(f'scaled finetune epochs: '
                            f'\n\ttask_size [{self._task_size}]'
                            f'\n\tbase epoch [{self.finetune_epochs}]'
                            f'\n\tscaled epoch [{total_epoch}]')

        elif stage0:
            total_epoch = self.APG_nb_epochs_stage0
        else:
            total_epoch = self.APG_nb_epochs
        resume_from_epoch = resume_from_epoch if not finetune else -1
        self.testing(ddp_model_backbone, self.adaptive_prompt_generator, val_loader)
        logger.info(f"start training for {total_epoch - (resume_from_epoch + 1)} epochs.")
        if resume_from_epoch != -1:
            scheduler.step(resume_from_epoch + 1)
        for epoch in range(resume_from_epoch + 1, total_epoch):
            if True:
                train_loader.sampler.set_epoch(epoch)
            if self.lib_loader is not None:
                self.lib_loader.sampler.set_epoch(epoch)
            if self.finetune_config and finetune:
                logger.info(f'start to finetune!')
                ddp_model_APG.eval()
                if self.finetune_repeat_new_cls and self._task > 9:
                    self.lib_loader.dataset.on_repeat_new_classes()
                    logger.info('Turned on repeating new classes.')
                train_epoch = fine_tune_adaptive_prompt_generator
                train_stats = train_epoch(ddp_model_backbone,
                                          self.lib_loader,
                                          ddp_model_APG,
                                          optimizer,
                                          epoch,
                                          loss_scaler,
                                          self.clip_grad,
                                          criterion1=criterion1,
                                          memory_loader=mem_loader,
                                          finetune_whole_fc=self.finetune_whole_fc,
                                          constraint_APG=finetune_constraint_APG,
                                          parameter_dict=finetune_parameter_dict,
                                          # triplet_criterion=triplet_criterion,
                                          prompt_tri=triplet_criterion if prompt_triplet else None,
                                          p_attn=p_attn,
                                          num_classes=nb_classes, )
                self.lib_loader.dataset.off_repeat_new_classes()
            elif pretrain:
                assert task_id == 0
                ddp_model_backbone.train()
                # train_stats = train_one_epoch_pretrain(ddp_model_backbone, train_loader, optimizer, self._device, epoch,
                #                                        loss_scaler, self.clip_grad, mixup_fn=mixup_fn,
                #                                        teacher_model=self.teacher_network)
                train_stats = {}
            else:
                self.adaptive_prompt_generator.train()
                ddp_model_APG.train()
                logger.info(f'Starting Training APG.')
                train_epoch = train_one_epoch_adaptive_prompt_generator
                train_stats = train_epoch(ddp_model_backbone,
                                          criterion,
                                          ddp_model_APG,
                                          train_loader,
                                          optimizer,
                                          self._device,
                                          epoch,
                                          loss_scaler,
                                          self.clip_grad,
                                          mixup_fn=mixup_fn,
                                          criterion1=criterion1,
                                          triplet_criterion=triplet_criterion if self.APG_Prompt_triplet else None,
                                          center_lib_loader=self.lib_loader,
                                          use_center_constraint=self.use_center_constraint_in_APG,
                                          train_backbone_model=True,
                                          use_apg=True,
                                          feature_triplet_criterion=feature_triplet_criterion if feat_triplet else None,
                                          old_new_feature_triplet_criterion=old_new_feature_triplet_criterion if feat_triplet_old_new else None,
                                          parameter_dict=self.parameter_dict,
                                          taskid=task_id,
                                          norm_constraint=self.APG_norm_constraint,
                                          p_attn=p_attn,
                                          prompt_per_cls=self.prompt_super_class_n_prompts if self.prompt_use_superclass else self.class_n_prompts,
                                          )

            scheduler.step(epoch)

            if self._result_folder and epoch % self.testing_interval == self.testing_interval - 1:
                if utils.is_main_process():
                    self.save_parameters(self._result_folder, run_id=run_id, epoch=epoch, end=epoch == total_epoch - 1,
                                         train_type=train_type)
                    self.save_parameters(self._result_folder, run_id=run_id, epoch=epoch, end=epoch == total_epoch - 1,
                                         train_type=train_type, prefix=train_type + '_')
                if pretrain:
                    self.save_parameters(self._result_folder, run_id=run_id, epoch=epoch, prefix='Pretrain_')
            torch.distributed.barrier()
            torch.cuda.synchronize()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'task': self._task
                         }

            if self._result_folder and utils.is_main_process():
                with open(os.path.join(self._result_folder, f'log_APG_{self._task}.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch >= 0 and epoch % self.testing_interval == self.testing_interval - 1:
                self.testing(ddp_model_backbone, ddp_model_APG, val_loader, )

            torch.distributed.barrier()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info('Training time {}'.format(total_time_str))

            pass
        if pretrain:
            self._network = ddp_model_backbone.module
        else:
            self._network = ddp_model_backbone.module
            self.adaptive_prompt_generator = ddp_model_APG.module
        if finetune and self.resume_fc_after_finetune:
            self._network.classifier.load_state_dict(old_fc)

    def testing(self, model, APG_model, val_loader, APG=True, output_key=None):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if isinstance(APG_model, torch.nn.parallel.DistributedDataParallel):
            APG_model = APG_model.module
        if APG:
            test_stats = evaluate(val_loader, model, self._device, APG=APG_model)
            logger.info(f'Test with APG--------------------------------------------')
            logger.info(
                f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
        logger.info(f'Test without APG---------------------------------------')
        test_stats = evaluate(val_loader, model, self._device, APG=None, output_key=output_key)
        logger.info(
            f"Accuracy of the network on the {len(val_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
        if model is not None:
            model.train()
        if APG_model is not None:
            APG_model.train()

    def build_center_lib_loader(self, task, base_task_size, task_size):
        all_centroid = torch.tensor([])
        # all_prompts = torch.tensor([])
        all_prompts = {}
        all_feat = torch.tensor([])
        all_target = torch.tensor([])
        cls_dist_immediate = {}
        cls_dist_feat = {}
        cls_feat_mean = {}
        cls_dist_augmented_feat = {}
        cls_all_augmented_feat = {}
        for cls, content in self.center_library.items():
            centroid_mean = content['centroid_mean']
            augmented_feat_mean = content['augmented_feat_mean']
            augmented_feat_cov = content['augmented_feat_cov']
            centroid_conv = content['centroid_cov']
            # centroid_conv = torch.eye(384)

            # prompts = content['prompts']
            # if task == 0:
            #     prompts_mean = content['feat_mean_prompts']
            # else:
            #     prompts_mean = content['prompts_mean']
            prompts_mean = content['prompts_mean']
            prompts_cov = content['prompts_cov']
            feat_mean = content['feat_mean']
            # feat_cov = torch.from_numpy(content['feat_cov'])
            feat_cov = content['feat_cov']
            # feat_cov = torch.eye(feat_cov.shape[1])
            tmp = torch.ones(centroid_conv.shape[0]) * 1e-6
            # feat_cov = feat_cov.mul(torch.eye(feat_cov.shape[0], dtype=torch.bool))
            # a = torch.eye(feat_mean.shape[0]) * 1e-5
            # distribution_immediate = MultivariateNormal(loc=centroid_mean, covariance_matrix=a)
            # distribution_feat = MultivariateNormal(loc=feat_mean, covariance_matrix=a)

            distribution_immediate = MultivariateNormal(loc=centroid_mean, covariance_matrix=centroid_conv + tmp)
            distribution_feat = MultivariateNormal(loc=feat_mean, covariance_matrix=feat_cov + tmp)
            distribution_augmented_feat = MultivariateNormal(loc=augmented_feat_mean,
                                                             covariance_matrix=augmented_feat_cov + tmp)
            # distribution_prompts = MultivariateNormal(loc=prompts_mean, covariance_matrix=prompts_cov+tmp)
            all_prompts[cls] = prompts_mean
            cls_dist_immediate[cls] = distribution_immediate
            cls_dist_feat[cls] = distribution_feat
            cls_feat_mean[cls] = feat_mean
            cls_dist_augmented_feat[cls] = distribution_augmented_feat
            # cls_all_augmented_feat[cls] = content['augmented_feat_all']
            # all_centroid = torch.cat((all_centroid, torch.from_numpy(centroid_mean).unsqueeze(0)))
            # all_prompts = torch.cat((all_prompts, prompts)).detach()
            # all_feat = torch.cat((all_feat, feat.unsqueeze(0)))
            target = torch.tensor([cls]).long()
            all_target = torch.cat((all_target, target))
        new_classes = [i for i in range(base_task_size + (task - 1) * task_size, base_task_size + task * task_size)] \
            if task > 0 else [i for i in range(base_task_size)]
        all_target = all_target.long()
        dataset = TensorAugDataset(cls_dist_immediate, cls_dist_feat, cls_dist_augmented_feat, cls_all_augmented_feat,
                                   cls_feat_mean, all_target, all_prompts, n=self.center_lib_n, new_classes=new_classes)
        # dataset = torch.utils.data.TensorDataset(all_centroid, all_prompts, all_feat, all_target)
        num_tasks = utils.get_world_size()
        batchsize = self.args.get("batch_size_center_lib", 128)
        logger.info(f'Center lib batchsize {batchsize}')
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.lib_loader = torch.utils.data.DataLoader(dataset, sampler=sampler_train,
                                                      batch_size=batchsize, num_workers=0, drop_last=True,
                                                      collate_fn=my_collect_fn)

    def reset_backbone_pos_embed(self):
        num_tokens = (self.adaptive_prompt_generator.all_tokens.get_all_tokens()).shape[1]
        self._network.backbone.reset_pos_embed(num_tokens)

    def reset_fc(self):
        self.old_fc = copy.deepcopy(self._network.classifier.state_dict())
        self._network.classifier.reset_weights()


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
