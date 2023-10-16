import warnings

import torch
from torch import optim

from inclearn import models
from inclearn.lib import data, schedulers
from timm.models import create_model
from inclearn.backbones import resnet


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "sgd_nesterov":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    raise NotImplementedError


def get_backbone(backbone_type, all_args, **kwargs):
    if backbone_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif 'deit' in backbone_type:
        if 'prompt' in backbone_type:
            return create_model(backbone_type,
                                insert_mode=all_args['prompt_config'].get('insert_mode', 'shallow'),
                                prompt_numbers=all_args['prompt_config'].get('num_prompt_per_task', 10),
                                pretrained=False, num_classes=0, drop_rate=all_args['drop'],
                                drop_path_rate=all_args['drop_path'], drop_block_rate=None)

        return create_model(backbone_type, pretrained=False, num_classes=0, drop_rate=all_args['drop'],
                            drop_path_rate=all_args['drop_path'], drop_block_rate=None)

    elif 'SelfPrompt' in backbone_type:
        return create_model(backbone_type, pretrained=False, num_classes=0, drop_rate=all_args['drop'],
                            extra_token_nums=all_args.get('prompt_config', {}).get('more_prompts', 1) + 1,
                            drop_path_rate=all_args['drop_path'], drop_block_rate=None,
                            immediate_layer=all_args['immediate_layer'],
                            immediate_after_norm=all_args.get('immediate_after_norm', False),
                            replace_cls=all_args.get('replace_cls', False))

    raise NotImplementedError("Unknwon backbones type {}.".format(backbone_type))


def get_model(args):
    dict_models = {
        "continual_trans": models.ContinualTrans
    }

    model = args["model"].lower()

    if model not in dict_models:
        raise NotImplementedError(
            "Unknown model {}, must be among {}.".format(args["model"], list(dict_models.keys()))
        )

    return dict_models[model](args)


def get_data(args, class_order=None):
    return data.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"],
        onehot=args["onehot"],
        increment=args["increment"],
        initial_increment=args["initial_increment"],
        sampler=get_sampler(args),
        sampler_config=args.get("sampler_config", {}),
        data_path=args["data_path"],
        class_order=class_order,
        seed=args["seed"],
        dataset_transforms=args.get("dataset_transforms", {}),
        all_test_classes=args.get("all_test_classes", False),
        metadata_path=args.get("metadata_path")
    )


def set_device(args):
    devices = []

    for device_type in args["device"]:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device_type))

        devices.append(device)

    args["device"] = devices


def get_sampler(args):
    if args["sampler"] is None:
        return None

    sampler_type = args["sampler"].lower().strip()

    if sampler_type == "npair":
        return data.NPairSampler
    elif sampler_type == "triplet":
        return data.TripletSampler
    elif sampler_type == "tripletsemihard":
        return data.TripletCKSampler

    raise ValueError("Unknown sampler {}.".format(sampler_type))


def get_lr_scheduler(
        scheduling_config, optimizer, nb_epochs, lr_decay=0.1, warmup_config=None, task=0
):
    if scheduling_config is None:
        return None
    elif isinstance(scheduling_config, str):
        warnings.warn("Use a dict not a string for scheduling config!", DeprecationWarning)
        scheduling_config = {"type": scheduling_config}
    elif isinstance(scheduling_config, list):
        warnings.warn("Use a dict not a list for scheduling config!", DeprecationWarning)
        scheduling_config = {"type": "step", "epochs": scheduling_config}

    if scheduling_config["type"] == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            scheduling_config["epochs"],
            gamma=scheduling_config.get("gamma") or lr_decay
        )
    elif scheduling_config["type"] == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduling_config["gamma"])
    elif scheduling_config["type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=scheduling_config["gamma"]
        )
    elif scheduling_config["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)
    elif scheduling_config["type"] == "cosine_with_restart":
        scheduler = schedulers.CosineWithRestarts(
            optimizer,
            t_max=scheduling_config.get("cycle_len", nb_epochs),
            factor=scheduling_config.get("factor", 1.)
        )
    elif scheduling_config["type"] == "cosine_annealing_with_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=scheduling_config.get("min_lr")
        )
    else:
        raise ValueError("Unknown LR scheduling type {}.".format(scheduling_config["type"]))

    if warmup_config:
        if warmup_config.get("only_first_step", True) and task != 0:
            pass
        else:
            print("Using WarmUp")
            scheduler = schedulers.GradualWarmupScheduler(
                optimizer=optimizer, after_scheduler=scheduler, **warmup_config
            )

    return scheduler
