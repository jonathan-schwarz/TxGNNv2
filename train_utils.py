"""Training utils."""

import torch

from collections import OrderedDict


def load_scheduler(scheduler_type, optimizer, n_epochs, steps_per_epoch):

    total_steps = n_epochs * steps_per_epoch
    if 'train_plateau' == scheduler_type:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=25)
    elif 'valid_plateau' == scheduler_type:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=5)
    elif 'cosine_decay' == scheduler_type:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps)
    elif 'cosine_decay_with_warmup' == scheduler_type:
        warmup_ratio = 0.03
        n_warmup_steps = int(warmup_ratio * total_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=n_warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[n_warmup_steps+1])
    elif 'multi_step_lr' == scheduler_type:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[0.25 * total_steps, 0.5 * total_steps, 0.75 * total_steps],
            gamma=0.1)

    return scheduler


def get_trainable_parameters(model, logging=True):
    num_trainable_params = 0
    num_all_param = 0

    trainable_params = OrderedDict()
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            num_all_param += param.numel()
            if param.requires_grad:
                trainable_params[param_name] = param
                num_trainable_params += param.numel()

        if logging:
            print(
                f"trainable params: {num_trainable_params} || all params: {num_all_param} || trainable %: {100 * num_trainable_params / num_all_param:.2f}"
            )

        return trainable_params


def construct_llm_state_dict(llm, partial_state_dict):
    full_state_dict = OrderedDict()
    with torch.no_grad():
        for param_name, param in llm.named_parameters():
            if param_name in partial_state_dict:
                full_state_dict[param_name] = partial_state_dict[param_name]
            else:
                full_state_dict[param_name] = param

    return full_state_dict