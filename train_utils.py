"""Training utils."""

import torch

from collections import OrderedDict
from torch.nn.utils.clip_grad import *


# Exlude these standard flags from wandb config
CONFIG_EXCLUDE_KEYS = ['?', 'alsologtostderr', 'hbm_oom_exit', 'help', 'helpfull', 'helpshort', 'helpxml',
    'log_dir', 'logger_levels', 'logtostderr', 'only_check_args', 'op_conversion_fallback_to_while_loop', 'pdb', 'pdb_post_mortem',
    'profile_file', 'runtime_oom_exit', 'run_with_pdb', 'run_with_profiling', 'showprefixforinfo', 'stderrthreshold',
    'test_random_seed', 'test_randomize_ordering_seed', 'test_srcdir', 'test_tmpdir', 'tt_check_filter', 'tt_single_core_summaries',
    'use_cprofile_for_profiling', 'v', 'verbosity', 'xml_output_file']


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
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, 1.0, total_steps)

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


def clip_grad_norm(model, llm, fromage_adapter, likelihood,
                    max_norm=10.0, norm_type=2.0):
    log_dict = {}

    log_dict['model_norm'] = clip_grad_norm_(
        model.parameters(), max_norm=max_norm, norm_type=norm_type)

    if llm is not None:
        log_dict['llm_norm'] = clip_grad_norm_(
        llm.parameters(), max_norm=max_norm, norm_type=norm_type)
    if fromage_adapter is not None:
        log_dict['fromage_norm'] = clip_grad_norm_(
            fromage_adapter.parameters(), max_norm=max_norm, norm_type=norm_type)
    if likelihood is not None:
        log_dict['likelihood_norm'] = clip_grad_norm_(
            likelihood.parameters(), max_norm=max_norm, norm_type=norm_type)

    return log_dict


def forward_pass(model_type, fromage_settings, model, llm, fromage_adapter, likelihood, model_input, labels=None, loss_fn=None, return_loss=True):
    if 'dkl' == model_type:
        # Get predictive output
        output = model(**model_input)
        if return_loss:
            loss = -loss_fn(output, labels[:, 0])
        pred_prob = likelihood(output).probs.mean(0)[:, 1]
    elif 'llm' in model_type:
        gnn_embeddings = model_input['gnn_embeddings']
        del model_input['gnn_embeddings']

        if fromage_settings['use_fromage'] and 'top_only' != fromage_settings['fromage_type']:
            # Embedding each drug / disease feature separately
            # TODO(schwarzjn): Should we process drug/disease embedding separately?
            batch_size = gnn_embeddings.shape[0]
            fromage_features = fromage_adapter(
                gnn_embeddings.view([batch_size, 2, fromage_settings['gnn_data_dim'] // 2])
            )
            fromage_settings['fromage_features'] = fromage_features

        # Apply LLM
        model_input['fromage_settings'] = fromage_settings
        llm_output = llm(**model_input).to(torch.float32)

        # Optionally pass output of transformer together with GNN to predictive module
        if fromage_settings['use_fromage'] and 'top' in fromage_settings['fromage_type']:
            llm_output = torch.concat([llm_output, gnn_embeddings], axis=-1)

        if 'mlp' in model_type:
            # Apply Linear/MLP predictor
            output = model(llm_output)
            if return_loss:
                loss = loss_fn(output, labels)

            # Probability of predicting class 1
            pred_prob = torch.sigmoid(output)
        elif 'dkl' in model_type:
            # Apply GP predictor
            output = model(llm_output)
            if return_loss:
                loss = -loss_fn(output, labels[:, 0])

            # Probability of predicting class 1
            pred_prob = likelihood(output).probs.mean(0)[:, 1]
        else:
            assert False, 'invalid choice'
    else:
        # Get predictive output
        output = model(**model_input)
        if return_loss:
            loss = loss_fn(output, labels)
        pred_prob = torch.sigmoid(output)

    pred_label = pred_prob.ge(0.5).float()

    if return_loss:
        return pred_prob, pred_label, loss
    else:
        return pred_prob, pred_label
