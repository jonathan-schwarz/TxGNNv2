"""Load TxGNN data."""
import numpy as np
import torch
import os


from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from finetune_models.llm_models import get_tokenizer

DATA_PATH = '/n/holystore01/LABS/mzitnik_lab/Users/jschwarz/TxGNNv2/data/pretrained_mine/'
DATASET_VERSION = 'complex_disease_matrix'

FILE_NAMES = {
    'did': 'drug_indication_disease',
    'dod': 'drug_off-label use_disease',
    'dcd': 'drug_contraindication_disease',
    'drid': 'disease_rev_indication_drug',
    'drod': 'disease_rev_off-label use_drug',
    'drcd': 'disease_rev_contraindication_drug',
}

MAX_LENGTH = 65
NUM_DISEASES = 66  # In `txgnn_did` test set
NUM_DRUGS = 7957


def load_split(data, mode, merge_str):

    u_names = data[mode + '_u_names']
    v_names = data[mode + '_v_names']
    try:
        text = np.concatenate([u_names, v_names], axis=1)
    except:
        text = np.concatenate(
            [u_names[:, np.newaxis], v_names[:, np.newaxis]], axis=1)
    labels = data[mode + '_labels']

    text = [merge_str.format(text[i][0], text[i][1]) for i in range(text.shape[0])]
    dict = {
        'text': text,
        'label': labels.astype(np.int64)[:, 0].tolist(),
    }
    dataset = Dataset.from_dict(dict)

    return dataset


def load_txgnn_dataset_text(dataset, tokenizer):
    path = os.path.join(DATA_PATH, '{}/separate/{}.npz')
    data = np.load(path.format(
        DATASET_VERSION, FILE_NAMES[dataset.split('_')[1]]))

    if 'did' in dataset:
        merge_str = 'Is {} an effective treatment for {}?'
    elif 'dod' in dataset:
        merge_str = 'Is {} effective for off-label use on {}?'
    elif 'dcd' in dataset:
        merge_str = 'Should {} be avoided for patients suffering from {}?'
    else:
        assert False, 'Reverse cases not yet supported'

    full_dataset = DatasetDict(
        {'train': load_split(data, 'train', merge_str),
         'valid': load_split(data, 'valid', merge_str),
         'test': load_split(data, 'test', merge_str)}
    )

    def _preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, pad_to_max_length=True)

    tokenized_dataset = full_dataset.map(_preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    task_type = "SEQ_CLS"

    return tokenized_dataset, data_collator, task_type


def load_txgnn_dataset_text_matrix(dataset, data, tokenizer, eval_bandit=False):
    if 'did' in dataset:
        merge_str = 'Is {} an effective treatment for {}?'
    elif 'dod' in dataset:
        merge_str = 'Is {} effective for off-label use on {}?'
    elif 'dcd' in dataset:
        merge_str = 'Should {} be avoided for patients suffering from {}?'
    else:
        assert False, 'Reverse cases not yet supported'

    u_names = data['u_names']
    v_names = data['v_names']

    if eval_bandit:
        text = np.concatenate(
            [data['u_names'][..., np.newaxis],
             data['v_names'][..., np.newaxis]], axis=-1)
        join_fn = lambda x: merge_str.format(x[0], x[1])
        text = np.apply_along_axis(join_fn, -1, text)
    else:
        try:
            text = np.concatenate([u_names, v_names], axis=-1)
        except:
            text = np.concatenate(
                [data['u_names'][:, np.newaxis],
                data['v_names'][:, np.newaxis]], axis=-1)
        text = [merge_str.format(text[i][0], text[i][1]) for i in range(text.shape[0])]

    full_dataset = DatasetDict(
        {'matrix': Dataset.from_dict({'text': text.flatten()})},
    )

    def _preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, pad_to_max_length=True)

    tokenized_dataset = full_dataset.map(_preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    task_type = "SEQ_CLS"

    return tokenized_dataset, data_collator, task_type


def load_txgnn_dataset_raw(dataset, device):
    path = os.path.join(DATA_PATH, '{}/separate/{}.npz')
    data = np.load(path.format(
        DATASET_VERSION, FILE_NAMES[dataset.split('_')[1]]))

    train_x = torch.Tensor(np.concatenate(
        [data['train_h_u'], data['train_h_v']], axis=1)).to(device)
    train_y = torch.Tensor(data['train_labels']).to(device)
    try:
        train_names = np.concatenate(
            [data['train_u_names'], data['train_v_names']], axis=1)
    except:
        train_names = np.concatenate(
            [data['train_u_names'][:, np.newaxis],
             data['train_v_names'][:, np.newaxis]], axis=1)

    valid_x = torch.Tensor(np.concatenate(
        [data['valid_h_u'], data['valid_h_v']], axis=1)).to(device)
    valid_y = torch.Tensor(data['valid_labels']).to(device)
    try:
        valid_names = np.concatenate(
            [data['valid_u_names'], data['valid_v_names']], axis=1)
    except:
        valid_names = np.concatenate(
            [data['valid_u_names'][:, np.newaxis],
             data['valid_v_names'][:, np.newaxis]], axis=1)

    test_x = torch.Tensor(np.concatenate(
        [data['test_h_u'], data['test_h_v']], axis=1)).to(device)
    test_y = torch.Tensor(data['test_labels']).to(device)
    try:
        test_names = np.concatenate(
            [data['test_u_names'], data['test_v_names']], axis=1)
    except:
        test_names = np.concatenate(
            [data['test_u_names'][:, np.newaxis],
             data['test_v_names'][:, np.newaxis]], axis=1)

    return (train_x, train_y, train_names), (valid_x, valid_y, valid_names), (test_x, test_y, test_names)


def load_txgnn_dataset(dataset, dataset_type, model, batch_size, device):
    # Format: (features, labels, drug/disease names)
    train, valid, test = load_txgnn_dataset_raw(dataset, device)

    # Pretrained GNN features
    data_dim = train[0].shape[1]
    num_classes = 2
    num_train_points = train[0].shape[0]
    num_valid_points = valid[0].shape[0]
    num_test_points = test[0].shape[0]
    # Currently unused
    inducing_x = None

    if dataset_type == 'embedding':
        tokenizer = None  # Unused

        train_set = torch.utils.data.TensorDataset(train[0], train[1])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        valid_set = torch.utils.data.TensorDataset(valid[0], valid[1])
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=num_valid_points, shuffle=False)
        test_set = torch.utils.data.TensorDataset(test[0], test[1])
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=num_test_points, shuffle=False)
    elif dataset_type == 'embedding_text':
        # LLM input
        tokenizer = get_tokenizer(model)
        tokenized_dataset, _, task_type = load_txgnn_dataset_text(
            dataset, tokenizer)


        def _build_dataset(gnn_features, labels, _tokenized_dataset):
            # TODO(schwarzjn): Fix code to avoid `drop_last`
            tensor_dataset =  torch.utils.data.TensorDataset(
                gnn_features,  # GNN features
                torch.Tensor(_tokenized_dataset['input_ids']).long(),
                torch.Tensor(_tokenized_dataset['attention_mask']).long(), labels)

            # We are shuffling in all cases to avoid ROC computation errors
            return torch.utils.data.DataLoader(
                tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        train_loader = _build_dataset(train[0], train[1], tokenized_dataset['train'])
        valid_loader = _build_dataset(valid[0], valid[1], tokenized_dataset['valid'])
        test_loader = _build_dataset(test[0], test[1], tokenized_dataset['test'])

    return train_loader, valid_loader, test_loader, num_train_points, data_dim, num_classes, inducing_x, tokenizer


def load_txgnn_dataset_matrix(dataset, dataset_type, model, batch_size, device, eval_bandit=False):

    if eval_bandit:
        path = os.path.join(DATA_PATH, '{}/matrix/test_matrix_{}_bandit_eval.npz')
    else:
        path = os.path.join(DATA_PATH, '{}/matrix/test_matrix_{}.npz')

    data = np.load(path.format(
        'complex_disease_matrix', FILE_NAMES[dataset.split('_')[1]]))

    matrix_x = torch.Tensor(np.concatenate(
        [data['h_u'], data['h_v']], axis=-1)).to(device)
    matrix_y = torch.Tensor(data['labels']).to(device)

    data_dim = matrix_x.shape[-1]
    num_classes = 2

    if dataset_type == 'embedding':
        matrix_set = torch.utils.data.TensorDataset(matrix_x, matrix_y)
        matrix_loader = torch.utils.data.DataLoader(
            matrix_set, batch_size=batch_size, shuffle=True)
        num_matrix_points = len(matrix_loader.dataset)

        tokenizer = None
    elif dataset_type == 'embedding_text':
        # LLM input
        tokenizer = get_tokenizer(model)
        tokenized_dataset, _, task_type = load_txgnn_dataset_text_matrix(
            dataset, data, tokenizer, eval_bandit)

        input_ids = torch.tensor(tokenized_dataset['matrix']['input_ids']).long()
        attention_mask = torch.tensor(tokenized_dataset['matrix']['attention_mask']).long()
        if eval_bandit:
            # Shape [num_diseases, num_drugs, seq length]
            input_ids = input_ids.reshape([NUM_DISEASES, NUM_DRUGS, -1])
            attention_mask = attention_mask.reshape([NUM_DISEASES, NUM_DRUGS, -1])

        matrix_set = torch.utils.data.TensorDataset(
            matrix_x, input_ids, attention_mask, matrix_y)
        matrix_loader =  torch.utils.data.DataLoader(
            matrix_set, batch_size=batch_size, shuffle=True, drop_last=True)
        num_matrix_points = len(matrix_loader.dataset)

    return matrix_loader, num_matrix_points, data_dim, num_classes, tokenizer


def assemble_batch(batch, model_type, use_fromage, device, return_labels=True):
    if 'llama' in model_type:
        model_input = {
            'input_ids': batch[1].to(device),
            'attention_mask': batch[2].to(device),
        }
        if return_labels:
            labels = batch[3].to(device)

        if use_fromage:
            model_input['gnn_embeddings'] = batch[0].to(device)
    else:
        train_x = batch[0].to(device)
        if return_labels:
            labels = batch[1].to(device)

        # Make compatible with MLP
        train_x = train_x.view(train_x.size(0), -1)
        model_input = {
            'input': train_x,
        }

    if return_labels:
        return model_input, labels
    else:
        return model_input
