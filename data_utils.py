"""Load TxGNN data."""
import numpy as np
import torch
import os

from datasets import load_dataset, Dataset, DatasetDict

from transformers import DataCollatorWithPadding

DATA_PATH = '/n/holystore01/LABS/mzitnik_lab/Users/jschwarz/TxGNNv2/data/pretrained_mine/'

FILE_NAMES = {
    'did': 'drug_indication_disease',
    'dod': 'drug_off-label use_disease',
    'dcd': 'drug_contraindication_disease',
    'drid': 'disease_rev_indication_drug',
    'drod': 'disease_rev_off-label use_drug',
    'drcd': 'disease_rev_contraindication_drug',
}


def load_split(outfile, mode, merge_str):

    train_u_names = outfile[mode + '_u_names']
    train_v_names = outfile[mode + '_v_names']
    train_text = np.concatenate([train_u_names, train_v_names], axis=1)
    train_labels = outfile[mode + '_labels']

    train_text = [merge_str.format(train_text[i][0], train_text[i][1]) for i in range(train_text.shape[0])]
    train_dict = {
        'text': train_text,
        'label': train_labels.astype(np.int32)[:, 0].tolist(),
    }
    dataset = Dataset.from_dict(train_dict)

    return dataset

def load_txgnn_dataset_text(dataset, dataset_use_v2, tokenizer):
    if dataset_use_v2:
        path = os.path.join(
            DATA_PATH, 'complex_disease_v2/separate/{}.npz')
    else:
        path = os.path.join(
            DATA_PATH, 'complex_disease/separate/{}.npz')

    data = np.load(path.format(FILE_NAMES[dataset.split('_')[1]]))

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
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_dataset = full_dataset.map(_preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    task_type = "SEQ_CLS"

    return tokenized_dataset, data_collator, task_type


def load_txgnn_dataset_embedding(dataset, batch_size, dataset_use_v2, device):
    if dataset_use_v2:
        path = os.path.join(
            DATA_PATH, 'complex_disease_v2/separate/{}.npz')
    else:
        path = os.path.join(
            DATA_PATH, 'complex_disease/separate/{}.npz')

    data = np.load(path.format(FILE_NAMES[dataset.split('_')[1]]))

    train_x = torch.Tensor(np.concatenate(
        [data['train_h_u'], data['train_h_v']], axis=1)).to(device)
    train_y = torch.Tensor(data['train_labels']).to(device)
    train_names = np.concatenate(
        [data['train_u_names'], data['train_v_names']], axis=1)
    train_set = torch.utils.data.TensorDataset(train_x, train_y)

    valid_x = torch.Tensor(np.concatenate(
        [data['valid_h_u'], data['valid_h_v']], axis=1)).to(device)
    valid_y = torch.Tensor(data['valid_labels']).to(device)
    valid_names = np.concatenate(
        [data['valid_u_names'], data['valid_v_names']], axis=1)
    valid_set = torch.utils.data.TensorDataset(valid_x, valid_y)

    test_x = torch.Tensor(np.concatenate(
        [data['test_h_u'], data['test_h_v']], axis=1)).to(device)
    test_y = torch.Tensor(data['test_labels']).to(device)
    test_names = np.concatenate(
        [data['test_u_names'], data['test_v_names']], axis=1)
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    data_dim = train_x.shape[1]
    num_classes = 2
    num_valid_points = valid_x.shape[0]
    num_test_points = test_x.shape[0]
    inducing_x = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    num_train_points = len(train_loader.dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=num_valid_points, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=num_test_points, shuffle=False)

    return train_loader, valid_loader, test_loader, num_train_points, data_dim, num_classes, inducing_x
