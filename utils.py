"""Load TxGNN data."""
import numpy as np
import os

from datasets import load_dataset, Dataset, DatasetDict

DATA_PATH = '/n/home06/jschwarz/data/TxGNN/pretrained_mine/complex_disease'

DD_ETYPES = {
    'did' : ('drug', 'indication', 'disease'), 
    'dod' : ('drug', 'off-label use', 'disease'),
    'dcd' : ('drug', 'contraindication', 'disease'), 
    'drid': ('disease', 'rev_indication', 'drug'), 
    'drod': ('disease', 'rev_off-label use', 'drug'),
    'drcd': ('disease', 'rev_contraindication', 'drug'), 
}


def load_split(path, mode, merge_str):
    outfile = np.load(os.path.join(DATA_PATH, path))

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

def load_txgnn_dataset(edge_type):
    print(edge_type)
    if 'did' == edge_type:
        merge_str = 'Is {} an effective treatment for {}?'
    elif 'dod' == edge_type:
        merge_str = 'Is {} effective for off-label use on {}?'
    elif 'dcd' == edge_type:
        merge_str = 'Should {} be avoided for patients suffering from {}?'
    else:
        assert False, 'Reverse cases not yet used'

    e_type = DD_ETYPES[edge_type]

    train_dataset = load_split('separate/{}.npz'.format('_'.join(e_type)), 'train', merge_str)
    valid_dataset = load_split('separate/{}.npz'.format('_'.join(e_type)), 'valid', merge_str)
    test_dataset = load_split('separate/{}.npz'.format('_'.join(e_type)), 'test', merge_str)
    full_dataset = DatasetDict({'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})

    return full_dataset
