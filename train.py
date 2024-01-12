"""Train and Fine-tune TxGNN model."""
from absl import app
from absl import flags

from txgnn import TxData
from txgnn import TxGNN
from txgnn import TxEval

import datetime

DATA_PATH = '/home/jos1479/data/'

FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_boolean('run_pretraining', False, 'Whether to use wandb.')
flags.DEFINE_boolean('run_finetuning', True, 'Whether to use wandb.')

flags.DEFINE_integer('pretrain_n_epochs', 2, 'Number of pre-training epochs.', lower_bound=-1)
flags.DEFINE_integer('pretrain_n_steps', -1, 'Number of pre-training steps.', lower_bound=-1)
flags.DEFINE_integer('pretrain_batch_size', 4, 'Pre-training Batch size.', lower_bound=1)

flags.DEFINE_integer('finetune_n_epochs', 1, 'Number of finetuning steps.', lower_bound=1)
flags.DEFINE_integer('finetune_batch_size', -1, 'Finetuning Batch size.', lower_bound=-1)
flags.DEFINE_boolean('finetune_dist_mult_only', False, 'Finetuning only DistMult predictor.')

# Model
flags.DEFINE_boolean('proto', True, 'Use Inter-disease prototypes?')
flags.DEFINE_integer('n_protos', 3, 'Inter-disease prototypes.', lower_bound=0)
flags.DEFINE_integer('model_dim', 32, 'Input/Hidden/Output embedding size.', lower_bound=1)

# Misc
flags.DEFINE_boolean('use_wandb', False, 'Whether to use wandb.')
flags.DEFINE_enum('data_split', 'random', 
                  ['adrenal_gland', 'anemia', 'cardiovascular', 'cell_proliferation', 
                   'complex_disease', 'mental_health', 'random'], 'Dataset split.')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_string('checkpoint', './checkpoints/model_ckpt', 'Checkpoint location.')
flags.DEFINE_string('exp_name', 'debug', 'Experiment name.')


def main(argv):
    assert not (FLAGS.run_pretraining and FLAGS.run_finetuning), 'Only one field can be specified'

    data = TxData(data_folder_path=DATA_PATH)
    data.prepare_split(split=FLAGS.data_split, seed = FLAGS.seed, no_kg = False)
    #id_mapping = data.retrieve_id_mapping()
    print('Finished data preparation.')

    gnn_model = TxGNN(
        data = data,  
        weight_bias_track = FLAGS.use_wandb,
        proj_name = 'TxGNNv2',
        exp_name = FLAGS.exp_name)

    if FLAGS.run_pretraining:
        print('Initialize model')
        gnn_model.model_initialize(
            n_hid = FLAGS.model_dim, # number of hidden dimensions
            n_inp = FLAGS.model_dim, 
            n_out = FLAGS.model_dim, 
            proto = FLAGS.proto, # whether to use metric learning module
            proto_num = FLAGS.n_protos, # number of similar diseases to retrieve for augmentation
            attention = False, # use attention layer (if use graph XAI, we turn this to false)
            sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
            bert_measure = 'disease_name', # type of bert embeddings, choose from ['disease_name, 'v1']
            agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
            exp_lambda = 0.7, # parameter of inflated exponential pdf. Equation 17.
            num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
            walk_mode = 'bit', # for protein_random_walk sim_measure, define how walk mode from ['bit', 'prob']
            path_length = 2 # for protein_random_walk sim_measure, define path length
        )

        path = './checkpoints/model_ckpt_{}_{}'.format(FLAGS.exp_name, str(datetime.datetime.now()))
        kwargs = {
            'n_epoch': None, 
            'n_steps': None,
            'learning_rate': 1e-3,
            'batch_size': FLAGS.pretrain_batch_size, 
            'train_print_per_n': 20,
            'checkpint_path': path,         
        }
        # Allow specification of either steps or epochs
        if FLAGS.pretrain_n_steps == -1:
            assert FLAGS.pretrain_n_epochs > 0
            kwargs['n_epoch'] = FLAGS.pretrain_n_epochs
        else:
            assert FLAGS.pretrain_n_epochs == -1
            kwargs['n_steps'] = FLAGS.pretrain_n_steps

        # Start training
        print('Start model pretraining.')
        gnn_model.pretrain(**kwargs)
    else:
        # To load a pretrained model: 
        print('Skipping pre-training, loading model from {}'.format(FLAGS.checkpoint))
        gnn_model.load_pretrained(FLAGS.checkpoint)

    if FLAGS.run_finetuning:
        print('Start model finetuning.')
        # Change to n_epoch = 500 when you use it
        gnn_model.finetune(
            n_epoch = FLAGS.finetune_n_epochs,
            batch_size = FLAGS.finetune_batch_size,
            learning_rate = 5e-4,
            train_print_per_n = 5,
            valid_per_n = 20,
            finetune_dist_mult_only = FLAGS.finetune_dist_mult_only)
        print('Done model finetuning.')

if __name__ == '__main__':
  app.run(main)
