'''
CRF train loop.
- no marginals
- no train metrics
'''
import json
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from .models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from .utils import add_dict_to_writer, PrecomputedCSVForOverlapCRFDataset
#from .utils.metrics_cleaned import compute_metrics, compute_metrics_with_propeptides
from .utils.manuscript_metrics import compute_all_metrics
from torch.optim import Adam
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
global_step = 0


def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4]) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type)

    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=1)

    return train_loader, valid_loader, test_loader


def get_model(args: argparse.Namespace):

    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size, 
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if args.label_type == 'simple_with_propeptides' else 2,
            dropout_input=args.dropout,
            num_states= 3 if args.label_type == 'simple_with_propeptides' else 2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size, 
            dropout_conv1=args.conv_dropout,
        )

    # NOTE just use already existing CLI args with names that don't really match. Works.
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size = args.embedding_dim,
            hidden_size= args.hidden_size,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 121 if 'with_propeptides' in args.label_type else 61,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def train(args, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], is_initiated: bool = False):
    global global_step
    global_step = 0
    train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions)


    if not is_initiated:
        # when we run in nested CV, we need to do this outside of train() to avoid reinitialization errors.
        url = "tcp://localhost:12355"
        torch.distributed.init_process_group(backend="nccl", init_method = url, world_size=1, rank=0)


    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=False,
        flatten_parameters=False,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        move_params_to_cpu =True,  # enable cpu offloading
        move_grads_to_cpu = True,
    )

    model = get_model(args)

    # NOTE FSDP does not support non-trainable weights yet. CRF has some.
    # https://github.com/pytorch/pytorch/issues/75943
    model = FSDP(model, **fsdp_params)


    model.feature_extractor.biLSTM.flatten_parameters()
    # model = get_model(args)
    # model.to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)
    writer = SummaryWriter(args.out_dir)

    previous_best = -100000000000

    for epoch in range(args.epochs):

        train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(train_loader, model, optimizer, writer, do_train=True)
        #train_metrics = compute_crf_metrics(train_probs, train_preds, train_peptides, train_labels)
        #train_metrics = metrics_fn(train_peptides, train_preds)
        #add_dict_to_writer(train_metrics, writer, global_step, prefix='Train')

        valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(valid_loader, model, optimizer, writer, do_train=False)
        #valid_metrics_old = compute_crf_metrics(valid_probs, valid_preds, valid_peptides, valid_labels)#, organism=valid_loader.dataset.data['organism'])
        #valid_metrics = metrics_fn(valid_peptides, valid_preds, valid_loader.dataset.data['organism'])
        valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
        add_dict_to_writer(valid_metrics, writer, global_step, prefix='Valid')
        writer.add_scalar('Valid/loss', valid_loss, global_step=global_step)


        print(f'Epoch {epoch} completed. Validation loss {valid_loss:.2f}')

        stopping_metric = (valid_metrics['f1 peptides'] + valid_metrics['f1 propeptides'])/2#(valid_metrics['F1 +- 3 peptide'] + valid_metrics['F1 +- 3 propeptide'])/2
        if stopping_metric > previous_best:
            previous_best = stopping_metric
            best_val_metrics = valid_metrics
            pickle.dump((valid_probs, valid_preds, valid_labels, valid_loader.dataset.names), open(os.path.join(args.out_dir, 'valid_outputs.pickle'), 'wb'))
            valid_metrics['epoch'] = epoch # keep track of best early stopping.
            json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))

            # valid_metrics = metrics_fn(valid_peptides, valid_preds, valid_loader.dataset.data['organism'])
            # valid_metrics['epoch'] = epoch # keep track of best early stopping.
            # json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics_old.json'), 'w'), indent=2)
    
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
    test_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, optimizer, writer, do_train=False)
    #test_metrics = compute_crf_metrics(test_probs, test_preds, test_peptides, test_labels, organism=test_loader.dataset.data['organism'])
    #test_metrics = metrics_fn(test_peptides, test_preds, test_loader.dataset.data['organism'])
    test_metrics = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]
    add_dict_to_writer(test_metrics, writer, global_step, prefix='Test')
    writer.add_scalar('Test/loss', test_loss, global_step=global_step)
    print('Test complete.')
    pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names), open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)

    return best_val_metrics, test_metrics

    

def run_dataloader(loader: torch.utils.data.DataLoader, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    writer: SummaryWriter,
                    do_train: bool = True,
                ) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''
    Run a dataloader through the model. Collect predicted probabilitities and
    true labels. Can be used both for training and prediction.
    '''
    global global_step

    true = [] # peptide coordinates
    labels = [] # labels made from coordinates
    probs = [] # per-position probabilities
    preds = [] # viterbi paths
    epoch_loss = []

    if do_train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(loader):
        
        model.zero_grad()

        embeddings, mask, label, peptides= batch
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if do_train:
            pos_probs, pos_preds, loss = model(embeddings, mask, label, skip_marginals=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.25 )
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train/loss', loss.item(), global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                pos_probs, pos_preds, loss = model(embeddings, mask, label)

        true.extend(peptides)
        probs.append(pos_probs.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        preds.extend(pos_preds)
        epoch_loss.append(loss.item())


    epoch_loss = sum(epoch_loss)/len(epoch_loss)

    return epoch_loss, probs, preds, true, labels





def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Embeddings dir produced by `extract.py`', default = '/data3/fegt_data/embeddings/')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/uniprot_12052022_cv_5_50/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/uniprot_12052022_cv_5_50/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1280)

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='train_run')
    p.add_argument('--epochs', type=int, default=30, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=100, help='samples that will be processed in parallel')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=64)

    p.add_argument('--label_type', type=str, default='multistate_with_propeptides')

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args


if __name__ == '__main__':
    train(parse_arguments())