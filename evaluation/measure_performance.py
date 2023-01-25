'''
Script to compute performance metrics.
Works on cross-validated predictions that are saved as test_outputs.pickle by the training script.
'''

import pandas as pd
import os
import pickle
from typing import List, Tuple
from tqdm.auto import tqdm

PEPTIDE_START_STATE, PEPTIDE_END_STATE = 1, 50
PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE = 51, 100

BEST_CHECKPOINTS = [

    # ESM2 models on balanced dataset - layer 33.
    '../hyperparam_runs_esm2_650_balanced_33/0/1/1',
    '../hyperparam_runs_esm2_650_balanced_33/0/1/2',
    '../hyperparam_runs_esm2_650_balanced_33/0/1/3',
    '../hyperparam_runs_esm2_650_balanced_33/0/1/4',

    '../hyperparam_runs_esm2_650_balanced_33/1/7/0',
    '../hyperparam_runs_esm2_650_balanced_33/1/7/2',
    '../hyperparam_runs_esm2_650_balanced_33/1/7/3',
    '../hyperparam_runs_esm2_650_balanced_33/1/7/4',

    '../hyperparam_runs_esm2_650_balanced_33/2/9/0',
    '../hyperparam_runs_esm2_650_balanced_33/2/9/1',
    '../hyperparam_runs_esm2_650_balanced_33/2/9/3',
    '../hyperparam_runs_esm2_650_balanced_33/2/9/4',

    '../hyperparam_runs_esm2_650_balanced_33/3/4/0',
    '../hyperparam_runs_esm2_650_balanced_33/3/4/1',
    '../hyperparam_runs_esm2_650_balanced_33/3/4/2',
    '../hyperparam_runs_esm2_650_balanced_33/3/4/4',

    '../hyperparam_runs_esm2_650_balanced_33/4/3/0',
    '../hyperparam_runs_esm2_650_balanced_33/4/3/1',
    '../hyperparam_runs_esm2_650_balanced_33/4/3/2',
    '../hyperparam_runs_esm2_650_balanced_33/4/3/3',
]

def convert_path_to_peptide_borders(pred: List[int], start_state, stop_state, offset: int=0) -> List[Tuple[int,int]]:
    '''Given a sequence of states, find the borders of contiguous peptide segments.
       Offset adds a constant to all coordinates (1-based indexing in uniprot)
    '''

    seq_peptides = []
    is_peptide = False

    for pos, p in enumerate(pred):
        
        if p == start_state and not is_peptide: # open a new peptide
            is_peptide = True
            peptide_start = pos

        # Close the peptide at the position that has the stop state. (can restart peptide immediately without NO-peptide gap.)
        elif p == stop_state and is_peptide: #close the peptide
            is_peptide = False
            seq_peptides.append((peptide_start +offset, pos +offset))
        else:
            pass # for positions that are not start_state or stop_state, do nothing.

    # close the last peptide if same as sequence end.
    if is_peptide:
        seq_peptides.append((peptide_start +offset,pos +offset))
        
    return seq_peptides


def parse_coordinate_string(coordinate_string: str, merge_overlaps: bool=True) -> List[Tuple[int,int]]:
    
    peptides = []
    coordinates = coordinate_string.split(',')
    
    if coordinate_string == '':
        return []
    # Cases to handle
    # --------------------111111111-------- 
    # ----------------11111111------------- N-terminal overlap
    # ---------------------111------------- inside of peptide
    # ----------------1111111111111111----- contains peptide
    # -----------------------------111111-- C-terminal overlap
    coordinates_parsed = []
    for coords in coordinates:
        s, e = coords.split('-')
        s, e = s.lstrip('('), e.rstrip(')')
        coordinates_parsed.append((int(s), int(e)))

    # start to end, long to short.
    sort_fn = lambda x: (x[0], -(x[1]-x[0]))
    coordinates_sorted = sorted(coordinates_parsed, key = sort_fn)

    coordinates_merged = []
    if merge_overlaps:
        previous_end = -1
        previous_start = -1
        for start, end in coordinates_sorted:
            if start>=previous_end:
                # the new start position comes after the previous end. 
                # Save the old one and open a new peptide.
                coordinates_merged.append([previous_start, previous_end])

                previous_start = start
                previous_end = end
            else:
                # the new start position is contained in the previous peptide.
                # continue the previous peptide.
                previous_end = max(previous_end, end) # either expand or keep prev if this one is smaller
        
        # handle the last peptide.
        coordinates_merged.append([previous_start, previous_end])

        return coordinates_merged[1:] # we add (-1,-1) to the list in the loop.

    else:
        return coordinates_sorted



def get_counts_for_protein(true_start_stop: List[Tuple[int,int]], pred_start_stop: List[Tuple[int,int]], tolerance: int =3) -> Tuple[int,int,int]:
    '''
    Counts the true positives, false negatives and false positives for one peptide backbone.
    This function handles overlapping annotations by treating the full overlap group as "one peptide"
    during counting (=matched if any of the constituent peptides matched)
    '''

    # Cases where the code below fails.
    # Pred peptides empty.
    if len(pred_start_stop) == 0:
        return 0, len(true_start_stop), 0
    # True peptides empty
    if len(true_start_stop) == 0:
        return 0, 0, len(pred_start_stop)
    

    # 1. cluster the true peptides and iterate true peptide clusters.
    # A cluster is a group of overlapping peptides.
    # A cluster can only contribute one count, even if there are multiple peptides.
    # Because the models cannot handle overlaps, getting any of the overlapping peptides out is "good enough"
    starts, stops = zip(*true_start_stop)
    true_df = pd.DataFrame([starts, stops], index = ['start', 'stop']).T
    true_df = true_df.sort_values(['start', 'stop'], ascending=[True, False])
    true_df['group'] = (true_df['stop'].cummax().shift() < true_df['start']).cumsum()
    true_df['matched'] = False

    starts, stops = zip(*pred_start_stop)
    pred_df = pd.DataFrame([starts, stops], index = ['start', 'stop']).T
    pred_df['matched'] = False

    for idx, row in true_df.iterrows():
        true_start, true_stop = row['start'], row['stop']

        for idx, row in pred_df.iterrows():
            pred_start, pred_stop = row['start'], row['stop']
            start_match = pred_start >= true_start-tolerance and pred_start <= true_start + tolerance
            stop_match = pred_stop >= true_stop-tolerance and pred_stop <= true_stop + tolerance
            if start_match and stop_match:
                true_df.loc[idx, 'matched'] = True
                pred_df.loc[idx, 'matched'] = True
                break # no need to check rest. peptide need only match one.

    
    
    # collapse groups. only one need to be matched per group.
    true_matched = true_df.groupby('group')['matched'].any()

    tp = true_matched.sum() # tp = count matched True groups
    fn = len(true_matched) - tp # fn = count non-matched True groups
    fp = (~pred_df['matched']).sum() # fp = count non-matched Pred

    return tp, fn, fp


def compute_peptide_finding_metrics(true_start_stop: List[List[Tuple[int,int]]], pred_start_stop: List[List[Tuple[int,int]]], tolerance: int = 3, suffix: str = ''):
    '''Compute per-peptide precision/recall at a given cleavage site tolerance.
    This is very inefficient as we don't assume the peptides to be sorted in any meaningful order for now.

    Precision: recovered peptides/ predicted peptides
    Recall:    recovered peptides/ true peptides
    '''
    assert len(true_start_stop) == len(pred_start_stop)
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    for true_peptides, predicted_peptides in zip(true_start_stop, pred_start_stop): #iterates over proteins in batch.
        tp, fn, fp = get_counts_for_protein(true_peptides, predicted_peptides, tolerance)
        true_positives += tp
        false_negatives += fn
        false_positives += fp
    

    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f1 = (2 * precision * recall) / (precision+recall) if (precision+recall) >0 else 0

    return precision, recall, f1


def get_confusion_for_protein(true_start_stop: List[Tuple[int,int]], pred_start_stop: List[Tuple[int,int]], true_types: List[str], pred_types: List[str], tolerance: int =3):
    '''
    Counts the true positives, false negatives and false positives for one peptide backbone.
    This function handles overlapping annotations by treating the full overlap group as "one peptide"
    during counting (=matched if any of the constituent peptides matched)
    '''
    # Cases where the code below fails.
    # Pred peptides empty.
    if len(pred_start_stop) == 0:
        starts, stops = zip(*true_start_stop)
        true_df = pd.DataFrame([starts, stops, true_types], index = ['start', 'stop', 'type']).T
        true_df['true'] = true_df['type']
        true_df['pred'] = None
        return true_df[['start', 'stop', 'true', 'pred']]
    # True peptides empty
    if len(true_start_stop) == 0:
        starts, stops = zip(*pred_start_stop)
        pred_df = pd.DataFrame([starts, stops, pred_types], index = ['start', 'stop', 'type']).T
        pred_df['true'] = None
        pred_df['pred'] = pred_df['type']
        return pred_df[['start', 'stop', 'true', 'pred']]


    # 1. cluster the true peptides and iterate true peptide clusters.
    # A cluster is a group of overlapping peptides.
    # A cluster can only contribute one count, even if there are multiple peptides.
    # Because the models cannot handle overlaps, getting any of the overlapping peptides out is "good enough"
    starts, stops = zip(*true_start_stop)
    true_df = pd.DataFrame([starts, stops, true_types], index = ['start', 'stop', 'type']).T
    true_df = true_df.sort_values(['start', 'stop'], ascending=[True, False])
    true_df['group'] = (true_df['stop'].cummax().shift() < true_df['start']).cumsum()
    true_df['matched'] = False
    true_df['match_type'] = 'None' # column type str.

    starts, stops = zip(*pred_start_stop)
    pred_df = pd.DataFrame([starts, stops, pred_types], index = ['start', 'stop', 'type']).T
    pred_df['matched'] = False
    pred_df['match_type'] = 'None'


    for idx, row in true_df.iterrows():
        true_start, true_stop, true_type = row['start'], row['stop'], row['type']

        for idx, row in pred_df.iterrows():
            pred_start, pred_stop, pred_type = row['start'], row['stop'], row['type']
            start_match = pred_start >= true_start-tolerance and pred_start <= true_start + tolerance
            stop_match = pred_stop >= true_stop-tolerance and pred_stop <= true_stop + tolerance
            if start_match and stop_match:
                true_df.loc[idx, 'matched'] = True
                pred_df.loc[idx, 'matched'] = True
                true_df.loc[idx, 'match_type'] = pred_type
                pred_df.loc[idx, 'match_type'] = pred_type
                break # no need to check rest. peptide need only match one.

    
    # now we can simplify this to a "normal" classification result

    true_df['true'] = true_df['type']
    true_df['pred'] = true_df['match_type']

    pred_df['true'] = 'None'
    pred_df['pred'] = pred_df['type']

    pred_df = pred_df.loc[~pred_df['matched']]

    df_out = pd.concat([true_df, pred_df])[['start', 'stop', 'true', 'pred']]
    return df_out
    

def compute_table_for_confusion(df):
    '''Get a table that has true/pred for each peptide.'''
    trues = (df['true_peptides'] + df['true_propeptides']).tolist()
    preds = (df['pred_peptides'] + df['pred_propeptides']).tolist()

    true_types = (df['true_peptides'].apply(lambda x: ['Peptide' for y in x]) + df['true_propeptides'].apply(lambda x: ['Propeptide' for y in x])).tolist()
    pred_types = (df['pred_peptides'].apply(lambda x: ['Peptide' for y in x]) + df['pred_propeptides'].apply(lambda x: ['Propeptide' for y in x])).tolist()

    dfs = []

    for protein_name, true_peptides, predicted_peptides, true_peptide_types, pred_peptide_types in zip(df.index, trues, preds, true_types, pred_types):
        assert len(true_peptides) == len(true_peptide_types)
        assert len(predicted_peptides) == len(pred_peptide_types)

        df_prot = get_confusion_for_protein(true_peptides, predicted_peptides, true_peptide_types, pred_peptide_types, tolerance=3)
        df_prot['protein'] = protein_name
        dfs.append(df_prot)

    df = pd.concat(dfs)

    return df


    

def score_one_model(predictions_file, true_df):
    data = pickle.load(open(predictions_file, 'rb'))
    probs, preds, labels, names = data
    peptide_borders = [convert_path_to_peptide_borders(pred, start_state=1, stop_state=50, offset=1) for pred in preds]
    propeptide_borders = [convert_path_to_peptide_borders(pred, start_state=51, stop_state=100, offset=1) for pred in preds]

    prediction_df = pd.DataFrame({'pred_peptides': peptide_borders, 'pred_propeptides': propeptide_borders}, index=names)

    df = prediction_df.join(true_df[['true_peptides', 'true_propeptides', 'organism']])

    peptide_prediction_df = compute_table_for_confusion(df)
    
    metrics = []
    for tolerance in [0,1,2,3]:
        true = df['true_peptides'].tolist()
        pred = df['pred_peptides'].tolist()
        prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
        true = df['true_propeptides'].tolist()
        pred = df['pred_propeptides'].tolist()
        prec_pro, rec_pro, f1_pro = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
        true = (df['true_peptides'] + df['true_propeptides']).tolist()
        pred = (df['pred_peptides'] + df['pred_propeptides']).tolist()
        prec_all, rec_all, f1_all = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)

        metrics.append({
            'precision peptides': prec_pep,
            'recall peptides': rec_pep,
            'f1 peptides': f1_pep,
            'precision propeptides': prec_pro,
            'recall propeptides': rec_pro,
            'f1 propeptides': f1_pro,
            'precision all': prec_all,
            'recall all': rec_all,
            'f1 all': f1_all,
        })
    
    return metrics, peptide_prediction_df

def main():

    # for each model, I want (prec, recall, f1) * (0,1,2,3) * (pep, propep, merged)

    df = pd.read_csv('../data/uniprot_12052022_cv_5_50/labeled_sequences.csv', index_col='protein_id')
    df = df.fillna('') # empty coordinates would become nan.
    coordinate_strings = df['coordinates'].tolist()
    propeptide_coordinate_strings = df['propeptide_coordinates'].tolist()
    coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in coordinate_strings]
    propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in propeptide_coordinate_strings]
    df['true_peptides'] = coordinates
    df['true_propeptides'] = propeptide_coordinates

    peptide_prediction_dfs = []
    metrics_dfs = []
    for checkpoint in tqdm(BEST_CHECKPOINTS):

        metrics, peptide_prediction_df = score_one_model(os.path.join(checkpoint, 'test_outputs.pickle'), df)
        metrics_df = pd.DataFrame.from_dict(metrics)
        metrics_df.index = pd.MultiIndex.from_product([metrics_df.index, [checkpoint]], names=['tolerance', 'model'])
        metrics_dfs.append(metrics_df)

        peptide_prediction_df['model'] = checkpoint
        peptide_prediction_dfs.append(peptide_prediction_df)

    
    metrics_df = pd.concat(metrics_dfs).sort_index()

    peptide_prediction_df = pd.concat(peptide_prediction_dfs).set_index(['model', 'protein', 'start', 'stop'])
    peptide_prediction_df.to_csv('peptide_predictions_esm2_l33_balancedsplit.csv')
    
    means = metrics_df.groupby(level=0).mean()
    means.to_csv('crf_model_means_esm2_l33_balancedsplit.csv')
    metrics_df.to_csv('crf_model_all_cv_esm2_l33_balancedsplit.csv')


if __name__ == '__main__':
    main()