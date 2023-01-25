import os
from os.path import join as pjoin
import tempfile
import subprocess
import resource
import time
import urllib
import re
import urllib.request
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm
import pandas as pd
from measure_performance import compute_peptide_finding_metrics, parse_coordinate_string

PROJECT_DIR = os.path.abspath(pjoin(os.path.dirname(__file__), '..'))
RESULTS_DIR = pjoin(PROJECT_DIR, 'results')

#BIN_DIR = "/novo/users/jref/share/fegt/bin_with_peptideranker/"
BIN_DIR = "/novo/users/fegt/in-silico-peptidomics/baselines/"
PEPTIDE_LOCATOR_BIN_DIR = pjoin(BIN_DIR, "PeptideLocator")

PEPTIDE_LOCATOR_PROTEIN_DIR = pjoin(RESULTS_DIR, 'peptide_locator/{organism}')
PEPTIDE_LOCATOR_PROTEIN_FILE = pjoin(PEPTIDE_LOCATOR_PROTEIN_DIR, '{protein_id}.fasta.PeptideLocator')
SMART_DIR = pjoin(RESULTS_DIR, 'peptide_locator/smart/{organism}')
SMART_FILE = pjoin(SMART_DIR, '{protein_id}.txt')
PEPTIDE_LOCATOR_BLAST = pjoin(PEPTIDE_LOCATOR_BIN_DIR, "BlastFiles/nr98rr1.fas")
SMART_LENGTH_CUTOFF = 30
BLANK_SMART_FILE = "\nUSER_PROTEIN_ID = {protein_id}\nNUMBER_OF_FEATURES_FOUND=0\n\n"

################################################################################
# PeptideLocator
################################################################################
def _get_smart(protein_id, protein_sequence):
    if len(protein_sequence) <= SMART_LENGTH_CUTOFF:
        return (BLANK_SMART_FILE.format(protein_id=protein_id), None)
    smart_regex = re.compile('^-- SMART RESULT', re.M)
    submit_url = "http://smart.embl.de/smart/show_motifs.pl"
    job_status_url = "http://smart.embl.de/smart/job_status.pl?jobid={}"
    data = {'SEQUENCE': protein_sequence, 'TEXTONLY': 1}
    data = urllib.parse.urlencode(data).encode('ascii')
    #  _f = open('smart.progress', 'a')
    error = None
    while True:
        # case 1) the server responds :D
        result = urllib.request.urlopen(submit_url, data=data).read().decode('utf-8')
        if re.search(smart_regex, result):
            return (result, error)

        # case 2) we are in a queue
        job_id = re.search(r"job_status\.pl\?jobid=(\d+\w+)", result)
        if job_id:  # we are in a queue
            job_id = job_id.groups()[0]
            while True:
                time.sleep(5)
                result = urllib.request.urlopen(
                        job_status_url.format(job_id)).read().decode('utf-8')
                if re.search(smart_regex, result):
                    return (result, error)
                if re.search('Job {} is not in the queue'.format(job_id), result):
                    error = "job_id not in queue"
                    break

        # case 3) we are not in a que, try again
        error = re.search("<title>(.+?)</title>", result).groups()[0]

        if error == 'SMART: Too many jobs from this IP':
            time.sleep(30)
        #  elif error == 'SMART: Error':
        else:
            time.sleep(10)



def _tf(protein_id, suffix, debug=False):
    if debug:
        tmp_file = open(pjoin(PEPTIDE_LOCATOR_BIN_DIR, 'tmp', suffix), 'w')
    else:
        tmp_file = tempfile.NamedTemporaryFile(
            prefix='PeptideLocator_{}_'.format(protein_id),
            suffix='.{}'.format(suffix),
            dir='{}/tmp/'.format(PEPTIDE_LOCATOR_BIN_DIR),
            mode='w+', encoding='utf-8')
    #  common_prefix = os.path.commonprefix((tmp_file.name, PEPTIDE_LOCATOR_BIN_DIR))
    return tmp_file


def peptide_locator(protein_id, protein_sequence, organism,
                    max_memory=100 * 10 ** 9):
    # set limits on the process to prevent the server from crashing
    # ./Porter is known to on rare occations use 100GB+ and 3hr+ CPU time
    # where in 99% of the cases it uses >1GB and >5sek
    # this makes the software trow a MemoryError occasionaly
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))

    peptide_locator_file = PEPTIDE_LOCATOR_PROTEIN_FILE.format(
        organism=organism, protein_id=protein_id)
    # check if result is already calcuated
    if os.path.isfile(peptide_locator_file):
        predictions = open(peptide_locator_file).read().split(' ')
        return tuple(map(float, predictions))
    else:
        _parent_folder = PEPTIDE_LOCATOR_PROTEIN_DIR.format(organism=organism)
        os.makedirs(_parent_folder, exist_ok=True)

    ########################################
    # Predict Domains (SMART)
    # io (network) heavy part
    ########################################
    os.makedirs(SMART_DIR.format(organism=organism), exist_ok=True)
    smart_file = SMART_FILE.format(organism=organism, protein_id=protein_id)
    if os.path.isfile(smart_file):
        smart_str = open(smart_file).read()
    else:
        smart_str = _get_smart(protein_id, protein_sequence)[0]
        open(smart_file, 'w').write(smart_str)

    _tmp_file = lambda ext: _tf(protein_id, ext)
    ########################################
    # CPU heavy part
    ########################################
    protein_length = len(protein_sequence)
    #  with contextlib.ExitStack() as s:
    with _tmp_file('fasta') as _fasta, \
            _tmp_file('flatblast') as _flatblast, \
            _tmp_file('dataset') as _dataset, \
            _tmp_file('dataset2') as _dataset2, \
            _tmp_file('biodataset') as _biodataset:
        _fasta.write(">{}\n{}\n".format(protein_id, protein_sequence))
        _fasta.flush()

        #######################################
        # Predict SS, SA, SM
        #######################################
        # blast
        BLASTPGP_BIN = '/novo/users/fegt/random/in-silico-peptidomics/baselines/blast-2.2.26/bin/blastpgp'
        with _tmp_file('tmp') as _tmp, _tmp_file('chk') as _chk, \
                _tmp_file('blastpgp') as _blastpgp:
            blast_cmd_shared = ["-F", "T", "-b", "3000", "-e", "0.001",
                                "-h", "1e-10", "-i", _fasta.name,
                                "-d", PEPTIDE_LOCATOR_BLAST]
            blast_cmd1 = [BLASTPGP_BIN, "-o", _tmp.name, "-C", _chk.name, "-j", "2"]
            blast_cmd2 = [BLASTPGP_BIN, "-o", _blastpgp.name, "-R", _chk.name, "-j", "1"]

            subprocess.call(blast_cmd1 + blast_cmd_shared, cwd=PEPTIDE_LOCATOR_BIN_DIR,
                            stderr=subprocess.PIPE)
            subprocess.call(blast_cmd2 + blast_cmd_shared, cwd=PEPTIDE_LOCATOR_BIN_DIR,
                            stderr=subprocess.PIPE)

            process_blast = ('./process-blast.pl', _blastpgp.name, _flatblast.name,
                             _fasta.name)

            # this script creates both flatblast and flatblast.app
            # TODO: are both nessesary?
            subprocess.call(process_blast, cwd=PEPTIDE_LOCATOR_BIN_DIR)

        print("1 20 3", protein_sequence, sep='\n', end='\n', file=_dataset, flush=True)
        print("1", protein_length, protein_sequence, sep='\n', end='\n',
              file=_dataset2, flush=True)

        # this wierd protein took 3 hours to complete :S...
        porter_cmd = "./{} {}_AI_model.txt {} {}"
        ss_sa_sm = []
        for (tool, ext, input_file) in (("./Porter", 'porter', _dataset.name),
                                        ("./PaleAle", "paleale", _dataset2.name),
                                        ("./Porter+", "motifs", _dataset2.name)):
            porter_cmd = (tool, "{}_AI_model.txt".format(tool), input_file, _flatblast.name)
            p = subprocess.Popen(porter_cmd, cwd=PEPTIDE_LOCATOR_BIN_DIR,
                                 stderr=subprocess.PIPE)
            error = p.stderr.read().decode('utf8')
            if error != '':
                raise IOError(error)
            out_file_name = "{}.{}".format(input_file, ext)
            with open(out_file_name) as out_file:
                ss_sa_sm.append(out_file.readlines()[2].rstrip())
            os.unlink(out_file_name)
        ss, sa, sm = ss_sa_sm
        sa = sa.split(' ')
        sm = sm.split(' ')

        ########################################
        # Predict Disorder
        ########################################
        iupred_cmd = ("./iupred", _fasta.name, "long")
        p_iupred = subprocess.Popen(iupred_cmd, stdout=subprocess.PIPE,
                                    cwd=pjoin(PEPTIDE_LOCATOR_BIN_DIR, 'iupred'),
                                    stderr=subprocess.PIPE)
        iu = []
        for line in p_iupred.stdout:
            line = line.decode('utf-8')
            if not re.search(r'^\s*#', line):
                index, aa, iupred_prediction = line.rstrip('\n').split()
                iu.append(iupred_prediction)

        ########################################
        # Split into Domain/Non-domain and Write Dataset
        ########################################
        #  seq = list(protein_sequence)
        domains = [0 for i in range(len(protein_sequence))]
        #  smart = {}

        starts = list(map(int, re.findall(r'START=(.+)', smart_str)))
        ends = list(map(int, re.findall(r'END=(.+)', smart_str)))
        if len(starts) and len(ends):
            for (start, end) in zip(starts, ends):
                for i in range(start - 1, end - 1):
                    domains[i] = 1

        previous_aa = domains[0]
        start = 0
        features = []
        for index, aa_type in enumerate(domains):
            if aa_type != previous_aa:
                features.append((start + 1, index, previous_aa))
                start = index
                previous_aa = aa_type
        if features:
            try:
                features.append((features[-1][1] + 1, len(domains), aa_type))
            except:
                # this should not fail any longer, but better safe than sorry
                print(protein_id)
        else:
            # there are no domains
            features = [(1, len(domains), aa_type)]

        ########################################
        # Make biodataset
        ########################################
        print(len(features), end='\n', file=_biodataset)
        #  dataset_name = os.path.basename(_fasta.name)
        for (start, end, domain_flag) in features:
            if domain_flag:
                _info = '{}.fasta.d.{}.{}'.format(protein_id, start, end)
            else:
                _info = '{}.fasta.nd.{}.{}'.format(protein_id, start, end)
            print(_info, end='\n', file=_biodataset)

            print(end - start + 1, end='\n', file=_biodataset)
            print(*protein_sequence[start - 1:end], sep=' ', end='\n', file=_biodataset)
            print(*ss[start - 1:end], sep=' ', end='\n', file=_biodataset)
            print(*sa[start - 1:end], sep=' ', end='\n', file=_biodataset)
            print(*sm[start - 1:end], sep=' ', end='\n', file=_biodataset)
            print(*iu[start - 1:end], sep=' ', end='\n', file=_biodataset)
            print(*([0] * (end - start + 1)), sep=' ', end='\n', file=_biodataset)
            print("{}\n".format(domain_flag), end='\n', file=_biodataset)
        _biodataset.flush()

        ########################################
        # Predict Bioactivity
        ########################################
        peptide_locator_cmd = ("./PeptideLocator", "PeptideLocator_model.txt",
                               _biodataset.name, "0.5")
        subprocess.call(peptide_locator_cmd, cwd=PEPTIDE_LOCATOR_BIN_DIR)

        peptide_locator_predictions = []
        peptide_locator_prediction_file = '{}.BIOpred'.format(_biodataset.name)
        with open(peptide_locator_prediction_file) as _f:
            for domain_chunk in _f.read().rstrip().split('\n\n'):
                for score in domain_chunk.split('\n')[-1].rstrip().split(' '):
                    peptide_locator_predictions.append(float(score))

        # cleanup
        os.unlink(peptide_locator_prediction_file)
        os.unlink('{}.app'.format(_flatblast.name))

        # dump result file, to make rerun fast
        with open(peptide_locator_file, 'w') as f:
            f.write(' '.join(map(str, peptide_locator_predictions)))

        return peptide_locator_predictions


def convert_binary_probs_to_peptide_borders(probs: List[np.ndarray], threshold: float = 0.5, offset: int=0) -> List[List[Tuple[int,int]]]:
    '''Given a sequence of probabilities, find the borders of contiguous segments that are higher than the threshold.'''
    peptides = []

    # process each sequence.
    for seq_probs in probs:
        seq_peptides = []
        is_peptide = False

        for pos, p in enumerate(seq_probs):
            
            if p>=threshold and not is_peptide: # open a new peptide
                is_peptide = True
                peptide_start = pos
            elif p<threshold and is_peptide: #close the peptide
                is_peptide = False
                seq_peptides.append((peptide_start +offset,pos-1 +offset))#peptide ended 1 position before.

        # close the last peptide if same as sequence end.
        if is_peptide:
            seq_peptides.append((peptide_start +offset,pos+offset))
        
        peptides.append(seq_peptides)

    return peptides


def get_preds():
    data = pd.read_csv('../data/uniprot_12052022_cv_5_50/labeled_sequences.csv', index_col='protein_id')
    data = data.fillna('')
    partition_df = pd.read_csv('../data/uniprot_12052022_cv_5_50/balanced_motifs/graphpart_assignments.csv', index_col='AC')
    data = data.loc[data.index.isin(partition_df.index)]
    data = data.reset_index()

    data['peptidelocator_output'] = None
    probabilities= []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        
        seq = row['sequence']
        name = row['protein_id']
        res = peptide_locator(name, seq, 'any')
        probabilities.append(np.array(res))
        data.loc[idx, 'peptidelocator_output'] = ';'.join([str(x) for x in res])
        data.to_csv('peptidelocator_predictions.csv')

    
    return probabilities, data


def main():

    probabilities, df = get_preds()
    

    # now compute performance.
    coordinate_strings = df['coordinates'].tolist()
    propeptide_coordinate_strings = df['propeptide_coordinates'].tolist()
    coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in coordinate_strings]
    propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in propeptide_coordinate_strings]
    df['true_peptides'] = coordinates
    df['true_propeptides'] = propeptide_coordinates

    
    # 1. convert output probs into thresholded preds and get contiguous positive segments
    peptide_borders = convert_binary_probs_to_peptide_borders(probabilities, threshold=0.5, offset=1)


    # import sys # dirty fix.
    # sys.path.append('src')
    # from utils.metrics_cleaned import probabilities_to_paths, convert_path_to_peptide_borders, compute_peptide_finding_metrics
    # from utils.crf_label_utils import parse_coordinate_string
    
    
    #paths = probabilities_to_paths(probabilities, threshold=0.5)
    #pred_borders = [convert_path_to_peptide_borders(x) for x in paths]
    #true_borders = [parse_coordinate_string(x) for x in data['coordinates'].tolist()]


    true = (df['true_peptides'] + df['true_propeptides']).tolist()
    true_pep = df['true_peptides'].tolist()
    pred = peptide_borders
        

    metrics = []
    for tolerance in [0,1,2,3]:
        # true = df['true_peptides'].tolist()
        # pred = df['pred_peptides'].tolist()
        # prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
        # true = df['true_propeptides'].tolist()
        # pred = df['pred_propeptides'].tolist()
        # prec_pro, rec_pro, f1_pro = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
        prec_all, rec_all, f1_all = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
        prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(true_pep, pred, tolerance=tolerance)

        metrics.append({
            'precision peptides': prec_pep,
            'recall peptides': rec_pep,
            'f1 peptides': f1_pep,
            # 'precision propeptides': prec_pro,
            # 'recall propeptides': rec_pro,
            'precision all': prec_all,
            'recall all': rec_all,
            'f1 all': f1_all,
        })



    df = pd.DataFrame.from_dict(metrics)
    df.to_csv('peptidelocator.csv')


    # threshold experiment
    dfs = []
    for threshold in np.arange(0, 1.05, 0.05):
        peptide_borders = convert_binary_probs_to_peptide_borders(probabilities, threshold=threshold, offset=1)
        pred = peptide_borders
        metrics = []
        for tolerance in [0,1,2,3]:
            # true = df['true_peptides'].tolist()
            # pred = df['pred_peptides'].tolist()
            # prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
            # true = df['true_propeptides'].tolist()
            # pred = df['pred_propeptides'].tolist()
            # prec_pro, rec_pro, f1_pro = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
            prec_all, rec_all, f1_all = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
            prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(true_pep, pred, tolerance=tolerance)

            metrics.append({
                'precision peptides': prec_pep,
                'recall peptides': rec_pep,
                'f1 peptides': f1_pep,
                'precision all': prec_all,
                'recall all': rec_all,
                'f1 all': f1_all,
            })
        
        df = pd.DataFrame.from_dict(metrics)
        df.index = pd.MultiIndex.from_product([df.index, [threshold]], names=['tolerance', 'threshold'])
        dfs.append(df)

    pd.concat(dfs).to_csv('peptidelocator_thresholds.csv')
    


    # metrics_all = {}
    # metrics_human = {}
    # metrics_mouse = {}
    # for tol in 0,1,2,3:
    #     all = compute_peptide_finding_metrics(true_borders, pred_borders, tolerance=tol)
    #     human = compute_peptide_finding_metrics(np.array(true_borders, dtype=object)[data['organism'] == 'Homo sapiens (Human)'], np.array(pred_borders, dtype=object)[data['organism'] == 'Homo sapiens (Human)'], tolerance=tol)
    #     mouse = compute_peptide_finding_metrics(np.array(true_borders, dtype=object)[data['organism'] == 'Mus musculus (Mouse)'], np.array(pred_borders, dtype=object)[data['organism'] == 'Mus musculus (Mouse)'], tolerance=tol)
    #     metrics_all.update(all)
    #     metrics_human.update(human)
    #     metrics_mouse.update(mouse)

    # df = pd.DataFrame.from_dict([metrics_all, metrics_human, metrics_mouse])
    # df.index=['All', 'Human', 'Mouse']
    # df.to_csv('results/peptidelocator_peptide_metrics.csv')
    

if __name__ == '__main__':
    main()

#('./Porter', './Porter_AI_model.txt', '/novo/users/fegt/random/in-silico-peptidomics/baselines/PeptideLocator/tmp/PeptideLocator_Q86YL7_pid0em32.dataset', '/novo/users/fegt/random/in-silico-peptidomics/baselines/PeptideLocator/tmp/PeptideLocator_Q86YL7_f1qvoh7m.flatblast')