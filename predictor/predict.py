'''
Script to run the model from commandline to be used on biolib.
'''
import argparse
import torch
import utils
from write_markdown_output import write_fancy_output
from concurrent.futures import ProcessPoolExecutor
import os
import json
from tqdm.auto import tqdm

MODEL_LIST_ESM1B = [
    'checkpoints_esm1b/0/1/model.pt',
    'checkpoints_esm1b/0/2/model.pt',
    'checkpoints_esm1b/0/3/model.pt',
    'checkpoints_esm1b/0/4/model.pt',
    'checkpoints_esm1b/1/0/model.pt',
    'checkpoints_esm1b/1/2/model.pt',
    'checkpoints_esm1b/1/3/model.pt',
    'checkpoints_esm1b/1/4/model.pt',
    'checkpoints_esm1b/2/0/model.pt',
    'checkpoints_esm1b/2/1/model.pt',
    'checkpoints_esm1b/2/3/model.pt',
    'checkpoints_esm1b/2/4/model.pt',
    'checkpoints_esm1b/3/0/model.pt',
    'checkpoints_esm1b/3/1/model.pt',
    'checkpoints_esm1b/3/2/model.pt',
    'checkpoints_esm1b/3/4/model.pt',
    'checkpoints_esm1b/4/0/model.pt',
    'checkpoints_esm1b/4/1/model.pt',
    'checkpoints_esm1b/4/2/model.pt',
    'checkpoints_esm1b/4/3/model.pt',
]
MODEL_LIST_ESM2 = [
    'checkpoints_esm2/0/1/model.pt',
    'checkpoints_esm2/0/2/model.pt',
    'checkpoints_esm2/0/3/model.pt',
    'checkpoints_esm2/0/4/model.pt',
    'checkpoints_esm2/1/0/model.pt',
    'checkpoints_esm2/1/2/model.pt',
    'checkpoints_esm2/1/3/model.pt',
    'checkpoints_esm2/1/4/model.pt',
    'checkpoints_esm2/2/0/model.pt',
    'checkpoints_esm2/2/1/model.pt',
    'checkpoints_esm2/2/3/model.pt',
    'checkpoints_esm2/2/4/model.pt',
    'checkpoints_esm2/3/0/model.pt',
    'checkpoints_esm2/3/1/model.pt',
    'checkpoints_esm2/3/2/model.pt',
    'checkpoints_esm2/3/4/model.pt',
    'checkpoints_esm2/4/0/model.pt',
    'checkpoints_esm2/4/1/model.pt',
    'checkpoints_esm2/4/2/model.pt',
    'checkpoints_esm2/4/3/model.pt',
]
# ESM_DIR = 'esm'

# MODEL_LIST = ['../'+x for x in MODEL_LIST] # local debug

def main():
    parser = argparse.ArgumentParser('PeptideCRF peptide prediction tool', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fastafile', '-ff' ,'-fasta', type=str, help='Amino acid sequences to predict in FASTA format.', required=True)
    parser.add_argument('--output_dir', '-od', type=str, help='Path at which to save the output files. Will be created if not existing already.', required=True)
    parser.add_argument('--batch_size', '-bs', type=int, help='Batch size (number of sequences).', default=10)
    parser.add_argument('--output_fmt', '-of', default='img', const='esm2', nargs='?', choices=['img', 'json'], help='The output format. img also includes the json file.')
    parser.add_argument('--esm', default='esm2', const='esm2', nargs='?', choices=['esm2', 'esm1b'], help ='Which ESM version to use.')
    parser.add_argument('--esm_pt', default=None,  help ='Optional path to a ESM .pt checkpoint. If not specified, uses the default loading and caching of the esm package.')


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_dict = {}
    out_dict['INFO'] = {}
    out_dict['PREDICTIONS'] = {}

    compute_marginals = args.output_fmt == 'img'
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. prepare data
    ids, seqs = utils.parse_fasta(args.fastafile)
    out_dict['INFO']['size'] = len(ids)

    # 2. prepare embedder and models
    #embeddings = utils.esm_embed(seqs, progress_bar=True, esm=args.esm, local_esm_path=args.esm_pt)
    embedder = utils.ESMEmbedder(args.esm, args.esm_pt)
    #embeddings = torch.rand(len(seqs), 500, 1280 )
    batches = utils.batchify_sequences(seqs, args.batch_size)
    models = utils.load_models(MODEL_LIST_ESM1B if args.esm == 'esm1b' else MODEL_LIST_ESM2)
    crf = utils.combine_crf(models)
    crf.to(device)

    # 3. predict
    all_probs = [] # list of len (n_seqs)
    all_peptides = [] # list of len (n_seqs)
    all_propeptides = [] # list of len (n_seqs)
    all_preds = [] # list of len (n_seqs)

    with torch.no_grad():
        for b in tqdm(batches, desc='Predicting...'):

            seqs, mask = b
            embeddings = embedder(seqs)

            embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
            mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
            
            embeddings = embeddings.to(device)
            mask = mask.to(device)

            batch_emissions = []
            batch_marginals = []
            for model in models:
                model.to(device)
                model.eval()

                features = model.feature_extractor(embeddings.permute(0,2,1), mask)
                emissions = model.features_to_emissions(features)
                emissions = model._repeat_emissions(emissions)

                if compute_marginals:
                    marginals = model.crf.compute_marginal_probabilities(emissions, mask)
                    batch_marginals.append(marginals.cpu())
                
                batch_emissions.append(emissions.cpu())

            batch_emissions = torch.stack(batch_emissions).mean(dim=0)
            ensemble_paths, ensemble_path_llhs = crf.decode(batch_emissions.to(device), mask.byte().to(device), top_k=1)

            # postprocess on the fly
            for path in ensemble_paths:
                peptides = utils.convert_path_to_peptide_borders(path, start_state=1, stop_state=50, offset=1)
                propeptides = utils.convert_path_to_peptide_borders(path, start_state=51, stop_state=100, offset=1)
                preds = utils.simplify_preds([path])[0]
                all_peptides.append(peptides)
                all_propeptides.append(propeptides)
                all_preds.append(preds)


            if compute_marginals:
                batch_marginals = torch.stack(batch_marginals).mean(dim=0)
                marginal_list = []
                for i in range(batch_marginals.shape[0]):
                    real_len = int(mask[i].sum().item())
                    marginal_list.append(batch_marginals[i, :real_len].cpu().numpy())

                # marginal_list is len(batch)
                probs = utils.simplify_probs(marginal_list)
                all_probs.extend(probs)

            


    # 3. make json-structured output.
    for name, peptides, propeptides in zip(ids, all_peptides, all_propeptides):
        # peptides = utils.convert_path_to_peptide_borders(path, start_state=1, stop_state=50, offset=1)
        # propeptides = utils.convert_path_to_peptide_borders(path, start_state=51, stop_state=100, offset=1)

        out_dict['PREDICTIONS'][name] = {}
        out_dict['PREDICTIONS'][name]['peptides'] = []
        for i in range(len(peptides)):
            out_dict['PREDICTIONS'][name]['peptides'].append({'start': peptides[i][0], 'end': peptides[i][1], 'type': 'Peptide'})
        for i in range(len(propeptides)):
            out_dict['PREDICTIONS'][name]['peptides'].append({'start': propeptides[i][0], 'end': propeptides[i][1], 'type': 'Propeptide'})



    # 4. write output
    if args.output_fmt == 'img':

        marginal_dict = {}
        marginal_dict['INFO'] = out_dict['INFO'].copy()
        marginal_dict['INFO']['dimensions'] = {0: 'None', 1: 'Peptide', 2: 'Propeptide'}
        marginal_dict['PREDICTIONS'] = {}

        jobs = []
        with ProcessPoolExecutor() as executor:

            for i in range(len(all_preds)):
                pred, prob, name = all_preds[i], all_probs[i], ids[i]

                marginal_dict['PREDICTIONS'][name] = {}
                marginal_dict['PREDICTIONS'][name]['probabilities'] = prob.tolist()
                marginal_dict['PREDICTIONS'][name]['predictions'] = pred

                save_path = os.path.join(args.output_dir, f"{utils.slugify(name[1:])}.png") # skip > and replace non-alpha
                out_dict['PREDICTIONS'][name]['figure'] = save_path
                f = executor.submit(utils.plot_predictions, prob, pred, save_path)
                jobs.append(f)

        for job in jobs:
            job.result()

    print(f'Writing JSON in {args.output_dir}')
    json.dump(out_dict, open(os.path.join(args.output_dir, 'peptide_predictions.json'), 'w'), indent=1)
    if args.output_fmt == 'img':
        json.dump(marginal_dict, open(os.path.join(args.output_dir, 'sequence_outputs.json'), 'w'), indent=1)

    write_fancy_output(out_dict)


if __name__ == '__main__':
    main()
