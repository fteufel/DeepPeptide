from bdb import set_trace
from typing import Any, Sequence, Tuple, List

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd
from hashlib import md5
import numpy as np


def make_hashes(names: List[str]) -> List[str]:
    hashes = []
    for name in names:
        hashes.append(md5(name.encode()).digest().hex())

    return hashes

from typing import Iterable, Callable, Any, List
def filter_lists(filter_condition: Callable[[Any], bool], filter_key_iterable: Iterable[Any] , *additional_iterables: Iterable[Any]) -> List[List[Any]]:
    """Generic function for filtering a number of equal-length lists based on filter_condition.
    If an element of filter_key_iterable is in allowed_values, the list entries are retained, else dropped.

    Args:
        filter_condition (Callable[[Any], bool]): Will be executed on values in `filter_key_iterable`. Retained if returns True.
        filter_key_iterable (Iterable[Any]): A Iterable of values to be filtered.
        *additional_iterables (Iterable[Any], optional): More iterables of the same length that should be filtered.

    Returns:
        List[List[Any]]: The filtered input lists. [filter_key_iterable, *additional_iterables]
    """
    out_list = []

    for values in zip(filter_key_iterable, *additional_iterables):
        if filter_condition(values[0]):
            out_list.append(values)
    #if filtering condition leaves 0 elements, reconstructing the correct number of lists via zip will fail.
    if len(out_list) == 0:
        return [list() for x in range(len(additional_iterables)+1)]
    else:
        return [list(x) for x in zip(*out_list)]


    


class PrecomputedCSVDataset(Dataset):
    '''Use together with modified extract.py script. Retrieves seqs via md5 hash.'''
    def __init__(
        self, 
        embeddings_dir: str, 
        data_file: str, 
        partitioning_file: str, 
        partitions: List[int]=[0], 
        label_type: str = 'binary',
        ):
        """
        Dataset to hold fasta files with precomputed embeddings.
        Can also parse graph-part partition assignments.


        Args:
            embeddings_dir (str): Directory containing precomputed embeddings produced by `extract.py`
            csv_file (str): csv with sequences, labels and other metadata.
            partitioning_file (str): Graph-Part output for `fasta_file`. Defaults to None.
            partitions (List[int], optional): Partitions to retain. Defaults to [0].
        """

        super().__init__()
        self.embeddings_dir = embeddings_dir


        data = pd.read_csv(data_file, index_col='protein_id')
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(partitions)]
        self.data = data

        self.names = data.index.tolist() # don't want to bother with pandas indexing here.

        self.label_type = label_type

        if label_type == 'binary':
            self.labels = data['is_peptide'].tolist()
        elif label_type == 'start_stop':
            self.labels = data['start_stop'].tolist()
        elif label_type == 'cleavage_sites':
            # simplify the start-stop labels. treat all sites as simple cleavage sites.
            labels = data['start_stop'].str.replace('2','1').str.replace('3','1').tolist() # 1 start, 2 end, 3 both
            self.labels = ['0'+x[1:-1] +'0' for x in labels] # start and end positions fo protein are never CS, also when there is a peptide.
        elif label_type == 'intensity':
            labels =  data['intensity'].str.split(';')
            self.labels = labels.apply(lambda x: [float(y) for y in x]).tolist()
        
        else:
            raise NotImplementedError(label_type)

        self.sequences = data['sequence'].tolist()
        self.organism = data['organism'].tolist()
        self.hashes = make_hashes(self.sequences)
        
        if 'tissue' in data.columns:
            tissues = data['tissue'].astype('category')
            self.tissues = tissues.cat.codes.tolist()
            self.tissue_names = tissues.cat.categories.tolist()
        else:
            self.tissues = np.ones(len(self.organism))
            self.tissue_names = ['unknown']


    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

        seq_hash = self.hashes[index]
        #print(seq_hash, self.names[index], self.sequences[index])
        try:
            embeddings = torch.load(os.path.join(self.embeddings_dir, f'{seq_hash}.pt'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find sequence hash {seq_hash} for {self.names[index]} in {self.embeddings_dir}.')
        label = self.labels[index]

        if self.label_type == 'intensity':
            label = np.array(label)
            np.log(label, where=label>0, out=label) # applies in-place to all >0.
            label = torch.FloatTensor(label)
            # if label.isnan().sum()>0:
            #     import ipdb; ipdb.set_trace()

        else:
            label = torch.IntTensor([int(x) for x in label])

        # mask : batch_size, seq_len
        mask = torch.ones(embeddings.shape[0])

        tissue = self.tissues[index]

        return embeddings, mask, label, tissue

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        embeddings, masks, labels, tissues = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        tissues = torch.LongTensor(tissues)


        return embeddings.permute(0,2,1), masks, labels, tissues


BLOSUM_normalized = np.array([[ 0.2901, 0.0310, 0.0256, 0.0297, 0.0216, 0.0256, 0.0405, 0.0783, 0.0148, 0.0432, 0.0594, 0.0445, 0.0175, 0.0216, 0.0297, 0.0850, 0.0499, 0.0054, 0.0175, 0.0688],
                              [ 0.0446, 0.3450, 0.0388, 0.0310, 0.0078, 0.0484, 0.0523, 0.0329, 0.0233, 0.0233, 0.0465, 0.1202, 0.0155, 0.0174, 0.0194, 0.0446, 0.0349, 0.0058, 0.0174, 0.0310],
                              [ 0.0427, 0.0449, 0.3169, 0.0831, 0.0090, 0.0337, 0.0494, 0.0652, 0.0315, 0.0225, 0.0315, 0.0539, 0.0112, 0.0180, 0.0202, 0.0697, 0.0494, 0.0045, 0.0157, 0.0270],
                              [ 0.0410, 0.0299, 0.0690, 0.3974, 0.0075, 0.0299, 0.0914, 0.0466, 0.0187, 0.0224, 0.0280, 0.0448, 0.0093, 0.0149, 0.0224, 0.0522, 0.0354, 0.0037, 0.0112, 0.0243],
                              [ 0.0650, 0.0163, 0.0163, 0.0163, 0.4837, 0.0122, 0.0163, 0.0325, 0.0081, 0.0447, 0.0650, 0.0203, 0.0163, 0.0203, 0.0163, 0.0407, 0.0366, 0.0041, 0.0122, 0.0569],
                              [ 0.0559, 0.0735, 0.0441, 0.0471, 0.0088, 0.2147, 0.1029, 0.0412, 0.0294, 0.0265, 0.0471, 0.0912, 0.0206, 0.0147, 0.0235, 0.0559, 0.0412, 0.0059, 0.0206, 0.0353],
                              [ 0.0552, 0.0497, 0.0405, 0.0902, 0.0074, 0.0645, 0.2965, 0.0350, 0.0258, 0.0221, 0.0368, 0.0755, 0.0129, 0.0166, 0.0258, 0.0552, 0.0368, 0.0055, 0.0166, 0.0313],
                              [ 0.0783, 0.0229, 0.0391, 0.0337, 0.0108, 0.0189, 0.0256, 0.5101, 0.0135, 0.0189, 0.0283, 0.0337, 0.0094, 0.0162, 0.0189, 0.0513, 0.0297, 0.0054, 0.0108, 0.0243],
                              [ 0.0420, 0.0458, 0.0534, 0.0382, 0.0076, 0.0382, 0.0534, 0.0382, 0.3550, 0.0229, 0.0382, 0.0458, 0.0153, 0.0305, 0.0191, 0.0420, 0.0267, 0.0076, 0.0573, 0.0229],
                              [ 0.0471, 0.0177, 0.0147, 0.0177, 0.0162, 0.0133, 0.0177, 0.0206, 0.0088, 0.2710, 0.1679, 0.0236, 0.0368, 0.0442, 0.0147, 0.0250, 0.0398, 0.0059, 0.0206, 0.1767],
                              [ 0.0445, 0.0243, 0.0142, 0.0152, 0.0162, 0.0162, 0.0202, 0.0213, 0.0101, 0.1154, 0.3755, 0.0253, 0.0496, 0.0547, 0.0142, 0.0243, 0.0334, 0.0071, 0.0223, 0.0962],
                              [ 0.0570, 0.1071, 0.0415, 0.0415, 0.0086, 0.0535, 0.0708, 0.0432, 0.0207, 0.0276, 0.0432, 0.2781, 0.0155, 0.0155, 0.0276, 0.0535, 0.0397, 0.0052, 0.0173, 0.0328],
                              [ 0.0522, 0.0321, 0.0201, 0.0201, 0.0161, 0.0281, 0.0281, 0.0281, 0.0161, 0.1004, 0.1968, 0.0361, 0.1606, 0.0482, 0.0161, 0.0361, 0.0402, 0.0080, 0.0241, 0.0924],
                              [ 0.0338, 0.0190, 0.0169, 0.0169, 0.0106, 0.0106, 0.0190, 0.0254, 0.0169, 0.0634, 0.1142, 0.0190, 0.0254, 0.3869, 0.0106, 0.0254, 0.0254, 0.0169, 0.0888, 0.0550],
                              [ 0.0568, 0.0258, 0.0233, 0.0310, 0.0103, 0.0207, 0.0362, 0.0362, 0.0129, 0.0258, 0.0362, 0.0413, 0.0103, 0.0129, 0.4935, 0.0439, 0.0362, 0.0026, 0.0129, 0.0310],
                              [ 0.1099, 0.0401, 0.0541, 0.0489, 0.0175, 0.0332, 0.0524, 0.0663, 0.0192, 0.0297, 0.0419, 0.0541, 0.0157, 0.0209, 0.0297, 0.2199, 0.0820, 0.0052, 0.0175, 0.0419],
                              [ 0.0730, 0.0355, 0.0434, 0.0375, 0.0178, 0.0276, 0.0394, 0.0434, 0.0138, 0.0533, 0.0651, 0.0454, 0.0197, 0.0237, 0.0276, 0.0927, 0.2465, 0.0059, 0.0178, 0.0710],
                              [ 0.0303, 0.0227, 0.0152, 0.0152, 0.0076, 0.0152, 0.0227, 0.0303, 0.0152, 0.0303, 0.0530, 0.0227, 0.0152, 0.0606, 0.0076, 0.0227, 0.0227, 0.4924, 0.0682, 0.0303],
                              [ 0.0405, 0.0280, 0.0218, 0.0187, 0.0093, 0.0218, 0.0280, 0.0249, 0.0467, 0.0436, 0.0685, 0.0312, 0.0187, 0.1308, 0.0156, 0.0312, 0.0280, 0.0280, 0.3178, 0.0467],
                              [ 0.0700, 0.0219, 0.0165, 0.0178, 0.0192, 0.0165, 0.0233, 0.0247, 0.0082, 0.1646, 0.1303, 0.0261, 0.0316, 0.0357, 0.0165, 0.0329, 0.0494, 0.0055, 0.0206, 0.2689]])


AA_VOCAB = {'[PAD]':0,
                  'A':1,  'R':2,  'N':3,  'D':4,  'C':5,  'Q':6,  'E':7,  'G':8,  'H':9,  'I':10, 
                  'L':11, 'K':12 ,'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20,
                    'U':0, 'X':0, 'Z':0, 'B':0}




class AminoAcidTokenizer():
    def __init__(self) -> None:
        self.vocabulary = AA_VOCAB
        self.pad_token = '[PAD]'
        self.pad_id = self.vocabulary[self.pad_token]
        self.vocab_size = len(self.vocabulary)

    def encode(self, string: str) -> List[int]:
        return [self.vocabulary[x] for x in string]



class BLOSUMCSVDataset(Dataset):
    '''Encode sequences via BLOSUM.'''
    def __init__(
        self, 
        data_file: str, 
        partitioning_file: str, 
        partitions: List[int]=[0], 
        label_type: str = 'binary',
        ):
        """
        Dataset to hold fasta files with precomputed embeddings.
        Can also parse graph-part partition assignments.


        Args:
            embeddings_dir (str): Directory containing precomputed embeddings produced by `extract.py`
            csv_file (str): csv with sequences, labels and other metadata.
            partitioning_file (str): Graph-Part output for `fasta_file`. Defaults to None.
            partitions (List[int], optional): Partitions to retain. Defaults to [0].
        """

        super().__init__()


        data = pd.read_csv(data_file, index_col='protein_id')
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(partitions)]

        self.names = data.index.tolist() # don't want to bother with pandas indexing here.

        if label_type == 'binary':
            self.labels = data['is_peptide'].tolist()
        elif label_type == 'start_stop':
            self.labels = data['start_stop'].tolist()
        elif label_type == 'cleavage_sites':
            # simplify the start-stop labels. treat all sites as simple cleavage sites.
            labels = data['start_stop'].str.replace('2','1').str.replace('3','1').tolist() # 1 start, 2 end, 3 both
            self.labels = ['0'+x[1:-1] +'0' for x in labels] # start and end positions fo protein are never CS, also when there is a peptide.
        
        else:
            raise NotImplementedError(label_type)

        self.sequences = data['sequence'].tolist()
        self.organism = data['organism'].tolist()
        self.hashes = make_hashes(self.sequences)
        
        if 'tissue' in data.columns:
            tissues = data['tissue'].astype('category')
            self.tissues = tissues.cat.codes.tolist()
            self.tissue_names = tissues.cat.categories.tolist()
        else:
            self.tissues = np.ones(len(self.organism))
            self.tissue_names = ['unknown']

        self.tokenizer = AminoAcidTokenizer()
        embed_weights = np.concatenate([np.zeros((1,20)), BLOSUM_normalized])
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(embed_weights), freeze=True)


    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:


        tokens = torch.LongTensor(self.tokenizer.encode(self.sequences[index]))
        with torch.no_grad():
            embeddings = self.embedding_layer(tokens).float()

        label = self.labels[index]
        label = torch.IntTensor([int(x) for x in label])

        # mask : batch_size, seq_len
        mask = torch.ones(embeddings.shape[0])

        tissue = self.tissues[index]

        return embeddings, mask, label, tissue

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        embeddings, masks, labels, tissues = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        tissues = torch.LongTensor(tissues)

        return embeddings.permute(0,2,1), masks, labels, tissues


from .crf_label_utils import peptide_list_to_label_sequence, parse_coordinate_string, peptide_list_to_binary_label_sequence
class PrecomputedCSVForCRFDataset(Dataset):
    '''Use together with modified extract.py script. Retrieves seqs via md5 hash.'''
    def __init__(
        self, 
        embeddings_dir: str, 
        data_file: str, 
        partitioning_file: str, 
        partitions: List[int]=[0], 
        label_type: str = 'simple',
        ):
        """
        Dataset to hold fasta files with precomputed embeddings.
        Can also parse graph-part partition assignments.


        Args:
            embeddings_dir (str): Directory containing precomputed embeddings produced by `extract.py`
            csv_file (str): csv with sequences, labels and other metadata.
            partitioning_file (str): Graph-Part output for `fasta_file`. Defaults to None.
            partitions (List[int], optional): Partitions to retain. Defaults to [0].
        """

        super().__init__()
        self.embeddings_dir = embeddings_dir

        data = pd.read_csv(data_file, index_col='protein_id')
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(partitions)]
        data = data.fillna('') # empty coordinates would become nan.
        self.data = data

        self.names = data.index.tolist() # don't want to bother with pandas indexing here.

        self.label_type = label_type

        # NOTE self.peptides is 1-based indexing straight from UniProt.

        if label_type == 'simple': # The easiest case 0 for no peptide, 1 for peptide.
            coordinate_strings = data['coordinates'].tolist()
            coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in coordinate_strings]
            sequences = data['sequence'].tolist()
            self.labels = [peptide_list_to_binary_label_sequence(peptides, len(seq)) for peptides, seq in zip(coordinates, sequences)]
            self.peptides = coordinates

        elif label_type == 'simple_with_propeptides':
            coordinate_strings = data['coordinates'].tolist()
            propeptide_coordinate_strings = data['propeptide_coordinates'].tolist()
            coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in coordinate_strings]
            propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in propeptide_coordinate_strings]
            sequences = data['sequence'].tolist()

            labels = [peptide_list_to_binary_label_sequence(peptides, len(seq)) for peptides, seq in zip(coordinates, sequences)]
            propeptide_labels = [peptide_list_to_binary_label_sequence(peptides, len(seq), label_value=2) for peptides, seq in zip(propeptide_coordinates, sequences)]

            # labels are 0-1, propeptide_labels are 0-2. There are no overlaps between their positions.
            self.labels = [x+y for x,y in zip(labels, propeptide_labels)]
            
            self.peptides_only = coordinates
            self.propeptides = propeptide_coordinates
            self.peptides = [(x,y) for x,y, in zip(coordinates, propeptide_coordinates)] # data loading works exactly the same. only metrics computation needs to unpack this.


        elif label_type == 'multistate': # The advanced peptide state grammar.

            # TODO decide how to handle overlap merges that cause peptides longer than max.
            # As it only affects a few we just drop them for now to avoid errors.
            data = data.loc[~data.index.isin(['P87352', 'Q91082', 'P10645'])]
            self.data = data
            self.names = data.index.tolist()

            coordinate_strings = data['coordinates'].tolist()
            coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in coordinate_strings]
            sequences = data['sequence'].tolist()
            self.labels = [peptide_list_to_label_sequence(peptides, len(seq)) for peptides, seq in zip(coordinates, sequences)]
            self.peptides = coordinates

        elif label_type == 'multistate_with_propeptides': # The advanced peptide state grammar.

            # TODO decide how to handle overlap merges that cause peptides longer than max.
            # As it only affects a few we just drop them for now to avoid errors.
            data = data.loc[~data.index.isin(['P87352', 'Q91082', 'P10645'])]
            self.data = data
            self.names = data.index.tolist()

            coordinate_strings = data['coordinates'].tolist()
            propeptide_coordinate_strings = data['propeptide_coordinates'].tolist()
            coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in coordinate_strings]
            propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=True) for x in propeptide_coordinate_strings]
            sequences = data['sequence'].tolist()

            labels = [peptide_list_to_label_sequence(peptides, len(seq)) for peptides, seq in zip(coordinates, sequences)]
            propeptide_labels = [peptide_list_to_label_sequence(peptides, len(seq), start_state=61) for peptides, seq in zip(propeptide_coordinates, sequences)]
            #self.labels = [peptide_list_to_label_sequence(peptides, len(seq)) for peptides, seq in zip(coordinates, sequences)]
            self.labels = [x+y for x,y in zip(labels, propeptide_labels)]
            
            self.peptides_only = coordinates
            self.propeptides = propeptide_coordinates
            self.peptides = [(x,y) for x,y, in zip(coordinates, propeptide_coordinates)] # data loading works exactly the same. only metrics computation needs to unpack this.



        elif label_type == 'multilabel':
            coordinate_strings = data['coordinates'].tolist()
            coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in coordinate_strings]
            sequences = data['sequence'].tolist()
            
            self.labels = None
            self.peptides = coordinates
            raise NotImplementedError('multilabel')
        else:
            raise NotImplementedError(label_type)

        self.sequences = data['sequence'].tolist()
        self.organism = data['organism'].tolist()
        self.hashes = make_hashes(self.sequences)
        

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

        seq_hash = self.hashes[index]
        try:
            embeddings = torch.load(os.path.join(self.embeddings_dir, f'{seq_hash}.pt'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find sequence hash {seq_hash} for {self.names[index]} in {self.embeddings_dir}.')
        
        label = torch.from_numpy(self.labels[index])
        mask = torch.ones(embeddings.shape[0])
        peptides = self.peptides[index]

        return embeddings, mask, label, peptides

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray]]:

        embeddings, masks, labels, peptides = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

        # TODO ensure this handles multitag matrices
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return embeddings.permute(0,2,1), masks, labels, peptides



class PrecomputedCSVForOverlapCRFDataset(Dataset):
    '''Use together with modified extract.py script. Retrieves seqs via md5 hash.'''
    def __init__(
        self, 
        embeddings_dir: str, 
        data_file: str, 
        partitioning_file: str, 
        partitions: List[int]=[0], 
        label_type = None, # for compatibility
        ):
        """
        Dataset to hold fasta files with precomputed embeddings.
        Can also parse graph-part partition assignments.
        When dealing with overlapping peptides, randomly samples one and uses it for labeling in a iteration.
        This is much slower than just loading a precomputed label, so use enough workers.


        Args:
            embeddings_dir (str): Directory containing precomputed embeddings produced by `extract.py`
            csv_file (str): csv with sequences, labels and other metadata.
            partitioning_file (str): Graph-Part output for `fasta_file`. Defaults to None.
            partitions (List[int], optional): Partitions to retain. Defaults to [0].
        """

        super().__init__()
        self.embeddings_dir = embeddings_dir

        data = pd.read_csv(data_file, index_col='protein_id')
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(partitions)]
        data = data.fillna('') # empty coordinates would become nan.
        self.data = data

        self.names = data.index.tolist() # don't want to bother with pandas indexing here.

        
        # NOTE self.peptides is 1-based indexing straight from UniProt.
        self.data = data
        self.names = data.index.tolist()

        coordinate_strings = data['coordinates'].tolist()
        propeptide_coordinate_strings = data['propeptide_coordinates'].tolist()
        coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in coordinate_strings]
        propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in propeptide_coordinate_strings]

        # NOTE we feed .data to our metrics fn. it expects some more columns.
        self.data['true_peptides'] = coordinates
        self.data['true_propeptides'] = propeptide_coordinates

        self.peptides_only = coordinates
        self.propeptides = propeptide_coordinates
        self.peptides = [(x,y) for x,y, in zip(coordinates, propeptide_coordinates)] # data loading works exactly the same. only metrics computation needs to unpack this.


        self.sequences = data['sequence'].tolist()
        self.organism = data['organism'].tolist()
        self.hashes = make_hashes(self.sequences)
        

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod 
    def _sample_from_overlapping_peptides(peptide_coordinates, propeptide_coordinates):
        '''Finds overlapping groups of peptides. Samples one peptide from each group.'''
        peptides_to_keep = []
        propeptides_to_keep = []

        # https://stackoverflow.com/questions/48243507/group-rows-by-overlapping-ranges
        all_peptides = peptide_coordinates + propeptide_coordinates
        types = ['Peptide'] * len(peptide_coordinates) + ['Propeptide'] * len(propeptide_coordinates)
        starts, ends = zip(*all_peptides)

        df = pd.DataFrame([starts, ends, types], index = ['start', 'end', 'type']).T
        df = df.sort_values(['start', 'end'], ascending=[True, False])

        df['group'] = (df['end'].cummax().shift() < df['start']).cumsum() # was <= originally. but indexing is inclusive in uniprot.
        df = df.sort_index()

        for group, group_df in df.groupby('group'):
            peptide = group_df.sample(1).iloc[0] # returns df, but we need the single row
            if peptide['type'] == 'Peptide':
                peptides_to_keep.append((peptide['start'], peptide['end']))
            else:
                propeptides_to_keep.append((peptide['start'], peptide['end']))

        return peptides_to_keep, propeptides_to_keep



    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

        seq_hash = self.hashes[index]
        seq_len = len(self.sequences[index])
        try:
            embeddings = torch.load(os.path.join(self.embeddings_dir, f'{seq_hash}.pt')).to(torch.float32) #esm2 comes as half
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find sequence hash {seq_hash} for {self.names[index]} in {self.embeddings_dir}.')

        peptides, propeptides = self.peptides[index]
        peptides, propeptides = self._sample_from_overlapping_peptides(peptides, propeptides)

        label = peptide_list_to_label_sequence(peptides, seq_len, max_len = 50) 
        propeptide_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=51, max_len=50) 
        label = label + propeptide_label # numpy arrays with no overlap at nonzero positions so we can just add the two.


        label = torch.from_numpy(label)
        mask = torch.ones(embeddings.shape[0])
        peptides = self.peptides[index]

        return embeddings, mask, label, self.peptides[index] # this is for the metrics.

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray]]:

        embeddings, masks, labels, peptides = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return embeddings.permute(0,2,1), masks, labels, peptides