'''
This file contains all model components.
Pasted from  
crf_models.py
lstm_cnn.py
'''

import torch
import torch.nn as nn
from crf import CRF


class LSTMCNN(nn.Module):

    def __init__(self, input_size: int = 1280, dropout_input=0.25, n_filters=32, filter_size=3, hidden_size=64, num_lstm_layers=1, dropout_conv1=0.15, n_tissues=0):
        '''
        bidirectional LSTM - CNN model to process sequence data. returns output of same length as the input.
        
        (batch_size, seq_len, hidden_size*2)
        '''
        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.n_tissues = n_tissues

        self.ReLU = nn.ReLU()


        self.input_dropout = nn.Dropout2d(p=dropout_input)  # keep_prob=0.75
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=n_filters,
                            kernel_size=filter_size, stride=1, padding=filter_size // 2)  # in:20, out=32
        self.conv1_dropout = nn.Dropout2d(p=dropout_conv1)  # keep_prob=0.85  # could this dropout be added directly to LSTM

        self.biLSTM = nn.LSTM(input_size=n_filters, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=n_filters * 2, kernel_size=5,
                            stride=1, padding=5 // 2)  # (128,64)

        if self.n_tissues>0:
            self.linear_tissue = nn.Linear(n_tissues, hidden_size)  # 4 -> 64



    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, tissue_ids: torch.LongTensor = None):
        '''
        embeddings: torch.Tensor (batch_size, embeddings_dim, seq_len)
        mask: torch.IntTensor (batch_size, seq_len)
        tissue_ids: torch.IntTensor (batch_size)

        Returns:
        hidden_states: torch.Tensor (batch_size, seq_len, n_filters * 2)
        '''
        out = embeddings # [batch_size, embeddings_dim, sequence_length]
        seq_lengths = mask.sum(dim=1)
        out = self.input_dropout(out)  # 2D feature map dropout

        out = self.ReLU(self.conv1(out))  # [batch_size,embeddings_dim,sequence_length] -> [batch_size,32,sequence_length]

        out = self.conv1_dropout(out)
        bilstminput = out.permute(0, 2, 1).float()  # changing []

        # biLSTM
        if tissue_ids is not None and self.n_tissues>0:
            tissue_hot = nn.functional.one_hot(tissue_ids, self.n_tissues)  # (indices, depth)
            l_tissue = self.linear_tissue(tissue_hot.float())
            init_hidden_state = torch.stack([l_tissue for i in range(2 * self.num_lstm_layers)])  # [2,128,64]
            init_hidden_state = (init_hidden_state, init_hidden_state) # cell and hidden state.
        else:
            init_hidden_state = None

        packed_x = nn.utils.rnn.pack_padded_sequence(bilstminput, seq_lengths.cpu().int(),
                                                    batch_first=True, enforce_sorted=False)  # Pack the inputs ("Remove" zeropadding)
        bi_out, states = self.biLSTM(packed_x, init_hidden_state)  # out [128,70,128]
        # lstm out: last hidden state of all seq elemts which is 64, then 128 because bi. This case the only hidden state. Since we have 1 pass.
        # states is tuple of hidden states and cell states at last t in seq (2,bt,64)


        bi_out, _ = nn.utils.rnn.pad_packed_sequence(bi_out, batch_first=True)  # (batch_size, seq_len, hidden_size*2), out: padded seq, lens
        h_t, c_t = states  # hidden and cell states at last time step, timestep t.


        
        # last assert condition does not hold when setting enforce_sorted=False
        # performance loss should be marginal. Never used packing before for LM experiments anyway, worked too.
        assert bi_out.size(1) == seq_lengths.max()# == seq_lengths[0], "Pad_packed error in size dim"
        packed_out = bi_out.permute(0, 2, 1).float()  # [128, 70, 128] -> [128, 128, 70]

        conv2_out = self.ReLU(self.conv2(packed_out)) # [128, 128, 70] -> [128, 64, 70]

        conv2_out = conv2_out.permute(0, 2, 1)

        return conv2_out

class CRFBaseModel(nn.Module):
    '''Extend this model by defining a feature_extractor.'''
    def __init__(
        self,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__()
        self.max_len = 50
        self.min_len = 5
        self.feature_extractor = None
        self.features_to_emissions = nn.Linear(64, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

    @staticmethod
    def get_crf_constraints(max_len: int = 60, min_len: int = 5, n_branches: int = 1):
        '''Build the peptide state space model.
        Each peptide starts as state 1 and goes through 2, 3.
        From 3, it can either go to 4 or skip ahead to any other state up to 59.
        From 59, go to 60. 
        This enforces a minimum peptide length of 5. Each peptide is forced to end in 60,
        so this state can learn peptide end properties.
        '''
        allowed_starts = [0,1]
        allowed_ends = [0, max_len]

        allowed_state_transitions = []
        allowed_state_transitions.append((0,0)) # None to None
        allowed_state_transitions.append((0,1)) # None to Peptide_0
        allowed_state_transitions.append((max_len,1)) # Peptide_50 to Peptide_0, no need to have 1 AA gap
        allowed_state_transitions.append((max_len,0)) # Peptide_50 (peptide end position) to None

        for i in range(1, max_len): 
            to_next = (i, i+1)
            allowed_state_transitions.append(to_next)

            if i >min_len-1: #make skip forward connections
                skip_to_i = (min_len-2,i) #3
                allowed_state_transitions.append(skip_to_i) 

        allowed_state_transitions.append((max_len-1,max_len)) # peptide end position -1 to peptide end position
        # logic of this state space model is that the end state is the same for all peptides, regardless their length.

        # add a self loop on the pre-last state. Should help avoid issues at inference when longer stuff might show up.
        allowed_state_transitions.append((max_len-1, max_len-1))

        # branch 1 + no state: 0-50
        # branch 2: 51-101
        if n_branches == 2:
            start = 1 + max_len
            end = 2*max_len

            allowed_starts.append(start)
            allowed_ends.append(end)
            allowed_state_transitions.append((0,start))
            allowed_state_transitions.append((end,start)) 
            allowed_state_transitions.append((end,0))

            # can go directly from end of peptide to start of propeptide and vice versa.
            allowed_state_transitions.append((end,1))
            allowed_state_transitions.append((start-1, start))

            for i in range(start, end): 
                to_next = (i, i+1)
                allowed_state_transitions.append(to_next)

                if i >min_len-1: #make skip forward connections
                    skip_to_i = (start+min_len-3, i)#((min_len-2,i))
                    allowed_state_transitions.append(skip_to_i) 

        # add a self loop on the pre-last state. Should help avoid issues at inference when longer stuff might show up.
        allowed_state_transitions.append((end-1, end-1))

        return allowed_state_transitions, allowed_starts, allowed_ends

    def _debug_crf(self, targets):
        '''Check label sequences for incompatibilities with the defined state grammar.'''
        for i in range(targets.shape[0]):

            for j in range(1, targets.shape[1]):
                l = int(targets[i,j].item())
                l_prev = int(targets[i,j-1].item())

                if (l_prev, l) not in self.allowed_transitions:
                    print(f'Found invalid transition from {l_prev} to {l}.')


    
    def _repeat_emissions(self, emissions):
        '''Turn a (batch_size, seq_len, 2) tensor into (batch_size, seq_len, num_states) by repeating the emissions at position 1.'''

        if emissions.shape[-1] == 2:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)    
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:(self.max_len+1)] = emissions[:,:,1].unsqueeze(-1)
        elif emissions.shape[-1] == 3:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:] = emissions[:,:,1].unsqueeze(-1)
            emissions_out[:,:, (self.max_len+1):] = emissions[:,:,2].unsqueeze(-1)
        else:
            raise NotImplementedError()
        
        return emissions_out


    def forward(self, embeddings, mask, targets = None, skip_marginals: bool = False, top_k: int = 1):

        features = self.feature_extractor(embeddings, mask) # (batch_size, seq_len, feature_dim)
        emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
        emissions = self._repeat_emissions(emissions) # (batch_size, seq_len, num_states)
        
        # viterbi_paths = self.crf.decode(emissions=emissions, mask = mask.byte())

        viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask = mask.byte(), top_k=top_k)

        #pad the viterbi paths
        # max_pad_len = max([len(x) for x in viterbi_paths])
        # pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        # pos_preds = torch.tensor(pos_preds, device = emissions.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(emissions, mask.byte()) if not skip_marginals else torch.softmax(emissions, dim=-1)

        if targets is not None:
            loss = self.crf(emissions = emissions, tags=targets.long(), mask = mask.byte(), reduction='mean') *-1

            if loss.item()>10000:
                self._debug_crf(targets)
            return (probs, viterbi_paths, loss)
        else:
            return probs, viterbi_paths, path_probs


class LSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__(num_labels, num_states)

        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)