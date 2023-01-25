import torch
import torch.nn as nn


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
                            kernel_size=filter_size, stride=1, padding=filter_size // 2) 
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
        bilstminput = out.permute(0, 2, 1)#.float()  # changing []

        # biLSTM
        if tissue_ids is not None and self.n_tissues>0:
            tissue_hot = nn.functional.one_hot(tissue_ids, self.n_tissues)  # (indices, depth)
            l_tissue = self.linear_tissue(tissue_hot.float())
            init_hidden_state = torch.stack([l_tissue for i in range(2 * self.num_lstm_layers)])  # [2,128,64]
            init_hidden_state = (init_hidden_state, init_hidden_state) # cell and hidden state.
        else:
            init_hidden_state = None

        #packed_x = nn.utils.rnn.pack_padded_sequence(bilstminput, seq_lengths.cpu().int(),
        #                                            batch_first=True, enforce_sorted=False)  # Pack the inputs ("Remove" zeropadding)
        #bi_out, states = self.biLSTM(packed_x, init_hidden_state)  # out [128,70,128]
        bi_out, states = self.biLSTM(bilstminput, init_hidden_state)
        # lstm out: last hidden state of all seq elemts which is 64, then 128 because bi. This case the only hidden state. Since we have 1 pass.
        # states is tuple of hidden states and cell states at last t in seq (2,bt,64)

        #print(bi_out)
        #bi_out, _ = nn.utils.rnn.pad_packed_sequence(bi_out, batch_first=True)  # (batch_size, seq_len, hidden_size*2), out: padded seq, lens
        h_t, c_t = states  # hidden and cell states at last time step, timestep t.


        
        # last assert condition does not hold when setting enforce_sorted=False
        # performance loss should be marginal. Never used packing before for LM experiments anyway, worked too.
        assert bi_out.size(1) == seq_lengths.max()# == seq_lengths[0], "Pad_packed error in size dim"
        packed_out = bi_out.permute(0, 2, 1)#.float()  # [128, 70, 128] -> [128, 128, 70]

        conv2_out = self.ReLU(self.conv2(packed_out)) # [128, 128, 70] -> [128, 64, 70]

        conv2_out = conv2_out.permute(0, 2, 1)

        return conv2_out


class SequenceTaggingLSTMCNN(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 32,
        filter_size: int = 3,
        hidden_size: int = 64,
        num_lstm_layers: int = 1,
        dropout_conv1: float = 0.15,
        classifier_hidden_size: int = 64,
        num_tissues: int = 0,
        num_labels: int = 1, # NOTE num_labels=1 is the binary prediction case.
        is_regression: bool = False
        ) -> None:


        super().__init__()

        self.is_regression = is_regression
        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=0.25, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=0.15, n_tissues=num_tissues)

        # if classifier has a hidden size, make a MLP. Otherwise just go to straight to label dim.
        if classifier_hidden_size>0:
            self.classifier = nn.Sequential(nn.Linear(n_filters*2, classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(classifier_hidden_size, num_labels),
                                                )
        else:
            self.classifier = nn.Linear(n_filters*2, num_labels)


    @staticmethod
    def _compute_classification_loss(logits, targets):
        
        # infer whether to use cross entropy or binary cross entropy
        if len(targets.shape)>2:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits, targets)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(dim=-1), targets.float())

        return loss
    
    @staticmethod
    def _compute_regression_loss(preds, targets):
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds.squeeze(dim=-1), targets)
        return loss



    def forward(self, embeddings, mask, targets = None, tissue_ids=None):

        features = self.feature_extractor(embeddings, mask, tissue_ids)


        logits = self.classifier(features)

        if targets is not None and not self.is_regression:
            loss = self._compute_classification_loss(logits, targets)
            return (logits, loss)
        elif targets is not None and self.is_regression:
            loss = self._compute_regression_loss(logits, targets)
            return (logits, loss)
        else:
            return logits

    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:


        from esm import pretrained
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)


        data = [
            ("protein1", sequence),
        ]
        labels, strs, toks = batch_converter(data)

        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        out = None

        toks = toks.to(device)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1).cpu()

        # set nan to zeros
        out[out!=out] = 0.0

        res = out.transpose(0,1)[1:-1] 
        seq_embedding = res[:,0]

        return seq_embedding

    def predict_from_sequence(self, sequence: str, tissue_id: int = None, return_logits: bool = False):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
            tissue_ids = torch.unsqueeze(torch.LongTensor(tissue_id),0) if tissue_id is not None else None
        

            logits = self(embedding, mask, tissue_ids= tissue_ids)

            return logits.squeeze() if return_logits else torch.sigmoid(logits).squeeze()


class LSTM(nn.Module):
    def __init__(self, input_size: int = 1280, dropout_input=0.25, hidden_size=64, num_lstm_layers=1, dropout_conv1=0.15, n_tissues=0):
        
        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.n_tissues = n_tissues

        self.ReLU = nn.ReLU()


        self.input_dropout = nn.Dropout2d(p=dropout_input)  # keep_prob=0.75

        self.biLSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        # self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=n_filters * 2, kernel_size=5,
        #                     stride=1, padding=5 // 2)  # (128,64)

        # if self.n_tissues>0:
        #     self.linear_tissue = nn.Linear(n_tissues, hidden_size)  # 4 -> 64



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
        # assert bi_out.size(1) == seq_lengths.max()# == seq_lengths[0], "Pad_packed error in size dim"
        packed_out = bi_out#.permute(0, 2, 1).float()  # [128, 70, 128] -> [128, 128, 70]

        return packed_out


class CNN(nn.Module):

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

        # self.biLSTM = nn.LSTM(input_size=n_filters, hidden_size=hidden_size, num_layers=num_lstm_layers,
        #                     bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        # self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=n_filters * 2, kernel_size=5,
        #                     stride=1, padding=5 // 2)  # (128,64)

        # if self.n_tissues>0:
        #     self.linear_tissue = nn.Linear(n_tissues, hidden_size)  # 4 -> 64



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
        # if tissue_ids is not None and self.n_tissues>0:
        #     tissue_hot = nn.functional.one_hot(tissue_ids, self.n_tissues)  # (indices, depth)
        #     l_tissue = self.linear_tissue(tissue_hot.float())
        #     init_hidden_state = torch.stack([l_tissue for i in range(2 * self.num_lstm_layers)])  # [2,128,64]
        #     init_hidden_state = (init_hidden_state, init_hidden_state) # cell and hidden state.
        # else:
        #     init_hidden_state = None

        # packed_x = nn.utils.rnn.pack_padded_sequence(bilstminput, seq_lengths.cpu().int(),
        #                                             batch_first=True, enforce_sorted=False)  # Pack the inputs ("Remove" zeropadding)
        # bi_out, states = self.biLSTM(packed_x, init_hidden_state)  # out [128,70,128]
        # lstm out: last hidden state of all seq elemts which is 64, then 128 because bi. This case the only hidden state. Since we have 1 pass.
        # states is tuple of hidden states and cell states at last t in seq (2,bt,64)


        # bi_out, _ = nn.utils.rnn.pad_packed_sequence(bi_out, batch_first=True)  # (batch_size, seq_len, hidden_size*2), out: padded seq, lens
        # h_t, c_t = states  # hidden and cell states at last time step, timestep t.


        
        # last assert condition does not hold when setting enforce_sorted=False
        # performance loss should be marginal. Never used packing before for LM experiments anyway, worked too.
        # assert bi_out.size(1) == seq_lengths.max()# == seq_lengths[0], "Pad_packed error in size dim"
        # packed_out = bi_out.permute(0, 2, 1).float()  # [128, 70, 128] -> [128, 128, 70]

        # conv2_out = self.ReLU(self.conv2(packed_out)) # [128, 128, 70] -> [128, 64, 70]

        # conv2_out = conv2_out.permute(0, 2, 1)

        return bilstminput



class SequenceTaggingLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        hidden_size: int = 64,
        classifier_hidden_size: int = 64,
        num_tissues: int = 0,
        num_labels: int = 1, # NOTE num_labels=1 is the binary prediction case.
        is_regression: bool = False
        ) -> None:


        super().__init__()

        self.is_regression = is_regression
        self.feature_extractor = LSTM(input_size=input_size, dropout_input=0.25, hidden_size=hidden_size,)#LSTMCNN(input_size=input_size, dropout_input=0.25, n_filters=32, filter_size=3, hidden_size=64, num_lstm_layers=1, dropout_conv1=0.15, n_tissues=num_tissues)

        # if classifier has a hidden size, make a MLP. Otherwise just go to straight to label dim.
        if classifier_hidden_size>0:
            self.classifier = nn.Sequential(nn.Linear(hidden_size*2, classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(classifier_hidden_size, num_labels),
                                                )
        else:
            self.classifier = nn.Linear(hidden_size*2, num_labels)


    @staticmethod
    def _compute_classification_loss(logits, targets):
        
        # infer whether to use cross entropy or binary cross entropy
        if len(targets.shape)>2:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits, targets)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(dim=-1), targets.float())

        return loss
    
    @staticmethod
    def _compute_regression_loss(preds, targets):
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds.squeeze(dim=-1), targets)
        return loss



    def forward(self, embeddings, mask, targets = None, tissue_ids=None):

        features = self.feature_extractor(embeddings, mask, tissue_ids)

        logits = self.classifier(features)

        if targets is not None and not self.is_regression:
            loss = self._compute_classification_loss(logits, targets)
            return (logits, loss)
        elif targets is not None and self.is_regression:
            loss = self._compute_regression_loss(logits, targets)
            return (logits, loss)
        else:
            return logits

    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:


        from esm import pretrained
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)


        data = [
            ("protein1", sequence),
        ]
        labels, strs, toks = batch_converter(data)

        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        out = None

        toks = toks.to(device)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1).cpu()

        # set nan to zeros
        out[out!=out] = 0.0

        res = out.transpose(0,1)[1:-1] 
        seq_embedding = res[:,0]

        return seq_embedding

    def predict_from_sequence(self, sequence: str, tissue_id: int = None, return_logits: bool = False):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
            tissue_ids = torch.unsqueeze(torch.LongTensor(tissue_id),0) if tissue_id is not None else None
        

            logits = self(embedding, mask, tissue_ids= tissue_ids)

            return logits.squeeze() if return_logits else torch.sigmoid(logits).squeeze()


class SequenceTaggingCNN(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 32,
        filter_size: int = 3,
        hidden_size: int = 64,
        dropout_conv1: float = 0.15,
        classifier_hidden_size: int = 64,
        num_tissues: int = 0,
        num_labels: int = 1, # NOTE num_labels=1 is the binary prediction case.
        is_regression: bool = False
        ) -> None:


        super().__init__()

        self.is_regression = is_regression
        self.feature_extractor = CNN(input_size=input_size, dropout_input=0.25, n_filters=n_filters, filter_size=3)#LSTMCNN(input_size=input_size, dropout_input=0.25, n_filters=32, filter_size=3, hidden_size=64, num_lstm_layers=1, dropout_conv1=0.15, n_tissues=num_tissues)

        # if classifier has a hidden size, make a MLP. Otherwise just go to straight to label dim.
        if classifier_hidden_size>0:
            self.classifier = nn.Sequential(nn.Linear(n_filters, classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(classifier_hidden_size, num_labels),
                                                )
        else:
            self.classifier = nn.Linear(n_filters, num_labels)


    @staticmethod
    def _compute_classification_loss(logits, targets):
        
        # infer whether to use cross entropy or binary cross entropy
        if len(targets.shape)>2:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits, targets)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(dim=-1), targets.float())

        return loss
    
    @staticmethod
    def _compute_regression_loss(preds, targets):
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds.squeeze(dim=-1), targets)
        return loss



    def forward(self, embeddings, mask, targets = None, tissue_ids=None):

        features = self.feature_extractor(embeddings, mask, tissue_ids)


        logits = self.classifier(features)

        if targets is not None and not self.is_regression:
            loss = self._compute_classification_loss(logits, targets)
            return (logits, loss)
        elif targets is not None and self.is_regression:
            loss = self._compute_regression_loss(logits, targets)
            return (logits, loss)
        else:
            return logits


class SequenceTaggingLSTMCNNCRF(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 32,
        filter_size: int = 3,
        hidden_size: int = 64,
        num_lstm_layers: int = 1,
        dropout_conv1: float = 0.15,
        classifier_hidden_size: int = 64,
        num_tissues: int = 0,
        num_labels: int = 1, # NOTE num_labels=1 is the binary prediction case.
        is_regression: bool = False
        ) -> None:


        super().__init__()

        self.is_regression = is_regression
        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=num_tissues)

        # if classifier has a hidden size, make a MLP. Otherwise just go to straight to label dim.
        if classifier_hidden_size>0:
            self.classifier = nn.Sequential(nn.Linear(n_filters*2, classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(classifier_hidden_size, num_labels),
                                                )
        else:
            self.classifier = nn.Linear(n_filters*2, num_labels)

        from .multi_tag_crf import CRF
        self.crf = CRF(num_labels, batch_first=True)




    def forward(self, embeddings, mask, targets = None, tissue_ids=None):

        features = self.feature_extractor(embeddings, mask, tissue_ids)


        logits = self.classifier(features)

        viterbi_paths = self.crf.decode(emissions=logits, mask = mask.byte())
        #pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        pos_preds = torch.tensor(pos_preds, device = logits.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(logits, mask.byte())
        logit_probs = torch.log(probs/(1-probs))

        if targets is not None and not self.is_regression:
            loss = self.crf(emissions = logits, tags=targets.long(), mask = mask.byte(), reduction='mean') *-1
            return (logits[:,:,1], pos_preds, loss)
        elif targets is not None and self.is_regression:
            raise NotImplementedError
        else:
            return logit_probs[:,:,1], pos_preds

    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:


        from esm import pretrained
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)


        data = [
            ("protein1", sequence),
        ]
        labels, strs, toks = batch_converter(data)

        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        out = None

        toks = toks.to(device)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1).cpu()

        # set nan to zeros
        out[out!=out] = 0.0

        res = out.transpose(0,1)[1:-1] 
        seq_embedding = res[:,0]

        return seq_embedding

    def predict_from_sequence(self, sequence: str, tissue_id: int = None, return_logits: bool = False):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
            tissue_ids = torch.unsqueeze(torch.LongTensor(tissue_id),0) if tissue_id is not None else None
        

            logits, pos_preds = self(embedding, mask, tissue_ids= tissue_ids)

            return logits.squeeze() if return_logits else torch.sigmoid(logits).squeeze()