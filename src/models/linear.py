import torch
import torch.nn as nn

class SequenceTaggingLinear(nn.Module):
    def __init__(
        self,
        dropout_input: float = 0.25,
        classifier_hidden_size: int = 64,
        num_labels: int = 1 # NOTE num_labels=1 is the binary prediction case.
        ) -> None:


        super().__init__()

        self.input_dropout = nn.Dropout2d(p=dropout_input)

        # if classifier has a hidden size, make a MLP. Otherwise just go to straight to label dim.
        if classifier_hidden_size>0:
            self.classifier = nn.Sequential(nn.Linear(1280, classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(classifier_hidden_size, num_labels),
                                                )
        else:
            self.classifier = nn.Linear(1280, num_labels)


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



    def forward(self, embeddings, mask, targets = None, tissue_ids=None):

        features = self.input_dropout(embeddings.permute(0, 2, 1))

        logits = self.classifier(features)

        if targets is not None:
            loss = self._compute_classification_loss(logits, targets)
            return (logits, loss)
        else:
            return logits