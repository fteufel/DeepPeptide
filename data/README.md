# Data

[`labeled_sequences.csv`](labeled_sequences.csv) and [`protein_sequences.fasta`](protein_sequences.fasta) contain all annotated proteins that were extracted from UniProt. As we trained and evaluated DeepPeptide in nested cross-validation, this data was homology partitioned into five folds using [GraphPart](https://github.com/graph-part/graph-part). This results in the removal of some sequences from the data in order to ensure that there is no homology between the folds.

The folds are indicated in [`graphpart_assignments.csv`](graphpart_assignments.csv). Column `cluster` indicates the fold. All sequences that are not listed in this file are not part of the 7,623 proteins that were used for training and evaluation.