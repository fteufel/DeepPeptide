# DeepPeptide

This is the DeepPeptide predictor. It is an ensemble of the 20 models trained in nested cross-validation.

## Usage

1. Install the requirements in `requirements.txt` using `pip install -r requirements.txt`.
2. Ensure you are in the `predictor` directory. (This is important so the model paths are correct.)
3. Run the predictor using `python3 predict.py -ff fasta_file.fasta -od testrun_outputs/`.


| Argument | Short | Default | Description |
|---------|-------|-------|-------|
| `--fasta_file` | `-ff` | `None` (required) | Path to the fasta file containing the protein sequences to predict. |
| `--output_dir` | `-od` | `None` (required) | Path to the directory where the output files will be saved. |
| `--batch_size` | `-bs` | `10` | Batch size for prediction. Use this to tune memory usage according to your hardware. In general, larger batches are better, but a batch needs to fit the memory constraints of the given hardware. |
| `--output_fmt` | `-of` | `img` | Output format. Can be `img`, which produces a plot for each sequence, or `json` which skips plot generation. |
| `--esm` | | `esm2` | Which ESM model to use. Can be `esm1b` or `esm2`. Note that this refers to two separately trained versions of the DeepPeptide architecture. |

### Input format

The predictor takes fasta-formatted protein sequences as input. Multiline and 1-line formats work.

### Output format

`predict.py` produces the following files in the output directory:
- `output.json`: A JSON file containing the peptide predictions for each protein in the input data.
- `output.md`: A markdown file that displays predictions as tables and includes plots. This is the output that is displayed on the webserver.
- `SEQUENCE_NAME.png`: A plot of the predictions for a single protein sequence.


