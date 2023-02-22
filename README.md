# Graph-Based Averaged Perceptron for Statistical Dependency Parsing
This is a Python implementation of a graph-based averaged perceptron for statistical dependency parsing, using the Chu-Liu-Edmonds algorithm for decoding. The project was tested on training English and German, and is designed to work with .conll06 datasets.

## Project structure

The repository has the following structure:

- `data/`: directory containing the dataset files in `.conll06` format
- `src/`: directory containing the Python source code
    - `dataset.py`: implements the class to load and preprocess the dataset files
    - `decoder.py`: implements the graph-based decoder using the Chu-Liu-Edmonds algorithm
    - `evaluation.py`: implements the UAS and LAS evaluation metrics to evaluate performance of the model
    - `features.py`: implements the feature templates used to extract features from the input data
    - `model.py`: implements the averaged perceptron model used for training and prediction
- `main.py`: main program that loads the datasets and models

## How to use
To run the main program, use the following command:

```bash
python main.py [arguments]
```

Here are the available arguments:
- main.py`: main program that loads the datasets and models

- `--train-feature-file`: the file path for the training feature file
- `--save-extractor`: the file path to save the feature extractor
- `--extractor`: the type of feature extractor to use (to be used when you save the trained extractor)
- `--process-data`: whether to process the dataset or not (default is False)
- `--train-input`: the file path for the training input data
- `--train-dataset`: the file path to save the training dataset
- `--dev-input`: the file path for the development input data
- `--dev-dataset`: the file path to save the development dataset
- `--evaluate`: the file path for the evaluation data
- `--evaluate-output`: the file path to save the evaluation output
- `--weights`: the file path to load the weights of the trained model
- `--save-model`: the file path to save the trained model (default is None)
- `--num-process`: Number of processes to use for feature extraction and processing. Default is 0, meaning no multiprocessing is used.

Here is an example of how to train the model on German data, save the feature extractor, and save the model with the highest UAS score in a checkpoint:

```bash
python main.py \
  --train-feature-file data/german/train/tiger-2.2.train.conll06 \
  --save-extractor ./datasets/de/extractor.p \
  --train-input data/german/train/tiger-2.2.train.conll06 \
  --train-dataset ./datasets/de/train_dataset.p \
  --dev-input data/german/dev/tiger-2.2.dev.conll06.gold \
  --dev-dataset ./datasets/de/dev.p \
  --num_process 16 \
  --process-data 1 \
  --save-model ./checkpoint/de/
```

To evaluate the model on a test set, use the --evaluate and --evaluate-output arguments when running the main.py program:

```bash
python main.py --evaluate <path_to_test_dataset> --evaluate-output <path_to_output_file>
```

This command will evaluate the model on the specified test set and output the results to the specified output file.


## Feature Templates

The feature templates from the `features.py` file are as follows:

| Feature Template | Description |
| --- | --- |
| `hshape` | Shape of the head token (e.g., for "Church" -> "Xxxxxx") |
| `dshape` | Shape of the dependent token |
| `hform` | Form of the head token |
| `hpos` | POS tag of the head token |
| `dform` | Form of the dependent token |
| `dpos` | POS tag of the dependent token |
| `hform, hpos` | Form and POS tag of the head token |
| `dform, dpos` | Form and POS tag of the dependent token |
| `hform, dform` | Form of the head and dependent tokens |
| `hlemma, dlemma` | Lemma of the head and dependent tokens |
| `hlemma, dpos` | Lemma of the head token and POS tag of the dependent token |
| `hpos, dlemma` | POS tag of the head token and lemma of the dependent token |
| `hform, dpos` | Form of the head token and POS tag of the dependent token |
| `hpos, dform` | POS tag of the head token and form of the dependent token |
| `hpos, dpos` | POS tags of the head and dependent tokens |
| `hform pos, dform, dpos` | Form and POS tag of the head and dependent tokens |
| `hform, hpos, dform` | Form and POS tag of the head token, and form of the dependent token |
| `hform, hpos, dpos` | Form and POS tag of the head token, and POS tag of the dependent token |
| `hform, dform, dpos` | Form of the head token, form of the dependent token, and POS tag of the dependent token |
| `hpos, dform, dpos` | POS tag of the head token, form of the dependent token, and POS tag of dependent token |
| `hpos, dpos, hpos+1, dpos-1` | The POS tag of the head word, the POS tag of the dependent word, and the POS tag of the next word after the head and the previous word before the dependent |
| `hpos, dpos, hpos-1, dpos+1` | The POS tag of the head word, the POS tag of the dependent word, and the POS tag of the previous word before the head and the next word after the dependent |
| `hpos, dpos, hpos+1, dpos+1` | The POS tag of the head word, the POS tag of the dependent word, and the POS tag of the next word after both head and dependent |
| `hpos, dpos, hpos-1, dpos-1` | The POS tag of the head word, the POS tag of the dependent word, and the POS tag of the previous word before both head and dependent |
| `hpos [bpos...] dpos` | The sequence of POS tags between the head and the dependent |
| `hpos bpos dpos` | A POS tag between the head and the dependent -- this goes through each POS tag between and outputs as a feature |