import argparse
import pickle
import gzip
from tqdm import trange, tqdm
from multiprocessing import Pool


from src.dataset import ConllDataset
from src.features import FeatureMapping
from src.model import AveragePerceptron
from src import evaluation

def train_feature_extractor(input_file, output_file):
    dataset = ConllDataset(input_file)
    feature_extractor = FeatureMapping()

    print('== Training feature extractor ==')
    for i in trange(len(dataset)):
        sentence = dataset[i]
        feature_extractor.get(sentence)

    print('Saving feature extractor to:', output_file)
    FeatureMapping.save(feature_extractor, output_file)

def get_dataset(input_file, extractor, type='training', num_process=32):
    dataset = ConllDataset(input_file)
    train_dataset = []

    batch = []
    items_per_process = 32
    for i, instance in tqdm(enumerate(dataset, start=1), total=len(dataset), desc=f'Extracting features from {type} dataset'):
        batch.append(instance)

        # This small trick because for each process we have to copy the extractor
        if i % (num_process * items_per_process)  == 0 or i == len(dataset):
            with Pool(num_process) as pool:
                output = pool.map(extractor.get_permutations, batch, items_per_process)
                
                for l in range(len(batch)):
                    train_dataset.append([
                        output[l],
                        batch[l].get_arcs()
                    ])
            batch = []
    return train_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feature-file', type=str)
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--save-extractor', type=str, default='./extractor.p')

    parser.add_argument('--process-data', type=bool, default=False)
    parser.add_argument('--train-input', type=str)
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--dev-input', type=str)
    parser.add_argument('--dev-dataset', type=str)
    parser.add_argument('--test-input', type=str)
    parser.add_argument('--test-dataset', type=str)

    parser.add_argument('--weights', type=str)
    parser.add_argument('--num_process', type=int, default=0)
    args = parser.parse_args()

    # ==== Feature extractor ====
    if args.train_feature_file:
        train_feature_extractor(args.train_feature_file, args.save_extractor)
        feature_extractor = FeatureMapping.load(args.save_extractor)
    if args.extractor:
        feature_extractor = FeatureMapping.load(args.extractor)
    feature_extractor.frozen = True

    # ==== Perceptron ====
    if args.weights:
        model = AveragePerceptron.load(args.weights)
    else:
        model = AveragePerceptron(feature_extractor, dim=len(feature_extractor) + 1, weight_init=1)

    # == Processing input files ==
    if args.process_data:
        if args.train_input:
            train_dataset = get_dataset(args.train_input, feature_extractor, 'training', args.num_process)

            with gzip.open(args.train_dataset,'wb') as stream:
                pickle.dump(train_dataset,stream,-1)

        if args.dev_input:
            dev_dataset = get_dataset(args.dev_input, feature_extractor, 'development', args.num_process)

            with gzip.open(args.dev_dataset,'wb') as stream:
                pickle.dump(dev_dataset,stream,-1)

        if args.test_input:
            pass
            # test_dataset = get_test_dataset(args.test_dataset, feature_extractor, 32)
            test_dataset = None

            with gzip.open(args.test_dataset,'wb') as stream:
                pickle.dump(test_dataset,stream,-1)

    # == Processing training dataset ==
    if args.train_dataset:
        with gzip.open(args.train_dataset,'rb') as stream:
            train_dataset = pickle.load(stream)
            # train_dataset = feature_extractor.features_to_tensors(dataset)

    if args.dev_dataset:
        with gzip.open(args.dev_dataset,'rb') as stream:
            dev_dataset = pickle.load(stream)
            # dev_dataset = feature_extractor.features_to_tensors(dataset)

    if args.test_dataset:
        with gzip.open(args.test_dataset,'rb') as stream:
            dataset = pickle.load(stream)
            test_dataset = feature_extractor.features_to_tensors(dataset)

    model.train(train_dataset, dev_dataset, epoch=20, batch_n=1000, eval_interval=2500, learning_rate=0.01)
    AveragePerceptron.save(model, 'datasets/perceptron_en_5k_epoch40.p')
