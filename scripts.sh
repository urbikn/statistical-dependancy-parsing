# Simple pass

# Full
python main.py --train-feature-file data/english/train/wsj_train.conll06 --save-extractor checkpoint/extractor_en.p

python main.py --extractor extractor_en.p --train-dataset data/english/train/wsj_train.first-5k.conll06 --save-train-dataset dataset-train-en-5k.p --dev-dataset-features dataset_dev.p 

# Train on 1k english and save, dev 1k english and save, with multiprocess=8
python main.py --extractor checkpoint/1k-en/extractor.p --train-dataset data/english/train/wsj_train.first-1k.conll06 --save-train-dataset checkpoint/1k-en/dataset-train-en-1k.p --dev-dataset data/english/dev/wsj_dev.conll06.gold --save-dev-dataset checkpoint/1k-en/dataset-dev-en.p --multiprocess 8


# ===== Train extractor, train and dev dataset =====
python main.py --train-feature-file data/english/train/wsj_train.first-5k.conll06 --save-extractor ./datasets/en_extractor_5k.p --train-input data/english/train/wsj_train.first-5k.conll06 --train-dataset ./datasets/en_train_dataset_5k.p --dev-input data/english/dev/wsj_dev.conll06.gold --dev-dataset ./datasets/en_dev.p --num_process 32 --process-data 1

# ==== Train train and dev dataset ====
python main.py --extractor ./datasets/en_extractor_5k.p --train-input data/english/train/wsj_train.first-5k.conll06 --train-dataset ./datasets/en_train_dataset_5k.p --dev-input data/english/dev/wsj_dev.conll06.gold --dev-dataset ./datasets/en_dev.p --num_process 32 --process-data 1

# ==== Use train and dev dataset ====
python main.py --extractor ./datasets/en_extractor_5k.p --train-dataset ./datasets/en_train_dataset_5k.p --dev-dataset ./datasets/en_dev.p

# ==== Use train and dev dataset /w perceptron weights ====
python main.py --extractor ./datasets/en_extractor_5k.p --train-dataset ./datasets/en_train_dataset_5k.p --dev-dataset ./datasets/en_dev.p --weights ./datasets/perceptron_en_5k.p
