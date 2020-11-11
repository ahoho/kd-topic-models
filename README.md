# Improving Neural Topic Models using Knowledge Distillation

Repo for our EMNLP 2020 paper. We will clean up the implementation for improved ease-of-use, but provide the code included in our original submission for the time being. 

If you use this code, please use the following citation:
```
@inproceedings{hoyle-etal-2020-improving,
    title = "Improving Neural Topic Models Using Knowledge Distillation",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Resnik, Philip",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.137",
    pages = "1752--1771",
}
```

# Rough Steps

1. As of now, you'll need two conda environments to run both the BERT teacher and topic modeling student (which is a modification of [Scholar](https://github.com/dallascard/scholar)). The environment files are defined in `teacher/teacher.yml` and `scholar/scholar.yml` for the teacher and topic model, respectively. For example:
    `conda env create -f teacher/teacher.yml`
    (edit the first line in the `yml` file if you want to change the name of the resulting environment; the default is `transformers28`).

2. We don't have a general-purpose data processing pipeline together, but you can use the IMDb format as a guide:
```
conda activate scholar
python data/imdb/download_imdb.py

# main preprocessing script
python preprocess_data.py data/imdb/train.jsonlist data/imdb/processed --vocab_size 5000 --test data/imdb/test.jsonlist
# create a dev split from the train data
create_dev_split.py
```

3. Run the teacher model. Below is what we used for IMDb
```
conda activate transformers28

python teacher/bert_reconstruction.py \
    --input-dir ./data/imdb/processed-dev \
    --output-dir ./data/imdb/processed-dev/logits \ 
    --do-train \
    --evaluate-during-training \
    --truncate-dev-set-for-eval 120 \
    --logging-steps 200 \
    --save-steps 1000 \
    --num-train-epochs 6 \
    --seed 42 \
    --num-workers 4 \
    --batch-size 20 \
    --gradient-accumulation-steps 8 \
    --document-split-pooling mean-over-logits
```

4. Collect the logits from the teacher model (the `--checkpoint-folder-pattern` argument accepts grub pattern matching in case you want to create logits for different stages of training; be sure to enclose in double quotes `"`)
```
conda activate transformers28

python teacher/bert_reconstruction.py \
    --output-dir ./data/imdb/processed-dev/logits \
    --seed 42 \
    --num-workers 6 \
    --get-reps \
    --checkpoint-folder-pattern "checkpoint-9000" \
    --save-doc-logits \
    --no-dev
```

5. Run the topic model (there are a number of extraneous experimental arguments in `run_scholar.py`, which we intend to strip out in a future version).
```
conda activate scholar

python scholar/run_scholar.py \
    ./data/imdb/processed-dev \
    --dev-metric npmi \
    -k 50 \
    --epochs 500 \
    --patience 500 \
    --batch-size 200 \
    --background-embeddings \
    --device 0 \
    --dev-prefix dev \
    -lr 0.002 \
    --alpha 0.5 \
    --eta-bn-anneal-step-const 0.25 \
    --doc-reps-dir ./data/imdb/processed-dev/logits/checkpoint-9000/doc_logits \
    --use-doc-layer \
    --no-bow-reconstruction-loss \
    --doc-reconstruction-weight 0.5 \
    --doc-reconstruction-temp 1.0 \
    --doc-reconstruction-logit-clipping 10.0 \
    -o ./outputs/imdb
```

