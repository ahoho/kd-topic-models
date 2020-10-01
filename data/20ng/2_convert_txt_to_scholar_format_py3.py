# python 3.8
from pathlib import Path
import json

from scipy import sparse
import numpy as np

from utils import save_sparse, load_json, save_json

if __name__ == "__main__":
    outdir = Path("aligned")
    outdir.mkdir(exist_ok=True)

    # data files
    train = np.loadtxt("intermediate/train.txt")
    test = np.loadtxt("intermediate/test.txt")

    train = sparse.coo_matrix(train)
    test = sparse.coo_matrix(test)

    save_sparse(train, "aligned/train")
    save_sparse(test,"aligned/test")

    # reorder vocabulary, save as list
    vocab = load_json("intermediate/vocab_dict.json")
    vocab = [# ensure correct order
        k for k, v in sorted(vocab.items(), key=lambda kv: kv[1])
    ]
    save_json(vocab, "aligned/train.vocab.json")