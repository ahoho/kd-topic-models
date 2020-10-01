import json

import numpy as np
from scipy import sparse


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix(
        (npy['data'], (npy['row'], npy['col'])), shape=npy['shape']
    )
    return coo_matrix.tocsc()


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def load_json(fpath):
    with open(fpath, 'r') as i:
        return json.load(i)


def save_json(obj, fpath):
    with open(fpath, 'w') as o:
        json.dump(obj, o)


def load_jsonlist(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as i:
        data = [json.loads(line) for line in i]
    return data


def save_jsonlist(dicts, fpath):
    with open(fpath, 'w', encoding='utf-8') as o:
        for d in dicts:
            json.dump(d, o)
            o.write('\n')