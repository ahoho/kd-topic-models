import os
import json
import codecs
import pickle
import numpy as np
from scipy import sparse

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data


def read_jsonlist(input_filename):
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            yield json.loads(line, encoding='utf-8')

class LazyJsonlistReader:
    """
    Adapted from https://stackoverflow.com/a/39564774/5712749
    Since original code loops and re-loops through a large list, created by a non-lazy
    version of `read_jsonlist`, we want a resetting generator

    Usage:
        `data = LazyJsonlistReader(fpath)`
    """
    def __init__(self, fpath):
        self.iterator_factory = lambda: read_jsonlist(fpath)
        self._fpath = fpath
        self._len = None

    def __iter__(self):
        """
        Will reset the generator each time this is called
        """
        return self.iterator_factory()

    def __len__(self):
        """
        Read length of dataset and cache
        """
        if self._len is None:
            with open(self._fpath) as f:
                for i, l in enumerate(f):
                    pass
            self._len = i + 1
        return self._len



def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for i, line in enumerate(lines):
            output_file.write(line)
            if add_newlines and i < len(lines) - 1:
                output_file.write('\n')
        
        if add_final_newline:
            output_file.write('\n')


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


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()
