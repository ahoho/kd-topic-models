import json

import numpy as np

from utils import (
    reverse_dict, load_sparse, load_json, save_sparse, save_json, save_jsonlist
)

from core import Data


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class ProcessedData(Data):
    def __init__(self, batch_size, data_path='', ctx=None, saveto='', **kwargs):
        self.saveto = saveto
        super(ProcessedData, self).__init__(batch_size, data_path, ctx)

    def load(self, path, features='BoW', match_avitm=True):
        if path[:2] == '~/':
            path = os.path.join(os.path.expanduser(path[:2]), path[2:])

        ### Specify the file locations
        train_path = path + '/train.npz'
        dev_path = path + '/dev.npz'
        test_path = path + '/test.npz'
        vocab_path = path + '/train.vocab.json'

        ### Load train
        train_csr = load_sparse(train_path)
        train = np.array(train_csr.todense()).astype('float32')

        ### Load dev
        self.dev_counts = load_sparse(dev_path).tocsc() # will be used for NPMI

        ### Load test
        test_csr = load_sparse(test_path)
        test = np.array(test_csr.todense()).astype('float32')

        ### load vocab
        # ENCODING = "ISO-8859-1"
        ENCODING = "utf-8"
        with open(vocab_path, encoding=ENCODING) as f:
             vocab_list = json.load(f)

        # construct maps
        vocab2dim = dict(zip(vocab_list, range(len(vocab_list))))
        dim2vocab = reverse_dict(vocab2dim)

        return [train, None, test, None, None, None], [None, None, None], [vocab2dim, dim2vocab, None, None]


class KDProcessedData(Data):
    def __init__(self, batch_size, data_path, logit_path, logit_clip=None, ctx=None, saveto='', **kwargs):
        self.saveto = saveto
        self.logit_path = logit_path
        self.logit_clip = logit_clip
        super(KDProcessedData, self).__init__(batch_size, data_path, ctx)

    def load(self, data_path, features='BoW', match_avitm=True):

        ### Specify the file locations
        train_path = data_path + '/train.npz'
        dev_path = data_path + '/dev.npz'
        test_path = data_path + '/test.npz'
        vocab_path = data_path + '/train.vocab.json'

        ### Load train
        train_csr = load_sparse(train_path)
        train_counts = np.array(train_csr.todense()).astype('float32')
        train_bert_logits = np.load(self.logit_path + "/train.npy")
        train = np.concatenate([train_counts, train_bert_logits], axis=1)

        if self.logit_clip is not None:
            # limit the document representations to the top k labels
            doc_tokens = np.sum(train_counts > 0, axis=1)
            vocab_size = train_counts.shape[1]
            
            for i, (row, total) in enumerate(zip(train_bert_logits, doc_tokens)):
                k = self.logit_clip * total # keep this many logits
                if k < vocab_size:
                    min_logit = np.quantile(row, 1 - k / vocab_size)
                    train_bert_logits[i, train_bert_logits[i] < min_logit] = -np.inf
        
        #min_logits = np.quantile(train_bert_logits, np.quantile(train_counts.sum(1), 0.9) / 20_000, axis=1)
        #train_bert_logits[train_bert_logits < min_logits.reshape(-1, 1)] = -np.inf
        
        ### Load dev
        self.dev_counts = load_sparse(dev_path).tocsc() # will be used for NPMI

        ### Load test
        test_csr = load_sparse(test_path)
        test_counts = np.array(test_csr.todense()).astype('float32')
        test_bert_logits = np.ones_like(test_counts)
        test = np.concatenate([test_counts, test_bert_logits], axis=1)

        ### load vocab
        # ENCODING = "ISO-8859-1"
        ENCODING = "utf-8"
        with open(vocab_path, encoding=ENCODING) as f:
             vocab_list = json.load(f)
        
        # construct maps
        vocab2dim = dict(zip(vocab_list, range(len(vocab_list))))
        dim2vocab = reverse_dict(vocab2dim)

        return [train, None, test, None, None, None], [None, None, None], [vocab2dim, dim2vocab, None, None]
