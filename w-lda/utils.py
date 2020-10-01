# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import socket
import pickle
import argparse
import time
import os
import json
from functools import reduce
import numpy as np
from scipy.special import logit
import scipy.sparse as sparse
import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
from tqdm import tqdm
import mxnet as mx
# from sklearn.neighbors import NearestNeighbors
# from IPython import embed
import collections


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


def compute_npmi_at_n_during_training(beta, ref_counts, n=10, smoothing=0.0):
    
    n_docs, _ = ref_counts.shape

    n_topics, vocab_size = beta.shape

    npmi_means = {}
    for k in range(n_topics):
        order = np.argsort(beta[k, :])[::-1]
        indices = order[:n]
        npmi_vals = []
        for i, index1 in enumerate(indices):
            for index2 in indices[i+1:n]:
                col1 = np.array((ref_counts[:, index1] > 0).todense(), dtype=int) + smoothing
                col2 = np.array((ref_counts[:, index2] > 0).todense(), dtype=int) + smoothing
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = np.sum(col1 * col2)
                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        npmi_means[k] = np.mean(npmi_vals)
    return npmi_means


def gpu_helper(gpu):
    if gpu >= 0 and gpu_exists(gpu):
        model_ctx = mx.gpu(gpu)
    else:
        model_ctx = mx.cpu()
    return model_ctx

def gpu_exists(gpu):
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(gpu))
    except:
        return False
    return True

def reverse_dict(d):
    return {v:k for k,v in d.items()}

def to_numpy(X):
    x_npy = []
    for x in X:
        if isinstance(x,list):
            x_npy += [to_numpy(x)]
        else:
            x_npy += [x.asnumpy()]
    return x_npy

def stack_numpy(X,xnew):
    for i in range(len(X)):
        if isinstance(xnew[i],list):
            X[i] = stack_numpy(X[i], xnew[i])
        else:
            X[i] = np.vstack([X[i], xnew[i]])
    return X


def get_topic_words_decoder_weights(D, data, ctx, k=10, decoder_weights=False):
    if decoder_weights:
        params = D.collect_params()
        params = params['decoder0_dense0_weight'].data().transpose()
    else:
        y = D.y_as_topics()
        params = D(y.copyto(ctx))
    top_word_ids = mx.nd.argsort(params, axis=1, is_ascend=False)[:,:k].asnumpy()
    if hasattr(data, 'id_to_word'):
        top_word_strings = [[data.id_to_word[int(w)] for w in topic] for topic in top_word_ids]
    else:
        top_word_strings = [[data.maps['dim2vocab'][int(w)] for w in topic] for topic in top_word_ids]

    return top_word_strings, params.asnumpy()


def get_topic_words(D, data, ctx, k=10):
    y, z = D.yz_as_topics()
    if z is not None:
        params = D(y.copyto(ctx), z.copyto(ctx))
    else:
        params = D(y.copyto(ctx), None)
    top_word_ids = mx.nd.argsort(params, axis=1, is_ascend=False)[:,:k].asnumpy()
    if hasattr(data, 'id_to_word'):
        top_word_strings = [[data.id_to_word[int(w)] for w in topic] for topic in top_word_ids]
    else:
        top_word_strings = [[data.maps['dim2vocab'][int(w)] for w in topic] for topic in top_word_ids]

    return top_word_strings


def calc_topic_uniqueness(top_words_idx_all_topics):
    """
    This function calculates topic uniqueness scores for a given list of topics.
    For each topic, the uniqueness is calculated as:  (\sum_{i=1}^n 1/cnt(i)) / n,
    where n is the number of top words in the topic and cnt(i) is the counter for the number of times the word
    appears in the top words of all the topics.
    :param top_words_idx_all_topics: a list, each element is a list of top word indices for a topic
    :return: a dict, key is topic_id (starting from 0), value is topic_uniquness score
    """
    n_topics = len(top_words_idx_all_topics)

    # build word_cnt_dict: number of times the word appears in top words
    word_cnt_dict = collections.Counter()
    for i in range(n_topics):
        word_cnt_dict.update(top_words_idx_all_topics[i])

    uniqueness_dict = dict()
    for i in range(n_topics):
        cnt_inv_sum = 0.0
        for ind in top_words_idx_all_topics[i]:
            cnt_inv_sum += 1.0 / word_cnt_dict[ind]
        uniqueness_dict[i] = cnt_inv_sum / len(top_words_idx_all_topics[i])

    return uniqueness_dict

def request_pmi(topic_dict=None, filename='', port=1234):
    try:
        # create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # get local machine name
        host = socket.gethostname()
        # host = socket.gethostbyname('localhost')

        # connection to hostname on the port.
        s.connect((host, port))

        if filename != '':
            s.sendall(pickle.dumps(filename), )
        else:
            s.send(pickle.dumps(topic_dict), )

        data = []
        while True:
            packet = s.recv(4096)
            # time.sleep(1.0)
            # print('looking at packet # {0}'.format(len(data)))
            # print(packet)
            # print(type(packet))
            wait = len(packet)
            if not packet:
                # embed()
                break
            data.append(packet)
            # print('received packet # {0}'.format(len(data)))
            # time.sleep(1.0)
        res_dict = pickle.loads(b"".join(data))

        s.close()
        pmi_dict = res_dict['pmi_dict']
        npmi_dict = res_dict['npmi_dict']
    except:
        # print('Failed to run NPMI calc, NPMI and PMI set to 0.0')
        pmi_dict = dict()
        npmi_dict = dict()
        for k in topic_dict:
            pmi_dict[k] = 0
            npmi_dict[k] = 0
        # embed()

    return pmi_dict, npmi_dict


def print_topics(topic_json, npmi_dict, topic_uniqs, data, print_topic_names=False):
    for k,v in topic_json.items():
        prefix_msg = '[ '
        if hasattr(data, 'maps') and print_topic_names:
            prefix_msg += data.maps['dim2topic'][k]
        else:
            prefix_msg += str(k)
        if hasattr(data, 'selected_topics') and print_topic_names:
            if data.maps['dim2topic'][k] in data.selected_topics:
                prefix_msg += '*'
        prefix_msg += ' - '
        prefix_msg += '{:.5g}'.format(topic_uniqs[k])
        prefix_msg += ' - '
        prefix_msg += '{:.5g}'.format(npmi_dict[k])
        prefix_msg += ']: '
        print(prefix_msg, v)

def print_topic_with_scores(topic_json, **kwargs):
    """

    :param topic_json:
    :param kwargs: dict_name: content_dict; special argument sortby='xxx' will enable descending sort in printed result
    :return:
    """
    topic_keys = sorted(list(topic_json.keys()))

    sortby = kwargs.pop('sortby', None)
    if sortby is None:
        sortby = kwargs.pop('sort_by', None)

    if sortby in kwargs.keys():
        topic_keys = sorted(kwargs[sortby], key=kwargs[sortby].get)[::-1]

    entries = []
    dict_names = sorted(list(kwargs.keys()))
    header_str = 'Avg scores: '
    for dn in dict_names:
        assert isinstance(kwargs[dn], dict)
        header_str += '{}: {:.2f} '.format(dn, mean_dict(kwargs[dn]))
    for k in topic_keys:
        score_str = []
        for dn in dict_names:
            # score_str.append('{} {:.2f}'.format(dn, kwargs[dn][k]))
            score_str.append('{:.2f}'.format(kwargs[dn][k]))
        score_str = ', '.join(score_str)
        entries.append('T{} [{}] '.format(k, score_str) + ', '.join(topic_json[k]))

    msg = header_str + '\n' + '\n'.join(entries)
    print(msg)
    return msg