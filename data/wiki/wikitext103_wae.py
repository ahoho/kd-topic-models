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

import os
import re
import time
import argparse

import numpy as np
import scipy.sparse as sparse 
from tqdm import tqdm
import nltk    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import utils

def reverse_dict(d):
        return {v:k for k,v in d.items()}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--lemmatize', dest="lemmatize", action="store_true")
    parser.add_argument('--do-not-lemmatize', dest="lemmatize", action="store_false")

    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=20000)

    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=1)
    
    parser.add_argument("--intro-only", action='store_true', default=False)
    parser.add_argument("--truncate-train-data", type=float, default=None)

    parser.add_argument("--dev-split", type=float)
    parser.add_argument("--test-split", type=float)

    args = parser.parse_args()

    data_dir = 'raw'

    # download data
    if not os.path.exists(f'{data_dir}/wiki.train.tokens'):
        os.system("curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip")
        os.system("unzip wikitext-103-v1.zip")
        os.system(f"mv wikitext-103 {data_dir}")
        os.system("rm wikitext-103-v1.zip")

    # parse into documents
    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] == '=' and line[-1] == '=':
            if line[2] != '=':
                return True
            else:
                return False
        else:
            return False


    def token_list_per_doc(input_dir, token_file):
        lines_list = []
        line_prev = ''
        prev_line_start_doc = False
        with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
            for l in f:
                line = l.strip()
                if prev_line_start_doc and line:
                    # the previous line should not have been start of a document!
                    lines_list.pop()
                    lines_list[-1] = lines_list[-1] + ' ' + line_prev

                if line:
                    if is_document_start(line) and not line_prev:
                        lines_list.append(line)
                        prev_line_start_doc = True
                    else:
                        lines_list[-1] = lines_list[-1] + ' ' + line
                        prev_line_start_doc = False
                else:
                    prev_line_start_doc = False
                line_prev = line

        print("{} documents parsed!".format(len(lines_list)))
        return lines_list

    train_file = 'wiki.train.tokens'
    train_doc_list = token_list_per_doc(data_dir, train_file)

    if args.intro_only is not None:
        # only keep the introduction of a document
        train_doc_list = [
            doc[:doc.index('= =')] if '= =' in doc else doc
            for doc in train_doc_list
        ]

    if args.dev_split is None:
        val_file = 'wiki.valid.tokens'
        test_file = 'wiki.test.tokens'
        val_doc_list = token_list_per_doc(data_dir, val_file)
        test_doc_list = token_list_per_doc(data_dir, test_file)
    else:
        held_out_split = args.dev_split + args.test_split

        train_doc_list, test_doc_list = train_test_split(
            train_doc_list, test_size=held_out_split, random_state=11235
        )
        val_doc_list, test_doc_list = train_test_split(
            test_doc_list, test_size=args.test_split / (held_out_split), random_state=11235
        )

    # artificially limit size of training data
    if args.truncate_train_data is not None:
        n = len(train_doc_list)
        train_doc_list = train_doc_list[:int(n * args.truncate_train_data)]

    nltk.download('wordnet')

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in doc.split() if len(t) >= 2 and re.match("[a-z].*", t)
                    and re.match(token_pattern, t)]

    print('Lemmatizing and counting, this may take a few minutes...')
    start_time = time.time()
    vectorizer = CountVectorizer(
        input='content',
        analyzer='word',
        stop_words='english',
        tokenizer=LemmaTokenizer() if args.lemmatize else None,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_df=args.max_df,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    train_vectors = vectorizer.fit_transform(tqdm(train_doc_list))
    val_vectors = vectorizer.transform(tqdm(val_doc_list))
    test_vectors = vectorizer.transform(tqdm(test_doc_list))

    vocab_list = vectorizer.get_feature_names()
    vocab_size = len(vocab_list)
    print('vocab size:', vocab_size)
    print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))

    def shuffle_and_dtype(docs, vectors):
        idx = np.arange(vectors.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        docs = [docs[i] for i in idx]
        vectors = vectors[idx]
        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        print(type(vectors), vectors.dtype)
        return docs, vectors

    train_doc_list, train_vectors = shuffle_and_dtype(train_doc_list, train_vectors)
    val_doc_list, val_vectors = shuffle_and_dtype(val_doc_list, val_vectors)
    test_doc_list, test_vectors = shuffle_and_dtype(test_doc_list, test_vectors)

    os.makedirs(args.output_dir, exist_ok=True)

    utils.save_json(vocab_list, f"{args.output_dir}/train.vocab.json")

    train_ids = list(range(len(train_doc_list)))
    val_ids = list(range(len(val_doc_list)))
    test_ids = list(range(len(test_doc_list)))

    # save ids
    utils.save_json(train_ids, f"{args.output_dir}/train.ids.json")
    utils.save_json(val_ids, f"{args.output_dir}/dev.ids.json")
    utils.save_json(test_ids, f"{args.output_dir}/test.ids.json")

    # save the raw text
    utils.save_jsonlist(
        ({"id": id, "text": text} for id, text in zip(train_ids, train_doc_list)),
        f"{args.output_dir}/train.jsonlist",
    )
    utils.save_jsonlist(
        ({"id": id, "text": text} for id, text in zip(val_ids, val_doc_list)),
        f"{args.output_dir}/dev.jsonlist",
    )
    utils.save_jsonlist(
        ({"id": id, "text": text} for id, text in zip(test_ids, test_doc_list)),
        f"{args.output_dir}/test.jsonlist",
    )

    utils.save_sparse(train_vectors, f'{args.output_dir}/train.npz')
    utils.save_sparse(val_vectors, f'{args.output_dir}/dev.npz')
    utils.save_sparse(test_vectors, f'{args.output_dir}/test.npz')
