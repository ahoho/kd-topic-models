import os
import sys
import argparse

import gensim
import git
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import file_handling as fh
from scholar import Scholar
from compute_npmi import compute_npmi_at_n_during_training


def main(call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument(
        "-k",
        dest="n_topics",
        type=int,
        default=20,
        help="Size of latent representation (~num topics)",
    )
    parser.add_argument(
        "-l",
        dest="learning_rate",
        type=float,
        default=0.002,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--eta-bn-anneal-step-const",
        type=float,
        default=0.75,
        help="When to terminate batch-norm annealing, as a percentage of total epochs"
    )

    parser.add_argument(
        "-m",
        dest="momentum",
        type=float,
        default=0.99,
        help="beta1 for Adam",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=200,
        help="Size of minibatches",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait without improvement to dev-metric",
    )
    parser.add_argument(
        "--dev-metric",
        dest="dev_metric",
        type=str,
        default="perplexity",  # TODO: constrain options
        help="Optimize accuracy, perplexity, or internal npmi",
    )
    parser.add_argument(
        "--npmi-words",
        type=int,
        default=10,
        help="Number of words to use when calculating npmi"
    )
    parser.add_argument(
        "--train-prefix",
        type=str,
        default="train",
        help="Prefix of train set",
    )
    parser.add_argument(
        "--dev-prefix",
        type=str,
        default=None,
        help="Prefix of dev set.",
    )
    parser.add_argument(
        "--test-prefix",
        type=str,
        default=None,
        help="Prefix of test set",
    )
    parser.add_argument(
        "--no-bow-reconstruction-loss",
        action="store_false",
        dest="reconstruct_bow",
        default=True,
        help="Include the standard reconstruction of document word counts",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Read labels from input_dir/[train|test].labels.csv",
    )
    parser.add_argument(
        "--prior-covars",
        type=str,
        default=None,
        help="Read prior covariates from files with these names (comma-separated)",
    )
    parser.add_argument(
        "--topic-covars",
        type=str,
        default=None,
        help="Read topic covariates from files with these names (comma-separated)",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        default=False,
        help="Use interactions between topics and topic covariates",
    )
    parser.add_argument(
        "--no-covars-predict",
        action="store_false",
        dest="covars_predict",
        default=True,
        help="Do not use covariates as input to classifier",
    )
    parser.add_argument(
        "--no-topics-predict",
        action="store_false",
        dest="topics_predict",
        default=True,
        help="Do not use topics as input to classifier",
    )
    parser.add_argument(
        "--min-prior-covar-count",
        type=int,
        default=None,
        help="Drop prior covariates with less than this many non-zero values in the training dataa",
    )
    parser.add_argument(
        "--min-topic-covar-count",
        type=int,
        default=None,
        help="Drop topic covariates with less than this many non-zero values in the training dataa",
    )
    parser.add_argument(
        "--classifier_loss_weight",
        type=float,
        default=1.0,
        help="Weight to give portion of loss from classification",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        default=False,
        help="Use default regularization",
    )
    parser.add_argument(
        "--l1-topics",
        type=float,
        default=0.0,
        help="Regularization strength on topic weights",
    )
    parser.add_argument(
        "--l1-topic-covars",
        type=float,
        default=0.0,
        help="Regularization strength on topic covariate weights",
    )
    parser.add_argument(
        "--l1-interactions",
        type=float,
        default=0.0,
        help="Regularization strength on topic covariate interaction weights",
    )
    parser.add_argument(
        "--l2-prior-covars",
        type=float,
        default=0.0,
        help="Regularization strength on prior covariate weights",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="Restart training with model in output-dir",
    )
    
    parser.add_argument(
        "--save-at-training-end",
        action="store_true",
        default=False,
        help="Save model at the end of training",
    )

    parser.add_argument(
        "--emb-dim",
        type=int,
        default=300,
        help="Dimension of input embeddings",
    )

    parser.add_argument(
        "--background-embeddings",
        nargs="?",
        const='random',
        help="`--background-embeddings <optional path to embeddings>`"
    )
    parser.add_argument(
        "--deviation-embeddings",
        nargs="?",
        const='random',
        help="`--deviation-embeddings <optional path to embeddings>`"
    )
    parser.add_argument(
        "--deviation-embedding-covar",
        help="The covariate by which to vary the embeddings"
    )

    parser.add_argument(
        "--fix-background-embeddings",
        dest="update_background_embeddings",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--fix-deviation-embeddings",
        dest="update_deviation_embeddings",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--ignore-deviation-embeddings",
        action="store_true",
        default=False,
        help="Experimental baseline to maintain parameter number",
    )
    parser.add_argument(
        "--zero-out-embeddings",
        action="store_true",
        default=False,
        help="Experimental switch to set all embeddings to 0",
    )

    parser.add_argument(
        "--doc-reps-dir",
        help="Use document representation & specify the location",
    )
    parser.add_argument(
        "--doc-reconstruction-weight",
        type=float,
        default=None,
        help="How much to weigh doc repesentation reconstruction (0 means none)",
    )
    parser.add_argument(
        "--doc-reconstruction-temp",
        type=float,
        default=None,
        help="Temperature to use when softmaxing over the doc reconstruction logits",
    )
    parser.add_argument(
        "--doc-reconstruction-min-count",
        type=float,
        default=0.,
        help="Minimum pseudo-count to accept",
    )
    parser.add_argument(
        "--doc-reconstruction-logit-clipping",
        type=float,
        default=None,
        help="Keep only the teacher logits corresponding to the top `N * x` unique words for each doc",
    )
    parser.add_argument(
        "--attend-over-doc-reps",
        action="store_true",
        default=False,
        help="Attend over the doc-representation sequence",
    )
    parser.add_argument(
        "--use-doc-layer",
        action="store_true",
        default=False,
        help="Use a document projection layer",
    )
    parser.add_argument(
        "--classify-from-doc-reps",
        action="store_true",
        help="Use document representations to classify?"
    )
    parser.add_argument(
        "--randomize-doc-reps",
        action="store_true",
        help="Baseline to randomize the document representations"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Hyperparameter for logistic normal prior",
    )
    parser.add_argument(
        "--no-bg",
        action="store_true",
        default=False,
        help="Do not use background freq",
    )
    parser.add_argument(
        "--dev-folds",
        type=int,
        default=0,
        help="Number of dev folds. Ignored if --dev-prefix is used. default=%default"
    )
    parser.add_argument(
        "--dev-fold",
        type=int,
        default=0,
        help="Fold to use as dev (if dev_folds > 0). Ignored if --dev-prefix is used. default=%default",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="GPU to use"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed"
    )

    options = parser.parse_args(call)

    input_dir = options.input_directory

    if options.r:
        options.l1_topics = 1.0
        options.l1_topic_covars = 1.0
        options.l1_interactions = 1.0
    
    if options.dev_prefix:
        options.dev_folds = 0
        options.dev_fold = 0

    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None

    # load the training data
    train_X, vocab, train_row_selector, train_ids = load_word_counts(
        input_dir, options.train_prefix
    )
    train_labels, label_type, label_names, n_labels = load_labels(
        input_dir, options.train_prefix, train_row_selector, options.labels
    )
    (
        train_prior_covars,
        prior_covar_selector,
        prior_covar_names,
        n_prior_covars,
    ) = load_covariates(
        input_dir,
        options.train_prefix,
        train_row_selector,
        options.prior_covars,
        options.min_prior_covar_count,
    )
    (
        train_topic_covars,
        topic_covar_selector,
        topic_covar_names,
        n_topic_covars,
    ) = load_covariates(
        input_dir,
        options.train_prefix,
        train_row_selector,
        options.topic_covars,
        options.min_topic_covar_count,
    )

    print("Loading document representations")
    train_doc_reps = load_doc_reps(
        options.doc_reps_dir,
        prefix=options.train_prefix,
        row_selector=train_row_selector,
        use_sequences=options.attend_over_doc_reps,
    )
    options.n_train, vocab_size = train_X.shape
    options.n_labels = n_labels

    if (
        options.doc_reconstruction_logit_clipping is not None 
        and options.doc_reconstruction_logit_clipping > 0
    ):
        # limit the document representations to the top k labels
        doc_tokens = np.array((train_X > 0).sum(1)).reshape(-1)

        for i, (row, total) in enumerate(zip(train_doc_reps, doc_tokens)):
            k = options.doc_reconstruction_logit_clipping * total # keep this many logits
            if k < vocab_size:
                min_logit = np.quantile(row, 1 - k / vocab_size)
                train_doc_reps[i, train_doc_reps[i] < min_logit] = -np.inf
    
    if n_labels > 0:
        print("Train label proportions:", np.mean(train_labels, axis=0))
    
    # split into training and dev if desired
    train_indices, dev_indices = train_dev_split(options, rng)
    train_X, dev_X = split_matrix(train_X, train_indices, dev_indices)
    train_labels, dev_labels = split_matrix(train_labels, train_indices, dev_indices)
    train_prior_covars, dev_prior_covars = split_matrix(
        train_prior_covars, train_indices, dev_indices
    )
    train_topic_covars, dev_topic_covars = split_matrix(
        train_topic_covars, train_indices, dev_indices
    )
    train_doc_reps, dev_doc_reps = split_matrix(
        train_doc_reps, train_indices, dev_indices
    )
    if dev_indices is not None:
        dev_ids = [train_ids[i] for i in dev_indices]
        train_ids = [train_ids[i] for i in train_indices]
    else:
        dev_ids = None
    
    doc_reps_dim = train_doc_reps.shape[-1] if options.doc_reps_dir else None
    n_train, _ = train_X.shape

    # load the dev data
    if options.dev_prefix is not None:
        dev_X, _, dev_row_selector, dev_ids = load_word_counts(
            input_dir, options.dev_prefix, vocab=vocab
        )
        dev_labels, _, _, _ = load_labels(
            input_dir, options.dev_prefix, dev_row_selector, options.labels
        )
        dev_prior_covars, _, _, _ = load_covariates(
            input_dir,
            options.dev_prefix,
            dev_row_selector,
            options.prior_covars,
            covariate_selector=prior_covar_selector,
        )
        dev_topic_covars, _, _, _ = load_covariates(
            input_dir,
            options.dev_prefix,
            dev_row_selector,
            options.topic_covars,
            covariate_selector=topic_covar_selector,
        )
        try:
            dev_doc_reps = load_doc_reps(
                options.doc_reps_dir,
                prefix=options.dev_prefix,
                row_selector=dev_row_selector,
                use_sequences=options.attend_over_doc_reps,
            )
        except FileNotFoundError:
            print("Dev document representation not found, will set to log of dev_X")
            dev_doc_reps = np.log(np.array(dev_X.todense()) + 1e-10) # HACK
    
    # load the test data
    test_ids = None
    if options.test_prefix is not None:
        test_X, _, test_row_selector, test_ids = load_word_counts(
            input_dir, options.test_prefix, vocab=vocab
        )
        test_labels, _, _, _ = load_labels(
            input_dir, options.test_prefix, test_row_selector, options.labels
        )
        test_prior_covars, _, _, _ = load_covariates(
            input_dir,
            options.test_prefix,
            test_row_selector,
            options.prior_covars,
            covariate_selector=prior_covar_selector,
        )
        test_topic_covars, _, _, _ = load_covariates(
            input_dir,
            options.test_prefix,
            test_row_selector,
            options.topic_covars,
            covariate_selector=topic_covar_selector,
        )
        test_doc_reps = load_doc_reps(
            options.doc_reps_dir,
            prefix=options.test_prefix,
            row_selector=test_row_selector,
            use_sequences=options.attend_over_doc_reps,
        )
        n_test, _ = test_X.shape

    else:
        test_X = None
        n_test = 0
        test_labels = None
        test_prior_covars = None
        test_topic_covars = None
        test_doc_reps = None


    # collect label data for the deviations
    if options.deviation_embeddings:
        if not options.deviation_embedding_covar:
            raise ValueError("Need to supply a covariate for deviation embeddings")

        # hijack the unused `prior_covars` by storing deviation covar data in these objects
        # luckily all checks are performed on `n_prior_covars`, which is 0
        # but the data still get passed around
        # TODO: does this logic all already happen above?

        deviation_covar = options.deviation_embedding_covar
        train_prior_covars, _, deviation_covar_names, _ = load_labels(
            input_dir, options.train_prefix, train_row_selector, deviation_covar
        )
        if options.dev_prefix is not None:
            dev_prior_covars, _, _, _ = load_labels(
                input_dir, options.dev_prefix, dev_row_selector, deviation_covar
            )
        if options.test_prefix is not None:
            test_prior_covars, _, _, _ = load_labels(
                input_dir, options.test_prefix, test_row_selector, deviation_covar
            )
        # experimental baseline setting
        if options.ignore_deviation_embeddings:
            train_prior_covars = np.ones_like(train_prior_covars)
            dev_prior_covars = np.ones_like(dev_prior_covars)
            test_prior_covars = np.ones_like(test_prior_covars) 


    # initialize the background using overall word frequencies
    init_bg = get_init_bg(train_X)
    if options.no_bg:
        init_bg = np.zeros_like(init_bg)

    # combine the network configuration parameters into a dictionary
    network_architecture = make_network(
        options=options,
        vocab_size=vocab_size,
        label_type=label_type,
        n_labels=n_labels,
        n_prior_covars=n_prior_covars,
        n_topic_covars=n_topic_covars,
        doc_reps_dim=doc_reps_dim,
    )

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ":", val)

    # load word vectors
    embeddings = {}
    if options.background_embeddings:
        fpath = None if options.background_embeddings == 'random' else options.background_embeddings
        embeddings['background'] = load_word_vectors(
            fpath=fpath, # if None, they are randomly initialized
            emb_dim=options.emb_dim,
            update_embeddings=options.update_background_embeddings,
            rng=rng,
            vocab=vocab,
        )
    if options.deviation_embeddings:
        fpath = None if options.deviation_embeddings == 'random' else options.deviation_embeddings
        for name in deviation_covar_names:
            embeddings[name] = load_word_vectors(
                fpath=fpath,
                emb_dim=options.emb_dim,
                update_embeddings=options.update_deviation_embeddings,
                rng=rng,
                vocab=vocab,
            )

    # create the model
    if options.restart:
        print(f"Loading existing model from '{options.output_dir}'")
        model, _ = load_scholar_model(
            os.path.join(options.output_dir, "torch_model.pt"),
            embeddings=embeddings,
        )
        model.train()
    else:
        model = Scholar(
            network_architecture,
            alpha=options.alpha,
            learning_rate=options.learning_rate,
            init_embeddings=embeddings,
            init_bg=init_bg,
            adam_beta1=options.momentum,
            device=options.device,
            seed=seed,
            classify_from_covars=options.covars_predict,
            classify_from_topics=options.topics_predict,
            classify_from_doc_reps=options.classify_from_doc_reps,
        )

    if options.randomize_doc_reps:
        min_dr, max_dr = train_doc_reps.min(), train_doc_reps.max()
        train_doc_reps = np.random.uniform(min_dr, max_dr, size=train_doc_reps.shape)
        dev_doc_reps = np.random.uniform(min_dr, max_dr, size=dev_doc_reps.shape)
        if test_doc_reps is not None:
            test_doc_reps = np.random.uniform(min_dr, max_dr, size=test_doc_reps.shape)

    # make output directory
    fh.makedirs(options.output_dir)

    # train the model
    print("Optimizing full model")
    if options.epochs > 0:
        model = train(
            model=model,
            network_architecture=network_architecture,
            options=options,
            X=train_X,
            Y=train_labels,
            PC=train_prior_covars,
            TC=train_topic_covars,
            DR=train_doc_reps,
            vocab=vocab,
            prior_covar_names=prior_covar_names,
            topic_covar_names=topic_covar_names,
            training_epochs=options.epochs,
            batch_size=options.batch_size,
            patience=options.patience,
            eta_bn_anneal_step_const=options.eta_bn_anneal_step_const,
            dev_metric=options.dev_metric,
            rng=rng,
            X_dev=dev_X,
            Y_dev=dev_labels,
            PC_dev=dev_prior_covars,
            TC_dev=dev_topic_covars,
            DR_dev=dev_doc_reps,
        )

    # load best model
    model_fpath = os.path.join(options.output_dir, "torch_model.pt")
    if not options.save_at_training_end and os.path.exists(model_fpath):
        model, _ = load_scholar_model(model_fpath, embeddings)
        model.eval()
    elif options.epochs > 0:
        save_scholar_model(options, model, epoch=options.epochs, is_final=True)
        model.eval()
    # display and save weights
    print_and_save_weights(options, model, vocab, prior_covar_names, topic_covar_names)

    # Evaluate perplexity on dev and test data
    if dev_X is not None:
        perplexity = evaluate_perplexity(
            model,
            dev_X,
            dev_labels,
            dev_prior_covars,
            dev_topic_covars,
            dev_doc_reps,
            options.batch_size,
            eta_bn_prop=0.0,
        )
        print("Dev perplexity = %0.4f" % perplexity)
        fh.write_list_to_text(
            [str(perplexity)], os.path.join(options.output_dir, "perplexity.dev.txt")
        )

    if test_X is not None:
        perplexity = evaluate_perplexity(
            model,
            test_X,
            test_labels,
            test_prior_covars,
            test_topic_covars,
            test_doc_reps,
            options.batch_size,
            eta_bn_prop=0.0,
        )
        print("Test perplexity = %0.4f" % perplexity)
        fh.write_list_to_text(
            [str(perplexity)], os.path.join(options.output_dir, "perplexity.test.txt")
        )

    # evaluate accuracy on predicting labels
    if n_labels > 0:
        print("Predicting labels")
        predict_labels_and_evaluate(
            model,
            train_X,
            train_labels,
            train_prior_covars,
            train_topic_covars,
            train_doc_reps,
            options.output_dir,
            subset="train",
        )

        if dev_X is not None:
            predict_labels_and_evaluate(
                model,
                dev_X,
                dev_labels,
                dev_prior_covars,
                dev_topic_covars,
                dev_doc_reps,
                options.output_dir,
                subset="dev",
            )

        if test_X is not None:
            predict_labels_and_evaluate(
                model,
                test_X,
                test_labels,
                test_prior_covars,
                test_topic_covars,
                test_doc_reps,
                options.output_dir,
                subset="test",
            )

    # print label probabilities for each topic
    if n_labels > 0:
        print_topic_label_associations(
            options,
            label_names,
            model,
            n_prior_covars,
            n_topic_covars,
        )

    # save document representations
    print("Saving document representations")
    save_document_representations(
        model,
        train_X,
        train_labels,
        train_prior_covars,
        train_topic_covars,
        train_doc_reps,
        train_ids,
        options.output_dir,
        "train",
        batch_size=options.batch_size,
    )

    if dev_X is not None:
        save_document_representations(
            model,
            dev_X,
            dev_labels,
            dev_prior_covars,
            dev_topic_covars,
            dev_doc_reps,
            dev_ids,
            options.output_dir,
            "dev",
            batch_size=options.batch_size,
        )

    if n_test > 0:
        save_document_representations(
            model,
            test_X,
            test_labels,
            test_prior_covars,
            test_topic_covars,
            test_doc_reps,
            test_ids,
            options.output_dir,
            "test",
            batch_size=options.batch_size,
        )


def load_word_counts(input_dir, input_prefix, vocab=None):
    print("Loading data")
    # laod the word counts and convert to a dense matrix
    X = fh.load_sparse(os.path.join(input_dir, input_prefix + ".npz"))
    X = X.astype(np.float32)
    # load the vocabulary
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + ".vocab.json"))
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    ids = fh.read_json(os.path.join(input_dir, input_prefix + ".ids.json"))

    # filter out empty documents and return a boolean selector for filtering labels and covariates
    row_selector = np.array(X.sum(axis=1) > 0, dtype=bool).reshape(-1)
    print("Found %d non-empty documents" % np.sum(row_selector))
    X = X[row_selector, :]
    ids = [doc_id for i, doc_id in enumerate(ids) if row_selector[i]]

    return X, vocab, row_selector, ids


def load_labels(input_dir, input_prefix, row_selector, labels=None):
    label_type = None
    label_names = None
    n_labels = 0
    # load the label file if given
    if labels is not None:
        label_file = os.path.join(
            input_dir, input_prefix + "." + labels + ".csv"
        )
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            # select the rows that match the non-empty documents (from load_word_counts)
            labels = labels[row_selector, :]
            n, n_labels = labels.shape
            print("Found %d labels" % n_labels)
        else:
            raise (FileNotFoundError("Label file {:s} not found".format(label_file)))

    return labels, label_type, label_names, n_labels


def load_covariates(
    input_dir,
    input_prefix,
    row_selector,
    covars_to_load,
    min_count=None,
    covariate_selector=None,
):

    covariates = None
    covariate_names = None
    n_covariates = 0
    if covars_to_load is not None:
        covariate_list = []
        covariate_names_list = []
        covar_file_names = covars_to_load.split(",")
        # split the given covariate names by commas, and load each one
        for covar_file_name in covar_file_names:
            covariates_file = os.path.join(
                input_dir, input_prefix + "." + covar_file_name + ".csv"
            )
            if os.path.exists(covariates_file):
                print("Loading covariates from", covariates_file)
                temp = pd.read_csv(covariates_file, header=0, index_col=0)
                covariate_names = covar_file_name + '_' + temp.columns
                covariates = np.array(temp.values, dtype=np.float32)
                # select the rows that match the non-empty documents (from load_word_counts)
                covariates = covariates[row_selector, :]
                covariate_list.append(covariates)
                covariate_names_list.extend(covariate_names)

            else:
                raise (
                    FileNotFoundError(
                        "Covariates file {:s} not found".format(covariates_file)
                    )
                )

        # combine the separate covariates into a single matrix
        covariates = np.hstack(covariate_list)
        covariate_names = covariate_names_list

        _, n_covariates = covariates.shape

        # if a covariate_selector has been given (from a previous call of load_covariates), drop columns
        if covariate_selector is not None:
            covariates = covariates[:, covariate_selector]
            covariate_names = [
                name for i, name in enumerate(covariate_names) if covariate_selector[i]
            ]
            n_covariates = len(covariate_names)
        # otherwise, choose which columns to drop based on how common they are (for binary covariates)
        elif min_count is not None and int(min_count) > 0:
            print("Removing rare covariates")
            covar_sums = covariates.sum(axis=0).reshape((n_covariates,))
            covariate_selector = covar_sums > int(min_count)
            covariates = covariates[:, covariate_selector]
            covariate_names = [
                name for i, name in enumerate(covariate_names) if covariate_selector[i]
            ]
            n_covariates = len(covariate_names)

    return covariates, covariate_selector, covariate_names, n_covariates


def load_doc_reps(input_dir, prefix, row_selector, use_sequences=False):
    """
    Load document representations, an [num_docs x doc_dim] matrix
    """
    if input_dir is not None:
        doc_rep_fpath = os.path.join(input_dir, f"{prefix}.npy")
        doc_reps = np.load(doc_rep_fpath)
        if not use_sequences:
            return doc_reps[row_selector, :]
        
        tokens_fpath = os.path.join(input_dir, f"{prefix}.tokens.npy")
        tokens = np.load(tokens_fpath)[:, :, None]
        mask = tokens > 0
        doc_reps = np.insert(doc_reps, [0], mask, axis=2)
        return doc_reps[row_selector, :]

def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[
                n_dev * options.dev_fold : n_dev * (options.dev_fold + 1)
            ]
        else:
            dev_indices = indices[n_dev * options.dev_fold :]
        train_indices = list(set(indices) - set(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def get_init_bg(data):
    # Compute the log background frequency of all words
    sums = np.sum(data, axis=0) + 1
    print("Computing background frequencies")
    print(
        "Min/max word counts in training data: %d %d"
        % (int(np.min(sums)), int(np.max(sums)))
    )
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg.reshape(-1)


def load_word_vectors(fpath, emb_dim, update_embeddings, rng, vocab):
    
    # load word2vec vectors if given
    if fpath is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        # randomly initialize word vectors for each term in the vocabualry
        embeddings = np.array(
            rng.rand(emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32
        )
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        if fpath.endswith('.model'):
            pretrained = gensim.models.Word2Vec.load(fpath)
        else:
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(
                fpath, binary=fpath.endswith('.bin')
            )

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
    else:
        
        update_embeddings = True # always true if unspecified
        embeddings = None
    
    return embeddings, update_embeddings


def make_network(
    options,
    vocab_size,
    doc_reps_dim=None,
    label_type=None,
    n_labels=0,
    n_prior_covars=0,
    n_topic_covars=0,
):
    # Assemble the network configuration parameters into a dictionary
    network_architecture = dict(
        embedding_dim=options.emb_dim,
        zero_out_embeddings=options.zero_out_embeddings,
        reconstruct_bow=options.reconstruct_bow,
        doc_reps_dim=doc_reps_dim,
        attend_over_doc_reps=options.attend_over_doc_reps,
        use_doc_layer=options.use_doc_layer,
        doc_reconstruction_weight=options.doc_reconstruction_weight,
        doc_reconstruction_temp=options.doc_reconstruction_temp,
        doc_reconstruction_min_count=options.doc_reconstruction_min_count,
        n_topics=options.n_topics,
        vocab_size=vocab_size,
        label_type=label_type,
        n_labels=n_labels,
        n_prior_covars=n_prior_covars,
        n_topic_covars=n_topic_covars,
        l1_beta_reg=options.l1_topics,
        l1_beta_c_reg=options.l1_topic_covars,
        l1_beta_ci_reg=options.l1_interactions,
        l2_prior_reg=options.l2_prior_covars,
        classifier_layers=1,
        classifier_loss_weight=options.classifier_loss_weight,
        use_interactions=options.interactions,
    )
    return network_architecture


def train(
    model,
    network_architecture,
    options,
    X,
    Y,
    PC,
    TC,
    DR,
    vocab,
    prior_covar_names,
    topic_covar_names,
    batch_size=200,
    training_epochs=100,
    patience=10,
    dev_metric="perplexity",
    display_step=1,
    X_dev=None,
    Y_dev=None,
    PC_dev=None,
    TC_dev=None,
    DR_dev=None,
    bn_anneal=True,
    init_eta_bn_prop=1.0,
    eta_bn_anneal_step_const=0.75,
    rng=None,
    min_weights_sq=1e-7,
):
    # Train the model
    n_train, vocab_size = X.shape
    mb_gen = create_minibatch(X, Y, PC, TC, DR, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)
    batches = 0

    best_dev_metrics = None

    num_epochs_no_improvement = 0

    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

    model.train()

    n_topics = network_architecture["n_topics"]
    n_topic_covars = network_architecture["n_topic_covars"]
    vocab_size = network_architecture["vocab_size"]

    # create matrices to track the current estimates of the priors on the individual weights
    if network_architecture["l1_beta_reg"] > 0:
        l1_beta = (
            0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
        )
    else:
        l1_beta = None

    if (
        network_architecture["l1_beta_c_reg"] > 0
        and network_architecture["n_topic_covars"] > 0
    ):
        l1_beta_c = (
            0.5
            * np.ones([vocab_size, n_topic_covars], dtype=np.float32)
            / float(n_train)
        )
    else:
        l1_beta_c = None

    if (
        network_architecture["l1_beta_ci_reg"] > 0
        and network_architecture["n_topic_covars"] > 0
        and network_architecture["use_interactions"]
    ):
        l1_beta_ci = (
            0.5
            * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32)
            / float(n_train)
        )
    else:
        l1_beta_ci = None

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        accuracy = 0.0
        avg_nl = 0.0
        avg_kld = 0.0
        # Loop over all batches
        for i in tqdm(range(total_batch), disable=True):
            # get a minibatch
            batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs = next(mb_gen)
            # do one minibatch update
            cost, recon_y, thetas, nl, kld = model.fit(
                batch_xs,
                batch_ys,
                batch_pcs,
                batch_tcs,
                batch_drs,
                eta_bn_prop=eta_bn_prop,
                l1_beta=l1_beta,
                l1_beta_c=l1_beta_c,
                l1_beta_ci=l1_beta_ci,
            )

            # compute accuracy on minibatch
            if network_architecture["n_labels"] > 0:
                accuracy += np.sum(
                    np.argmax(recon_y, axis=1) == np.argmax(batch_ys, axis=1)
                ) / float(n_train)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            avg_nl += float(nl) / n_train * batch_size
            avg_kld += float(kld) / n_train * batch_size
            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print(
                    "Encountered NaN, stopping training. Please check the learning_rate settings and the momentum."
                )
                sys.exit()

        # if we're using regularization, update the priors on the individual weights
        if network_architecture["l1_beta_reg"] > 0:
            weights = model.get_weights().T
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta = 0.5 / weights_sq / float(n_train)

        if (
            network_architecture["l1_beta_c_reg"] > 0
            and network_architecture["n_topic_covars"] > 0
        ):
            weights = model.get_covar_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_c = 0.5 / weights_sq / float(n_train)

        if (
            network_architecture["l1_beta_ci_reg"] > 0
            and network_architecture["n_topic_covars"] > 0
            and network_architecture["use_interactions"]
        ):
            weights = model.get_covar_interaction_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_ci = 0.5 / weights_sq / float(n_train)

        # Display logs per epoch step
        if epoch % display_step == 0 and epoch >= 0:
            epoch_metrics = {}
            if network_architecture["n_labels"] > 0:
                print(
                    "Epoch:",
                    "%d" % epoch,
                    "; cost =",
                    "{:.9f}".format(avg_cost),
                    "; training accuracy (noisy) =",
                    "{:.9f}".format(accuracy),
                )
            else:
                print("Epoch:", "%d" % epoch, "cost=", "{:.9f}".format(avg_cost))

            if X_dev is not None:
                # switch to eval mode for intermediate evaluation
                model.eval()

                # perplexity
                dev_perplexity = 0.0
                dev_perplexity = evaluate_perplexity(
                    model,
                    X_dev,
                    Y_dev,
                    PC_dev,
                    TC_dev,
                    DR_dev,
                    batch_size,
                    eta_bn_prop=eta_bn_prop,
                )
                n_dev, _ = X_dev.shape
                epoch_metrics["perplexity"] = dev_perplexity

                # accuracy
                dev_accuracy = 0
                if network_architecture["n_labels"] > 0:
                    dev_pred_probs = predict_label_probs(
                        model, X_dev, PC_dev, TC_dev, DR_dev, eta_bn_prop=eta_bn_prop
                    )
                    dev_predictions = np.argmax(dev_pred_probs, axis=1)
                    dev_accuracy = float(
                        np.sum(dev_predictions == np.argmax(Y_dev, axis=1))
                    ) / float(n_dev)
                    epoch_metrics["accuracy"] = dev_accuracy

                # NPMI
                dev_npmi = compute_npmi_at_n_during_training(
                    model.get_weights(), ref_counts=X_dev.tocsc(), n=options.npmi_words, smoothing=0.,
                )
                epoch_metrics["npmi"] = dev_npmi

                print(
                    f"Dev perplexity = {dev_perplexity:0.4f}; "
                    f"Dev accuracy = {dev_accuracy:0.4f}; "
                    f"Dev NPMI = {dev_npmi:0.4f}"
                )

                best_dev_metrics = update_metrics(epoch_metrics, best_dev_metrics, epoch)
                if best_dev_metrics[dev_metric]["epoch"] == epoch:
                    num_epochs_no_improvement = 0
                    save_scholar_model(options, model, epoch, best_dev_metrics)
                else:
                    num_epochs_no_improvement += 1
                if patience is not None and num_epochs_no_improvement >= patience:
                    print(f"Ran out of patience ({patience} epochs), returning model")
                    return model
                # switch back to training mode
                model.train()

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(eta_bn_anneal_step_const * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

    # finish tr
    model.eval()
    return model


def create_minibatch(X, Y, PC, TC, DR, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = X[ixs, :].astype("float32")
        X_mb = X_mb.todense()
        if Y is not None:
            Y_mb = Y[ixs, :].astype("float32")
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype("float32")
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype("float32")
        else:
            TC_mb = None

        if DR is not None:
            DR_mb = DR[ixs, :].astype("float32")
        else:
            DR_mb = None


        yield X_mb, Y_mb, PC_mb, TC_mb, DR_mb


def get_minibatch(X, Y, PC, TC, DR, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    X_mb = X[ixs, :].astype("float32")
    X_mb = X_mb.todense()

    if Y is not None:
        Y_mb = Y[ixs, :].astype("float32")
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype("float32")
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype("float32")
    else:
        TC_mb = None

    if DR is not None:
       DR_mb = DR[ixs, :].astype("float32")
    else:
        DR_mb = None    

    return X_mb, Y_mb, PC_mb, TC_mb, DR_mb


def update_metrics(current, best=None, epoch=None):
    """
    Update the best metrics with the current metrics if they have improved
    """
    if best is None:
        best = {
            "perplexity": {"value": np.inf},
            "accuracy": {"value": -np.inf},
            "npmi": {"value": -np.inf},
        }

    for metric in current:
        sign = -1 if metric == "perplexity" else +1

        if sign * current[metric] > sign * best[metric]["value"]:
            best[metric] = {
                "value": current[metric],
                "epoch": epoch
            }
    
    return best


def predict_label_probs(model, X, PC, TC, DR, batch_size=200, eta_bn_prop=0.0):
    # Predict a probability distribution over labels for each instance using the classifier part of the network

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    pred_probs_all = []

    # make predictions on minibatches and then combine
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs = get_minibatch(
            X, None, PC, TC, DR, i, batch_size
        )
        Z, pred_probs = model.predict(
            batch_xs, batch_pcs, batch_tcs, batch_drs, eta_bn_prop=eta_bn_prop
        )
        pred_probs_all.append(pred_probs)

    pred_probs = np.vstack(pred_probs_all)

    return pred_probs


def print_and_save_weights(
    options, model, vocab, prior_covar_names=None, topic_covar_names=None
):

    # print background
    bg = model.get_bg()
    if not options.no_bg:
        print_top_bg(bg, vocab)

    # print topics
    emb = model.get_weights()
    print("Topics:")
    maw, sparsity = print_top_words(emb, vocab)
    print("sparsity in topics = %0.4f" % sparsity)
    save_weights(options.output_dir, emb, bg, vocab, sparsity_threshold=1e-5)

    fh.write_list_to_text(
        ["{:.4f}".format(maw)], os.path.join(options.output_dir, "maw.txt")
    )
    fh.write_list_to_text(
        ["{:.4f}".format(sparsity)], os.path.join(options.output_dir, "sparsity.txt")
    )

    if prior_covar_names is not None:
        prior_weights = model.get_prior_weights()
        print("Topic prior associations:")
        print("Covariates:", " ".join(prior_covar_names))
        for k in range(options.n_topics):
            output = str(k) + ": "
            for c in range(len(prior_covar_names)):
                output += "%.4f " % prior_weights[c, k]
            print(output)
        if options.output_dir is not None:
            np.savez(
                os.path.join(options.output_dir, "prior_w.npz"),
                weights=prior_weights,
                names=prior_covar_names,
            )

    if topic_covar_names is not None:
        beta_c = model.get_covar_weights()
        print("Covariate deviations:")
        maw, sparsity = print_top_words(beta_c, vocab, topic_covar_names)
        print("sparsity in covariates = %0.4f" % sparsity)
        if options.output_dir is not None:
            np.savez(
                os.path.join(options.output_dir, "beta_c.npz"),
                beta=beta_c,
                names=topic_covar_names,
            )

        if options.interactions:
            print("Covariate interactions")
            beta_ci = model.get_covar_interaction_weights()
            print(beta_ci.shape)
            if topic_covar_names is not None:
                names = [
                    str(k) + ":" + c
                    for k in range(options.n_topics)
                    for c in topic_covar_names
                ]
            else:
                names = None
            maw, sparsity = print_top_words(beta_ci, vocab, names)
            if options.output_dir is not None:
                np.savez(
                    os.path.join(options.output_dir, "beta_ci.npz"),
                    beta=beta_ci,
                    names=names,
                )
            print("sparsity in covariate interactions = %0.4f" % sparsity)


def print_top_words(
    beta,
    feature_names,
    topic_names=None,
    n_pos=8,
    n_neg=8,
    sparsity_threshold=1e-5,
    values=False,
):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ""
        # get the top words
        for j in range(n_pos):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + " "
                if values:
                    output += "(" + str(beta[i][order[j]]) + ") "

        order.reverse()
        if n_neg > 0:
            output += " / "
        # get the bottom words
        for j in range(n_neg):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + " "
                if values:
                    output += "(" + str(beta[i][order[j]]) + ") "

        # compute sparsity
        sparsity = float(
            np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i]))
        )
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += "; sparsity=%0.4f" % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ": " + output
        else:
            output = str(i) + ": " + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)


def print_top_bg(bg, feature_names, n_top_words=10):
    # Print the most highly weighted words in the background log frequency
    print("Background frequencies of top words:")
    print(" ".join([feature_names[j] for j in bg.argsort()[: -n_top_words - 1 : -1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[: -n_top_words - 1 : -1]))


def evaluate_perplexity(model, X, Y, PC, TC, DR, batch_size, eta_bn_prop=0.0):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    doc_sums = np.array(X.sum(axis=1), dtype=np.float32).reshape(-1)
    X = X.astype("float32")
    if Y is not None:
        Y = Y.astype("float32")
    if PC is not None:
        PC = PC.astype("float32")
    if TC is not None:
        TC = TC.astype("float32")
    if DR is not None:
        DR = DR.astype("float32")
    losses = []

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs = get_minibatch(
            X, Y, PC, TC, DR, i, batch_size
        )
        batch_losses = model.get_losses(
            batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs, eta_bn_prop=eta_bn_prop
        )
        losses.append(batch_losses)
    losses = np.hstack(losses)
    perplexity = np.exp(np.mean(losses / doc_sums))
    return perplexity


def save_weights(output_dir, beta, bg, feature_names, sparsity_threshold=1e-5):
    # Save model weights to npz files (also the top words in each topic
    np.savez(os.path.join(output_dir, "beta.npz"), beta=beta)
    if bg is not None:
        np.savez(os.path.join(output_dir, "bg.npz"), bg=bg)
    fh.write_to_json(
        feature_names, os.path.join(output_dir, "vocab.json"), sort_keys=False
    )

    topics_file = os.path.join(output_dir, "topics.txt")
    lines = generate_topics(
        beta, feature_names, n=100, sparsity_threshold=sparsity_threshold
    )

    fh.write_list_to_text(lines, topics_file)

def generate_topics(beta, feature_names, n=100, sparsity_threshold=1e-5):
    """
    Create topics from the beta parameter
    """
    lines = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words = [
            feature_names[j] for j in order[:n] if beta[i][j] > sparsity_threshold
        ]
        output = " ".join(pos_words)
        lines.append(output)
    
    return lines

def predict_labels_and_evaluate(
    model, X, Y, PC, TC, DR, output_dir=None, subset="train", batch_size=200
):
    # Predict labels for all instances using the classifier network and evaluate the accuracy
    pred_probs = predict_label_probs(model, X, PC, TC, DR, batch_size, eta_bn_prop=0.0)
    np.savez(
        os.path.join(output_dir, "pred_probs." + subset + ".npz"), pred_probs=pred_probs
    )
    predictions = np.argmax(pred_probs, axis=1)
    accuracy = float(np.sum(predictions == np.argmax(Y, axis=1)) / float(len(Y)))
    print(subset, "accuracy on labels = %0.4f" % accuracy)
    if output_dir is not None:
        fh.write_list_to_text(
            [str(accuracy)], os.path.join(output_dir, "accuracy." + subset + ".txt")
        )


def print_topic_label_associations(
    options, label_names, model, n_prior_covars, n_topic_covars
):
    # Print associations between topics and labels
    if options.n_labels > 0 and options.n_labels < 7:
        print("Label probabilities based on topics")
        print("Labels:", " ".join([name for name in label_names]))
    probs_list = []
    for k in range(options.n_topics):
        Z = np.zeros([1, options.n_topics]).astype("float32")
        Z[0, k] = 1.0
        Y = None
        if n_prior_covars > 0:
            PC = np.zeros([1, n_prior_covars]).astype("float32")
        else:
            PC = None
        if n_topic_covars > 0:
            TC = np.zeros([1, n_topic_covars]).astype("float32")
        else:
            TC = None

        probs = model.predict_from_topics(Z, PC, TC)
        probs_list.append(probs)
        if options.n_labels > 0 and options.n_labels < 7:
            output = str(k) + ": "
            for i in range(options.n_labels):
                output += "%.4f " % probs[0, i]
            print(output)

    probs = np.vstack(probs_list)
    np.savez(
        os.path.join(options.output_dir, "topics_to_labels.npz"),
        probs=probs,
        label=label_names,
    )


def save_document_representations(
    model, X, Y, PC, TC, DR, ids, output_dir, partition, batch_size=200
):
    # compute the mean of the posterior of the latent representation for each documetn and save it
    if Y is not None:
        Y = np.zeros_like(Y)

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    thetas = []

    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs = get_minibatch(
            X, Y, PC, TC, DR, i, batch_size
        )
        thetas.append(
            model.compute_theta(batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs)
        )
    theta = np.vstack(thetas)

    np.savez(
        os.path.join(output_dir, "theta." + partition + ".npz"), theta=theta, ids=ids
    )


def save_scholar_model(options, model, epoch=0, dev_metrics={}, is_final=False):
    """
    Save the Scholar model
    """

    # get git info
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        sha = None
    fname = 'torch_model.pt' if not is_final else 'torch_model_final.pt'
    torch.save(
        {
            # Scholar arguments
            "scholar_kwargs": model._call,
            # Embeddings path
            "options": options,
            # Torch weights
            "model_state_dict": model._model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
            # Training state
            "epoch": epoch,
            "dev_metrics": dev_metrics,
            "git_hash": sha,
        },
        os.path.join(options.output_dir, fname),
    )


def load_scholar_model(inpath, embeddings=None, map_location=None):
    """
    Load the Scholar model
    """
    checkpoint = torch.load(inpath, map_location=map_location)
    scholar_kwargs = checkpoint["scholar_kwargs"]
    scholar_kwargs["init_embeddings"] = embeddings

    model = Scholar(**scholar_kwargs)
    model._model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint


if __name__ == "__main__":
    main()

