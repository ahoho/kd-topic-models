import argparse
import json
import random
import logging
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

def set_seed(args):
    """
    Taken from 
    github.com/huggingface/transformers/blob/master/examples/mm-imdb/run_mmimdb.py
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


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
    with open(fpath, "r") as infile:
        return json.load(infile)

def save_json(obj, fpath):
    with open(fpath, "w") as outfile:
        return json.dump(obj, outfile)

class DocDataset(Dataset):
    """
    Mildly adapted from 
    github.com/huggingface/transformers/blob/master/examples/mm-imdb/utils_mmimdb.py
    """
    def __init__(self, data, tokenizer, word_counts, max_seq_length=None):
        """
        Initialize with both the "raw" input IMDB data and the processed count matrix
        Args:
          data: list of sentences
          tokenizer: a tokenizer from the transformers liberary
          word_counts: a sparse numpy matrix
          max_seq_length: maximum length of the sequence
        """
        self.examples = tokenizer.batch_encode_plus(
            data,
            add_special_tokens=True,
            max_length=tokenizer.max_len,
        )

        self.tokenizer = tokenizer
        self.word_counts = word_counts
        self.n_classes = word_counts.shape[1]
        self.max_seq_length = max_seq_length or tokenizer.max_len
        
        assert(len(self.examples["input_ids"]) == word_counts.shape[0])

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, i):
        input_ids = torch.tensor(self.examples["input_ids"][i], dtype=torch.long)
        attention_mask = torch.tensor(self.examples["attention_mask"][i], dtype=torch.int)
        word_counts = torch.tensor(self.word_counts[i].todense().astype(np.int32))[0]
        return input_ids, attention_mask, word_counts

    def collate(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [t for t, _, _ in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [m for _, m, _ in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        word_counts = torch.stack([c for _, _, c in batch])
        return input_ids, attention_masks, word_counts

class BertForDocReconstruction(BertForSequenceClassification):
    def __init__(self, config, softmax_temp=1):
        super().__init__(config)
        self.softmax_temp = softmax_temp
    
    def forward(self, *args, **kwargs):
        """
        Identical signature to the parent class
        """
        labels = kwargs.pop("labels")
        outputs = super().forward(*args, **kwargs, labels=None)
        probs = torch.softmax(outputs[0] / self.softmax_temp, dim=1)
        outputs = (probs,) + outputs

        if labels is not None:
            loss = -(labels * (probs + 1e-10).log()).sum(1)
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)


def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """
    Train the model

    Borrows heavily from
    https://github.com/huggingface/transformers/blob/master/examples/mm-imdb/run_mmimdb.py
    """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
    )

    t_total = (
        len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train
    logger.info("***** Running training *****")
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_ppl, n_no_improve = np.inf, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            input_ids, attention_mask, word_counts = [b.to(args.device) for b in batch]
            loss, probs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=word_counts,
            )
            
            loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
        
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
                optimizer.step()
                scheduler.step() # update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if eval_dataset is not None and args.evaluate_during_training:           
                        results = evaluate(args, eval_dataset, model, tokenizer)

                        logger.info(f"Evaluation results: ")
                        with open(Path(args.output_dir, "eval_results.txt"), "a") as writer:
                            for k, v in results.items():
                                logger.info(f"{k}: {v:0.4f}")
                                writer.write(f"{k}: {v:0.4f}\n")
                                logs[f"eval_{k}"] = v

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    #print(json.dumps({**logs, **{"step": global_step}}))
                    #for k, v in {**logs, **{"step": global_step}}.items():


                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = Path(args.output_dir, f"checkpoint-{global_step}")
                    if not output_dir.exists()  :
                        output_dir.mkdir(parents=True)
                    
                    torch.save(model.state_dict(), Path(output_dir, WEIGHTS_NAME))
                    torch.save(args, Path(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
        
        if eval_dataset is not None:
            results = evaluate(args, eval_dataset, model, tokenizer)
            if results["perplexity"] < best_ppl:
                best_ppl = results["perplexity"]
                n_no_improve = 0
            else:
                n_no_improve += 1

            if n_no_improve > args.patience:
                train_iterator.close()
                break

    return global_step, tr_loss / global_step

def evaluate(
    args,
    eval_dataset,
    model,
    tokenizer,
    return_probs=False,
    return_logits=False,
    return_pooled_layer=False,
):
    """
    Evaluate the model on held-out data, also borrwed a lot from 
    github.com/huggingface/transformers/blob/master/examples/mm-imdb/run_mmimdb.py
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size,
        collate_fn=eval_dataset.collate
    )

    logger.info("***** Running evaluation *****")
    doc_sums = np.array(eval_dataset.word_counts.sum(axis=1), dtype=np.float32).reshape(-1)
    eval_loss = []
    eval_probs = []
    eval_logits = []
    eval_hidden = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            input_ids, attention_mask, word_counts = [b.to(args.device) for b in batch]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=word_counts,
            )

            eval_loss.append(outputs[0].to("cpu").detach().numpy())
            if return_probs:
                eval_probs.append(outputs[1].to("cpu").detach().numpy())
            if return_logits:
                eval_logits.append(outputs[2].to("cpu").detach().numpy())
            if return_pooled_layer:
                layer = outputs[-1][-1] # just the last layer for now
                if return_pooled_layer == 'cls':
                    eval_hidden.append(layer[:, 0, :].to("cpu").detach().numpy())
                if return_pooled_layer == 'mean':
                    eval_hidden.append(layer.mean(1).to("cpu").detach().numpy())
                if return_pooled_layer == 'seq':
                    eval_hidden.append(layer.to("cpu").detach().numpy())
        
    eval_loss = np.hstack(eval_loss)
    results = {
        "perplexity": float(np.exp(np.mean(eval_loss / doc_sums)))
    }
    
    if return_probs:
        results['probs'] = np.vstack(eval_probs)
    if return_logits:
        results['logits'] = np.vstack(eval_logits)
    if return_pooled_layer:
        results['hidden'] = np.vstack(eval_hidden)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        help="Directory of processed data. All data must be the same size/order as the jsonlist file!",
    )
    # Train data
    parser.add_argument("--vocab-fname", default="train.vocab.json")
    parser.add_argument(
        "--train-text-fname",
        default="train.jsonlist",
        help="The input text path. Should be the .jsonlist file",
    )
    parser.add_argument("--train-counts-fname", default="train.npz")
    parser.add_argument("--train-ids-fname", default="train.ids.json")

    parser.add_argument("--softmax-temp", default=1.0, type=float, help="Output softmax temperature")

    # Dev data -- EITHER use dev-split OR already split data
    parser.add_argument("--no-dev", action="store_true", default=False)
    parser.add_argument("--dev-text-fname", default="dev.jsonlist")
    parser.add_argument("--dev-counts-fname", default="dev.npz")
    parser.add_argument("--dev-ids-fname", default="dev.ids.json")

    parser.add_argument("--dev-split", type=float, default=0.0, help="Size of dev split")

    parser.add_argument(
        "--bert-model",
        default="bert-base-uncased",
        help="BERT model to be used"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for model predictions and checkpoints"
    )

    # get out representations
    parser.add_argument(
        "--get-reps", action="store_true", help="Get out document representations."
    )
    parser.add_argument(
        "--checkpoint-folder-pattern",
        help="Load checkpoints with this glob pattern"
    )
    parser.add_argument(
        "--save-doc-probs",
        action="store_true",
        help="Save the estimated probability."
    )
    parser.add_argument(
        "--save-doc-logits",
        action="store_true",
        help="Save the unnormalized logits"
    )
    parser.add_argument(
        "--save-pooled-hidden-layer",
        choices=['mean', 'cls', 'seq'],
        help="Save a hidden BERT layer"
    )

    # other arguments
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max-seq-length",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do-train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do-eval", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--evaluate-during-training",
        action="store_true",
        help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning-rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num-train-epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Training (and evaluation) batch size"
    )
    parser.add_argument(
        "--patience", default=5, type=int, help="Patience for Early Stopping."
    )

    parser.add_argument(
        "--warmup-steps",
        default=0,
        type=int,
        help="Linear warmup over warmup-steps."
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of worker threads for dataloading"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()

    # reload the original arguments if using a previous model
    if args.get_reps and Path(args.output_dir, "training_args.bin").exists():
        original_args = torch.load(Path(args.output_dir, "training_args.bin"))

        # only carry over the reloading arguments
        original_args.do_train = False
        original_args.get_reps = True
        original_args.output_dir = args.output_dir
        original_args.checkpoint_folder_pattern = args.checkpoint_folder_pattern
        original_args.save_doc_probs = args.save_doc_probs
        original_args.save_doc_logits = args.save_doc_logits
        original_args.save_pooled_hidden_layer = args.save_pooled_hidden_layer
        original_args.batch_size = args.batch_size
        original_args.no_dev = args.no_dev

        args = original_args    

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )    

    # set seed
    set_seed(args)

    # load in the data, eliminating any empty documents
    vocab = load_json(Path(args.input_dir, args.vocab_fname))
    data = [
        json.loads(l)["text"]  for l in open(Path(args.input_dir, args.train_text_fname))
    ]
    word_counts = load_sparse(Path(args.input_dir, args.train_counts_fname))
    ids = load_json(Path(args.input_dir, args.train_ids_fname))
    
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, cache_dir=args.cache_dir
    )
    
    # create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initialize the model and tokenizer
    # TODO: add support for other models
    # split up into train/dev if desired
    logger.info(f"Loading and tokenizing data")
    if not args.no_dev and args.dev_split > 0:
        from sklearn.model_selection import train_test_split
        args.dev_text_fname = None # do not load any dev data
        train_ids, dev_ids, train_data, dev_data, train_word_counts, dev_word_counts = (
            train_test_split(
                ids, data, word_counts, test_size=args.dev_split
            )
        )
        
        train_dataset = DocDataset(train_data, tokenizer, train_word_counts)
        dev_dataset = DocDataset(dev_data, tokenizer, dev_word_counts)
        
        # save the splits for posterity
        torch.save(train_dataset.examples, Path(args.output_dir, "train.data.pt"))
        torch.save(dev_dataset.examples, Path(args.output_dir, "dev.data.pt"))
        save_sparse(train_word_counts, Path(args.output_dir, "train.npz"))
        save_sparse(dev_word_counts, Path(args.output_dir, "dev.npz"))
        save_json(train_ids, Path(args.output_dir, "train.ids.json"))
        save_json(dev_ids, Path(args.output_dir, "dev.ids.json"))

        save_json(vocab, Path(args.output_dir, "train.vocab.json"))
    if not args.no_dev and args.dev_text_fname is not None:
        dev_data = [
            json.loads(l)["text"]  for l in open(Path(args.input_dir, args.dev_text_fname))
        ]
        dev_word_counts = load_sparse(Path(args.input_dir, args.dev_counts_fname))
        dev_ids = load_json(Path(args.input_dir, args.dev_ids_fname))

        train_dataset = DocDataset(data, tokenizer, word_counts)
        dev_dataset = DocDataset(dev_data, tokenizer, dev_word_counts)
    if args.no_dev:
        train_dataset = DocDataset(data, tokenizer, word_counts)
        dev_dataset = None

    # train!
    if args.do_train:
        model = BertForDocReconstruction.from_pretrained(
            args.bert_model,
            num_labels=word_counts.shape[1],
            softmax_temp=args.softmax_temp,
            cache_dir=args.cache_dir,
        )
        model.to(args.device)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, dev_dataset)
        logger.info(f"Global step: {global_step}, average_loss: {tr_loss:0.4f}")

        logger.info(f"Saving checkpoints to {args.output_dir}")

        torch.save(model.state_dict(), Path(args.output_dir, WEIGHTS_NAME))
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, Path(args.output_dir, "training_args.bin"))

    if args.get_reps:
        config = BertConfig.from_pretrained(
            args.bert_model,
            num_labels=word_counts.shape[1],
            output_hidden_states=True
        )
        
        checkpoint_dirs = list(Path(args.output_dir).glob(args.checkpoint_folder_pattern))
        for dir in tqdm(checkpoint_dirs, desc="Checkpoints"):
            model = BertForDocReconstruction(config, softmax_temp=args.softmax_temp)
            model.to(args.device)
            model.load_state_dict(torch.load(Path(dir, "pytorch_model.bin")))
            train_results = evaluate(
                args,
                train_dataset,
                model,
                tokenizer,
                return_probs=args.save_doc_probs,
                return_logits=args.save_doc_logits,
                return_pooled_layer=args.save_pooled_hidden_layer,
            )
            dev_results = None
            if dev_dataset is not None:
                dev_results = evaluate(
                    args,
                    dev_dataset,
                    model,
                    tokenizer,
                    return_probs=args.save_doc_probs,
                    return_logits=args.save_doc_logits,
                    return_pooled_layer=args.save_pooled_hidden_layer,
                )
            
            if args.save_doc_probs:
                subdir = Path(dir, "doc_probs")
                subdir.mkdir(exist_ok=True)
                np.save(Path(subdir, "train.npy"), train_results['probs'])
                np.save(
                    Path(subdir, "train.pcounts.npy"), # save pseudo-counts as well
                    train_results['probs'] * np.array(train_dataset.word_counts.sum(1))
                )
                if dev_results is not None:
                    np.save(Path(subdir, "dev.npy"), dev_results['probs'])
                    np.save(
                        Path(subdir, "dev.pcounts.npy"),
                        dev_results['probs'] * np.array(dev_dataset.word_counts.sum(1))
                    )
            
            if args.save_doc_logits:
                subdir = Path(dir, "doc_logits")
                subdir.mkdir(exist_ok=True)
                np.save(Path(subdir, "train.npy"), train_results['logits'])
                if dev_results is not None:
                    np.save(Path(subdir, "dev.npy"), dev_results['logits'])
            
            if args.save_pooled_hidden_layer:
                subdir = Path(dir, f"doc_hidden_{args.save_pooled_hidden_layer}")
                subdir.mkdir(exist_ok=True)
                np.save(Path(subdir, f"train.npy"), train_results['hidden'])
                if dev_results is not None:
                    np.save(Path(subdir, f"dev.npy"), dev_results['hidden'])