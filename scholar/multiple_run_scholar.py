import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from run_scholar import main
import file_handling as fh
from compute_npmi import compute_npmi_at_n

def tu(topics, l=10):
    """
    Topic uniqueness measure from https://www.aclweb.org/anthology/P19-1640.pdf
    """
    tu_results = []
    for topics_i in topics:
        w_counts = 0
        for w in topics_i[:l]:
            w_counts += 1 / np.sum([w in topics_j[:l] for topics_j in topics]) # count(k, l)
        tu_results.append((1 / l) * w_counts)
    return tu_results

if __name__ == "__main__":
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument("--runs", default=1, type=int)
    run_parser.add_argument("--global-seed", type=int)
    run_parser.add_argument("--store-all", default=False, action='store_true')
    run_parser.add_argument("--dev-folds", type=int)
    run_parser.add_argument("--npmi-words", type=int, default=10)
    run_parser.add_argument("--min-acceptable-npmi", type=float, default=0.)
    run_parser.add_argument(
        "--ext-counts-fpath",
    )
    run_parser.add_argument(
        "--ext-vocab-fpath",
    )
    run_args, additional_args = run_parser.parse_known_args()

    outdir_parser = argparse.ArgumentParser()
    outdir_parser.add_argument("-o")
    outdir_args, _ = outdir_parser.parse_known_args(additional_args)

    nyt_counts = fh.load_sparse(run_args.ext_counts_fpath)
    nyt_vocab = fh.read_json(run_args.ext_vocab_fpath)
    
    np.random.seed(run_args.global_seed)
    run_seeds = iter([
        121958, 671155, 131932, 365838, 259178, 921881, 616685, 919314, 130398,
        5591, 11235, 2020, 19, 8000, 1001, 12345,
    ])
    
    # copy over code
    Path(outdir_args.o).mkdir(parents=True, exist_ok=True)
    shutil.copy("run_scholar.py", Path(outdir_args.o, "run_scholar.py"))
    shutil.copy("scholar.py", Path(outdir_args.o, "scholar.py"))

    if Path(outdir_args.o, "dev_metrics.csv").exists():
        old_path = Path(outdir_args.o, "dev_metrics.csv")
        ctime = datetime.fromtimestamp(old_path.stat().st_ctime).strftime("%Y-%m-%d")
        new_path = Path(outdir_args.o, f"dev_metrics_{ctime}.csv")
        Path(old_path).rename(new_path)

    for run in range(run_args.runs):
        print(f"On run {run}")
        if run_args.dev_folds:
            fold = run % run_args.dev_folds
            if fold == 0:
                seed = next(run_seeds) # renew seed
            additional_args += ["--dev-fold", f"{fold}", "--dev-folds", f"{run_args.dev_folds}"] 
        else:
            fold = None
            seed = next(run_seeds)
        additional_args += ['--seed', f'{seed}']
        
        # run scholar
        main(additional_args)

        # load model and store metrics
        try:
            checkpoint = torch.load(Path(outdir_args.o, "torch_model.pt"))
        except EOFError:
            print("Got EOFError, restarting run")
            continue

        m = checkpoint['dev_metrics']
        ppl, npmi, acc = m['perplexity'], m['npmi'], m['accuracy']
        topics = fh.read_text(Path(outdir_args.o, "topics.txt"))

        results = pd.DataFrame({
               'seed': seed,
               'fold': fold, 
               'perplexity_value': float(ppl['value']),
               'perplexity_epoch': int(ppl.get('epoch', 0)),

               'npmi_value': float(npmi['value']),
               'npmi_epoch': int(npmi.get('epoch', 0)),

               'npmi_ext_value': compute_npmi_at_n(
                   topics, nyt_vocab, nyt_counts, n=run_args.npmi_words, silent=True,
                ),
               'tu': np.mean(tu([t.strip().split() for t in topics])),
               
               'accuracy_value': float(acc['value']),
               'accuracy_epoch': int(acc.get('epoch', 0)),
            },
            index=[run],
        )

        results.to_csv(
            Path(outdir_args.o, "dev_metrics.csv"),
            mode='a',
            header=run==0, # only save header for the first run
        )

        if run_args.store_all:
            seed_path = Path(outdir_args.o, str(seed))
            if not seed_path.exists():
                seed_path.mkdir()
            for fpath in Path(outdir_args.o).glob("*"):
                if fpath.name not in ['torch_model.pt', 'dev_metrics.csv'] and fpath.is_file():
                    shutil.copyfile(fpath, Path(seed_path, fpath.name))

        # stop entirely if run was very bad
        if npmi['value'] < run_args.min_acceptable_npmi:
            with open("stopped_due_to_low_npmi.txt", "w") as outfile:
                outfile.write("")
            print(f"Stopped: NPMI of {npmi['value']:0.4f} < {run_args.min_acceptable_npmi}")
            break

    # Save the arguments
    fh.write_to_json(checkpoint["options"].__dict__, Path(outdir_args.o, "args.json"))