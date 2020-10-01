#python 3.8
from pathlib import Path
import argparse

from utils import load_jsonlist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    input_fpaths = Path(args.input_dir).glob("*.jsonlist")
    for fpath in input_fpaths:
        doc_text = [doc['text'] for doc in load_jsonlist(fpath)]
        with open(Path(args.input_dir, f"{fpath.stem}.raw.txt"), "w") as outfile:
            for text in doc_text:
                outfile.write(f"{text}\n")