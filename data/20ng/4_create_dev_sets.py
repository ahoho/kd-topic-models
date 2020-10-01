#python 3.8
import shutil 
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils import load_jsonlist, save_jsonlist, load_sparse, save_sparse, load_json, save_json

if __name__ == "__main__":
    INPUT_DIR = "./replicated"
    DEV_SIZE = 0.2
    SEED = 11235
    dev_dir = Path(INPUT_DIR, "dev")
    dev_dir.mkdir(exist_ok=True)
    for fpath in Path(INPUT_DIR).glob("*"):
        if fpath.is_file():
            shutil.copy(str(fpath), str(Path(dev_dir, fpath.name)))

    # Load in the ids
    data = load_jsonlist(Path(dev_dir, "train.jsonlist"))
    counts = load_sparse(Path(dev_dir, "train.npz"))
    tokens = load_json(Path(dev_dir, "train.tokens.json"))
    ids = load_json(Path(dev_dir, "train.ids.json"))

    # split
    (
        data_train,
        data_dev,
        counts_train,
        counts_dev,
        tokens_train,
        tokens_dev,
        ids_train,
        ids_dev,
    ) = train_test_split(data, counts, tokens, ids, test_size=DEV_SIZE, random_state=SEED)

    # save
    save_jsonlist(data_train, Path(dev_dir, "train.jsonlist"))
    save_jsonlist(data_dev, Path(dev_dir, "dev.jsonlist"))
    
    save_sparse(counts_train, Path(dev_dir, "train.npz"))
    save_sparse(counts_dev, Path(dev_dir, "dev.npz"))

    save_json(tokens_train, Path(dev_dir, "train.tokens.json"))
    save_json(tokens_dev, Path(dev_dir, "dev.tokens.json"))

    save_json(ids_train, Path(dev_dir, "train.ids.json"))
    save_json(ids_dev, Path(dev_dir, "dev.ids.json"))