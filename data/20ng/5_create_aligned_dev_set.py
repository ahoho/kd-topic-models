#python 3.8
import shutil 
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils import load_jsonlist, save_jsonlist, load_sparse, save_sparse, load_json, save_json

if __name__ == "__main__":

    dev_dir = Path("./aligned/dev")
    dev_dir.mkdir(exist_ok=True)
    for fpath in Path("./aligned").glob("*"):
        if fpath.is_file():
            shutil.copy(str(fpath), str(Path(dev_dir, fpath.name)))

    # Load in the ids from the replicated fpath
    repl_train_ids = load_json("./replicated/dev/train.ids.json")
    repl_dev_ids = load_json("./replicated/dev/dev.ids.json")

    data = load_jsonlist(Path(dev_dir, "train.jsonlist"))
    counts = load_sparse(Path(dev_dir, "train.npz"))
    ids = load_json(Path(dev_dir, "train.ids.json"))

    # split based on how the replication data was split
    data_train = [doc for doc in data if doc['id'] in repl_train_ids]
    data_dev = [doc for doc in data if doc['id'] in repl_dev_ids]

    counts_train = counts[np.array([doc['id'] in repl_train_ids for doc in data]), :]
    counts_dev = counts[np.array([doc['id'] in repl_dev_ids for doc in data]), :]

    ids_train = [id for id in ids if id in repl_train_ids]
    ids_dev = [id for id in ids if id in repl_dev_ids]

    assert(len(data_train) == counts_train.shape[0] == len(ids_train))
    assert(len(data_dev) == counts_dev.shape[0] == len(ids_dev))
    
    # save
    save_jsonlist(data_train, Path(dev_dir, "train.jsonlist"))
    save_jsonlist(data_dev, Path(dev_dir, "dev.jsonlist"))
    
    save_sparse(counts_train, Path(dev_dir, "train.npz"))
    save_sparse(counts_dev, Path(dev_dir, "dev.npz"))

    save_json(ids_train, Path(dev_dir, "train.ids.json"))
    save_json(ids_dev, Path(dev_dir, "dev.ids.json"))