import os

# specify the root location where u downloaded the dataset
# defaults to this folder; override with L3D_DATA_ROOT (e.g. on the GPU box)
root_location = os.environ.get("L3D_DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))
use_full_dataset = os.environ.get("L3D_FULL_DATASET", "0") == "1"


def dataset_paths(full):
    """
    (SHAPENET_PATH, R2N2_PATH, SPLITS_PATH) for the chair-only dataset, or for the
    extended chair/plane/car dataset (Q3.3) when full is True.
    """
    dataset_name = "r2n2_shapenet_dataset_full" if full else "r2n2_shapenet_dataset"
    # split file contains data entry for 3 classes / for the 03001627 (chair) class
    split = "split_3c.json" if full else "split_03001627.json"
    root = f"{root_location}/{dataset_name}"
    return f"{root}/shapenet", f"{root}/r2n2", f"{root}/{split}"


# defaults, selected by L3D_FULL_DATASET
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)
SHAPENET_PATH, R2N2_PATH, SPLITS_PATH = dataset_paths(use_full_dataset)
