"""Q3.3: precomputed ground truth for fast point-cloud training.

Loading a training sample through R2N2 parses the full mesh (.obj) every step;
car meshes are ~44k vertices, ~1 s each, which makes data loading the bottleneck
on CPU. This builds, once per split, a single file with every model's --load_feat
features and n_cache points sampled uniformly on its ground-truth surface.
Training then draws a random view and a random n_points subset of the cached
points per step, which is again a uniform surface sample, with no mesh parsing.

    python gt_cache.py --classes 3c --split train     # -> gt_cache/3c_train.pt
"""
import argparse
import json
import os
from multiprocessing import Pool
from os import path

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

import dataset_location

CACHE_DIR = "gt_cache"


def cache_path(classes, split):
    return path.join(CACHE_DIR, f"{classes}_{split}.pt")


def _load_model(job):
    # one model: (features for its split views, n_cache surface points); runs in a worker
    shapenet_path, r2n2_path, synset, model, views, n_cache, seed = job
    feats = np.load(path.join(r2n2_path, "ShapeNetRendering", synset, model, "rendering", "feats.npy"))
    verts, faces, _ = load_obj(path.join(shapenet_path, synset, model, "model.obj"), load_textures=False)
    torch.manual_seed(seed)
    points = sample_points_from_meshes(Meshes([verts], [faces.verts_idx]), n_cache)[0]
    return torch.from_numpy(feats[: len(views)]).float(), points.half()


def build(classes="3c", split="train", n_cache=10000, workers=4, output=None):
    shapenet_path, r2n2_path, splits_path = dataset_location.dataset_paths(classes == "3c")
    with open(splits_path) as f:
        split_dict = json.load(f)[split]
    models = [(synset, model, views) for synset, ms in split_dict.items() for model, views in ms.items()]
    jobs = [(shapenet_path, r2n2_path, s, m, v, n_cache, i) for i, (s, m, v) in enumerate(models)]

    print(f"caching {len(jobs)} {split} models ({classes}) with {workers} workers")
    feats, points = [], []
    with Pool(workers) as pool:
        for i, (f, p) in enumerate(pool.imap(_load_model, jobs, chunksize=8)):
            feats.append(f)
            points.append(p)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    output = output or cache_path(classes, split)
    os.makedirs(path.dirname(output), exist_ok=True)
    torch.save({
        "synset_ids": [s for s, _, _ in models],
        "model_ids": [m for _, m, _ in models],
        "feats": torch.stack(feats),    # (models, views, 512) float32
        "points": torch.stack(points),  # (models, n_cache, 3) float16
    }, output)
    print(f"wrote {output}")
    return output


class CachedPointDataset(torch.utils.data.Dataset):
    """
    Training samples from a cache: (feature of a random view, n_points random cached
    surface points). Everything is in memory, so no DataLoader workers are needed.
    """
    def __init__(self, cache_file, n_points):
        cache = torch.load(cache_file, map_location="cpu")
        self.feats, self.points = cache["feats"], cache["points"]
        self.synset_ids = cache["synset_ids"]
        self.n_points = n_points
        assert n_points <= self.points.shape[1], "n_points exceeds the cached points per model"

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, i):
        view = torch.randint(self.feats.shape[1], ()).item()
        idx = torch.randperm(self.points.shape[1])[: self.n_points]
        return self.feats[i, view], self.points[i, idx].float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build a ground-truth cache")
    parser.add_argument("--classes", default="3c", choices=["chair", "3c"], type=str)
    parser.add_argument("--split", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--n_cache", default=10000, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()
    build(args.classes, args.split, args.n_cache, args.workers, args.output)
