import os

import torch


def get_cache_filename(output_file):
    root, _ = os.path.splitext(output_file)
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return root+".pt"


def save_cache(obj, output_file):
    path = get_cache_filename(output_file)
    torch.save(obj, path)
    return path


def load_cache(path):
    # map to CPU so caches written on the GPU box load anywhere; caches hold
    # Meshes objects, which need the full unpickler
    return torch.load(path, map_location="cpu", weights_only=False)
