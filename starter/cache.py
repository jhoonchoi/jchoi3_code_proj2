import os

import torch


def get_cache_filename(output_file):
    root, _ = os.path.splitext(output_file)
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return root+".pt"


def get_checkpoint_path(args):
    """
    Checkpoint file for a train/eval run. Non-default settings that change the
    model's shape (mesh template, point count) go into the name so different
    runs never overwrite each other; defaults keep checkpoint_{type}.pth.
    """
    name = f"checkpoint_{args.type}"
    if args.type == "mesh" and args.template != "ico4":
        name += f"_{args.template}"
    if args.type == "point" and args.n_points != 1000:
        name += f"_n{args.n_points}"
    if args.type == "vox" and args.pos_weight is not None:
        name += f"_pw{args.pos_weight:g}"
    return name + ".pth"


def save_cache(obj, output_file):
    path = get_cache_filename(output_file)
    torch.save(obj, path)
    return path


def load_cache(path):
    # map to CPU so caches written on the GPU box load anywhere; caches hold
    # Meshes objects, which need the full unpickler
    return torch.load(path, map_location="cpu", weights_only=False)
