import argparse
import json
import os
from os import path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

import dataset_location
import eval_model
from model import SingleViewto3D
from starter.cache import get_checkpoint_path
from starter.media import colorbar_frame, hstack_frames, save_gif
from starter.render import error_points, image_to_frame, render_turntable, voxels_to_mesh

# models to analyse: (tag, flags that identify the model); missing checkpoints are skipped
MODELS = [
    ("vox", ["--type", "vox"]),
    ("point", ["--type", "point"]),
    ("mesh", ["--type", "mesh"]),
    ("mesh_chair", ["--type", "mesh", "--template", "chair"]),
    ("mesh_torus", ["--type", "mesh", "--template", "torus"]),
    ("vox_pw3", ["--type", "vox", "--pos_weight", "3"]),
]
F1_THRESHOLD = 0.05


def get_args_parser():
    parser = argparse.ArgumentParser("Error heatmaps", add_help=False)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--models", default=",".join(t for t, _ in MODELS), type=str)
    parser.add_argument("--gallery", default="0,200,400,600", type=str)  # test chair indices; "" skips
    parser.add_argument("--n_height", default=300, type=int)  # test chairs in the height plot; 0 skips
    parser.add_argument("--n_bins", default=20, type=int)
    parser.add_argument("--n_pred_samples", default=5000, type=int)  # points sampled on predicted surfaces
    parser.add_argument("--n_gt_samples", default=10000, type=int)
    parser.add_argument("--vmax", default=0.1, type=float)  # color scale saturates here
    parser.add_argument("--cmap", default="turbo", type=str)  # dark at both ends, so visible on white
    parser.add_argument("--point_radius", default=0.012, type=float)
    parser.add_argument("--n_frames", default=36, type=int)
    return parser


"""
Data: one seeded view per test chair, read straight from disk
"""
def test_queries(seed):
    with open(dataset_location.SPLITS_PATH) as f:
        split = json.load(f)["test"]
    rng = np.random.default_rng(seed)
    return [(synset, model, int(rng.choice(views)))
            for synset, models in split.items() for model, views in models.items()]


def rendering_dir(synset, model):
    return path.join(dataset_location.R2N2_PATH, "ShapeNetRendering", synset, model, "rendering")


def load_feat(synset, model, view):
    return torch.from_numpy(np.load(path.join(rendering_dir(synset, model), "feats.npy"))[view]).float()


def load_image(synset, model, view):
    image = Image.open(path.join(rendering_dir(synset, model), "%02d.png" % view))
    return torch.from_numpy(np.array(image) / 255.0)[..., :3].float()


def load_mesh(synset, model):
    verts, faces, _ = load_obj(
        path.join(dataset_location.SHAPENET_PATH, synset, model, "model.obj"), load_textures=False
    )
    return Meshes(verts=[verts], faces=[faces.verts_idx])


"""
Models and predicted surfaces
"""
def load_model(flags):
    args = eval_model.get_args_parser().parse_args(flags + ["--load_feat", "--device", "cpu"])
    ckpt = get_checkpoint_path(args)
    if not path.exists(ckpt):
        return None, args, ckpt
    model = SingleViewto3D(args)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    model.eval()
    return model, args, ckpt


@torch.no_grad()
def predicted_surface(model, args, feat, n_samples):
    """
    Points on the predicted surface, (N, 3); None for an empty voxel prediction.
    """
    pred = model(feat[None], args)
    if args.type == "point":
        return pred[0]
    if args.type == "mesh":
        return sample_points_from_meshes(pred, n_samples)[0]
    mesh = voxels_to_mesh(torch.sigmoid(pred))  # the same voxel surface the GIFs render
    return None if mesh is None else sample_points_from_meshes(mesh, n_samples)[0]


def nn_dist(src, dst):
    # distance from each src point to its nearest dst point
    return knn_points(src[None], dst[None], K=1).dists[0, :, 0].sqrt()


"""
Error heatmap GIFs
"""
def render_error_gif(image, pred_pts, gt_pts, args, output_file):
    vmax, cmap = args.vmax, args.cmap
    kw = dict(dist=1.5, elev=15, n_frames=args.n_frames, point_radius=args.point_radius,
              device="cpu", progress=False)
    gt_err = nn_dist(gt_pts, pred_pts) if pred_pts is not None else torch.full((len(gt_pts),), vmax)
    gt_frames = render_turntable(error_points(gt_pts, gt_err, vmax, cmap), **kw)
    if pred_pts is not None:
        pred_err = nn_dist(pred_pts, gt_pts)
        pred_frames = render_turntable(error_points(pred_pts, pred_err, vmax, cmap), **kw)
    else:  # empty voxel prediction: blank panel, and every ground-truth point counts as missing
        pred_frames = [np.full_like(gt_frames[0], 255)] * len(gt_frames)
    h = gt_frames[0].shape[0]
    bar = colorbar_frame(h, vmax, "distance", ticks=[0, F1_THRESHOLD, vmax], cmap=cmap)
    image = image_to_frame(image, height=h)
    n = len(gt_frames)
    # one full turn every 2.4 s, like eval_model's GIFs (72 frames at 30 fps), for any n_frames
    fps = max(1, round(n / 2.4))
    save_gif(hstack_frames([image] * n, pred_frames, gt_frames, [bar] * n), output_file, fps=fps)


"""
Height plot
"""
def height_bins(points, gt_ymin, gt_height, n_bins):
    # normalized height on the ground-truth chair (0 = floor, 1 = top) -> bin index;
    # predicted points outside the chair's height range land in the end bins
    h = (points[:, 1] - gt_ymin) / gt_height
    return (h * n_bins).long().clamp(0, n_bins - 1)


def run(args):
    torch.manual_seed(args.seed)
    queries = test_queries(args.seed)
    wanted = [t.strip() for t in args.models.split(",") if t.strip()]

    models = {}
    for tag, flags in MODELS:
        if tag not in wanted:
            continue
        model, margs, ckpt = load_model(flags)
        if model is None:
            print(f"{tag}: no checkpoint ({ckpt}), skipped")
            continue
        models[tag] = (model, margs)
    print(f"models: {', '.join(models)}")

    # 1. error heatmap GIFs
    os.makedirs("vis", exist_ok=True)
    gallery = [int(i) for i in args.gallery.split(",") if i.strip() != ""]
    for i in [i for i in gallery if i < len(queries)]:
        synset, model_id, view = queries[i]
        feat, image = load_feat(synset, model_id, view), load_image(synset, model_id, view)
        gt_pts = sample_points_from_meshes(load_mesh(synset, model_id), args.n_gt_samples)[0]
        for tag, (model, margs) in models.items():
            pred_pts = predicted_surface(model, margs, feat, args.n_pred_samples)
            out = f"vis/{i}_{tag}_error.gif"
            render_error_gif(image, pred_pts, gt_pts, args, out)
            print(f"wrote {out}")

    # 2. mean missing / spurious error vs normalized height
    if args.n_height <= 0:
        return
    n_bins = args.n_bins
    sums = {tag: {"missing": torch.zeros(n_bins), "spurious": torch.zeros(n_bins),
                  "n_missing": torch.zeros(n_bins), "n_spurious": torch.zeros(n_bins)} for tag in models}
    empty = {tag: 0 for tag in models}
    for synset, model_id, view in queries[: args.n_height]:
        feat = load_feat(synset, model_id, view)
        gt_pts = sample_points_from_meshes(load_mesh(synset, model_id), args.n_gt_samples // 2)[0]
        ymin, height = gt_pts[:, 1].min(), (gt_pts[:, 1].max() - gt_pts[:, 1].min()).clamp(min=1e-6)
        gt_bins = height_bins(gt_pts, ymin, height, n_bins)
        for tag, (model, margs) in models.items():
            pred_pts = predicted_surface(model, margs, feat, args.n_pred_samples)
            if pred_pts is None:
                empty[tag] += 1  # no surface to measure against; reported, not averaged
                continue
            s = sums[tag]
            s["missing"].index_add_(0, gt_bins, nn_dist(gt_pts, pred_pts))
            s["n_missing"].index_add_(0, gt_bins, torch.ones(len(gt_bins)))
            pred_bins = height_bins(pred_pts, ymin, height, n_bins)
            s["spurious"].index_add_(0, pred_bins, nn_dist(pred_pts, gt_pts))
            s["n_spurious"].index_add_(0, pred_bins, torch.ones(len(pred_bins)))

    centers = ((torch.arange(n_bins) + 0.5) / n_bins).tolist()
    curves = {tag: {k: (s[k] / s[f"n_{k}"].clamp(min=1)).tolist() for k in ["missing", "spurious"]}
              for tag, s in sums.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    titles = {"missing": "Missing geometry: ground truth -> nearest prediction",
              "spurious": "Spurious geometry: prediction -> nearest ground truth"}
    for ax, kind in zip(axes, ["missing", "spurious"]):
        for tag in curves:
            ax.plot(centers, curves[tag][kind], marker="o", markersize=3, label=tag)
        ax.axhline(F1_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax.set_title(titles[kind], fontsize=9)
        ax.set_xlabel("normalized height on the chair (0 = floor, 1 = top)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean distance")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Where the errors are, over {min(args.n_height, len(queries))} test chairs "
                 f"(dashed: F1 threshold {F1_THRESHOLD})", fontsize=10)
    plt.savefig("error_height.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    os.makedirs("output", exist_ok=True)
    with open("output/error_height.json", "w") as f:
        json.dump({"height_bin_centers": centers, "curves": curves, "empty_predictions": empty,
                   "n_chairs": min(args.n_height, len(queries)), "seed": args.seed}, f, indent=2)
    print("wrote error_height.png and output/error_height.json")
    for tag in curves:
        note = f" ({empty[tag]} empty predictions excluded)" if empty[tag] else ""
        print(f"  {tag}: mean missing {np.mean(curves[tag]['missing']):.4f}, "
              f"spurious {np.mean(curves[tag]['spurious']):.4f}{note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Error heatmaps", parents=[get_args_parser()])
    run(parser.parse_args())
