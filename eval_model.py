import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import numpy as np
import utils_vox
import matplotlib.pyplot as plt 
from starter.render import render_comparison
from starter.cache import get_checkpoint_path

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--template', default='ico4', choices=['ico4', 'chair', 'torus'], type=str)
    parser.add_argument('--pos_weight', default=None, type=float)  # only selects the checkpoint
    parser.add_argument('--vox_decoder', default='deconv', choices=['deconv', 'implicit'], type=str)
    parser.add_argument('--point_decoder', default='mlp', choices=['mlp', 'parametric'], type=str)
    parser.add_argument('--n_charts', default=10, type=int)
    # classes the model was trained on (selects the checkpoint) and the test set to use;
    # both default to L3D_FULL_DATASET
    default_classes = '3c' if dataset_location.use_full_dataset else 'chair'
    parser.add_argument('--classes', default=default_classes, choices=['chair', '3c'], type=str)
    parser.add_argument('--eval_classes', default=default_classes, choices=['chair', '3c'], type=str)
    parser.add_argument('--tag', default=None, type=str)  # output name suffix, defaults to --type
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {output_tag(args)}')
    plt.savefig(f'eval_{output_tag(args)}', bbox_inches='tight')
    plt.close(fig)


def output_tag(args):
    # name used for eval_{tag}.png and vis/{step}_{tag}.gif; defaults to the type
    return args.tag or args.type


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def return_zero_metrics(thresholds):
    metrics = {}
    for t in thresholds:
        metrics["Precision@%f" % t] = torch.tensor(0.0)
        metrics["Recall@%f" % t] = torch.tensor(0.0)
        metrics["F1@%f" % t] = torch.tensor(0.0)
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        Z,Y,X = voxels_src.shape[-3:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        # marching cubes returns verts in the grid's [z, y, x] index order
        vertices_src = torch.tensor(vertices_src).float()[:, [2, 1, 0]]
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        if mesh_src.isempty():
            return return_zero_metrics(thresholds)
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        
        # voxel index space -> world coordinates (inverse of the dataset voxelization);
        # r2n2_custom puts the voxels in the mesh frame, so no further alignment
        pred_points = utils_vox.Mem2Ref(pred_points, Z, Y, X)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    shapenet_path, r2n2_path, splits_path = dataset_location.dataset_paths(args.eval_classes == '3c')
    r2n2_dataset = R2N2("test", shapenet_path, r2n2_path, splits_path, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []
    f1_05_by_class = {}

    if args.load_checkpoint:
        checkpoint = torch.load(get_checkpoint_path(args), map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {get_checkpoint_path(args)} (step {checkpoint['step']})")
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = torch.sigmoid(predictions)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if (step % args.vis_freq) == 0:
            # visualization block
            if args.type == "point" and args.point_decoder == "parametric" and args.load_feat:
                # render the learned chart surfaces (one color per chart), not just the
                # sampled points; with --load_feat images_gt holds the features
                vis_obj, vis_type = model.decoder.chart_meshes(images_gt[:1]), "textured_mesh"
            else:
                vis_obj, vis_type = predictions, args.type
            render_comparison(
                image=feed_dict["images"][0],  # the RGB view the features came from
                obj=vis_obj,
                obj_type=vis_type,
                mesh_gt=mesh_gt,
                output_file=f'vis/{step}_{output_tag(args)}.gif',
            )

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        f1_05_by_class.setdefault(feed_dict['label'][0], []).append(float(f1_05))
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    for label, scores in f1_05_by_class.items():
        print(f"Avg F1@0.05 {label}: {np.mean(scores):.3f} ({len(scores)} models)")
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
