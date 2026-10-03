import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt,pos_weight=None):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# pos_weight: optional weight on occupied voxels (None = plain BCE, as in Q1.1);
	# a 1-element tensor broadcasts over the whole grid
	if pos_weight is not None:
		pos_weight = torch.tensor([pos_weight], dtype=voxel_src.dtype, device=voxel_src.device)
	loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	return loss(voxel_src,voxel_tgt)

def _chamfer_one_way(x, y):
	# x, y: b x n_points x 3
	nn = knn_points(x, y, K=1)
	dists = nn.dists[...,0]
	dist = dists.mean(1).mean(0)
	return dist

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_tgt: b x n_points x 3
	precision = _chamfer_one_way(point_cloud_src, point_cloud_tgt)
	recall = _chamfer_one_way(point_cloud_tgt, point_cloud_src)
	return precision + recall

def smoothness_loss(mesh_src):
	return mesh_laplacian_smoothing(mesh_src)