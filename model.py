from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
import os
from pytorch3d.utils import ico_sphere, torus
import pytorch3d
from starter.cache import load_cache

# Q1.3 fit of training chair 0 (main.fit_mesh): an ico_sphere(4) deformed into a chair
FITTED_CHAIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "fit_mesh.pt")


def make_template(name, device):
    """
    Template mesh for the mesh decoder to deform. The offset layer is
    zero-initialized, so every prediction starts exactly at this shape.
    """
    if name == "ico4":
        return ico_sphere(4, device)
    if name == "chair":
        fitted, _ = load_cache(FITTED_CHAIR_PATH)
        return fitted.to(device)
    if name == "torus":
        # 40 x 64 = 2560 verts, close to ico_sphere(4)'s 2562; the ring lies in the
        # xy-plane, so its hole goes front to back through the chair
        return torus(r=0.1, R=0.3, sides=40, rings=64, device=device)
    raise ValueError(f"unknown template {name}")


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = nn.Sequential(
                nn.Linear(512, 8192),                       # 512 -> 8192
                nn.Unflatten(1, (128, 4, 4, 4)),            # 8192 -> 128 * 4^3
                nn.BatchNorm3d(128),                        # 128 * 4^3
                nn.ReLU(),                                  # 128 * 4^3
                nn.ConvTranspose3d(                         # 128 * 4^3 -> 64 * 8^3
                    in_channels=128, 
                    out_channels=64, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1
                ),
                nn.BatchNorm3d(64),                         # 64 * 8^3
                nn.ReLU(),                                  # 64 * 8^3
                nn.ConvTranspose3d(                         # 64 * 8^3 -> 32 * 16^3
                    in_channels=64, 
                    out_channels=32, 
                    kernel_size=4,
                    stride=2, 
                    padding=1
                ),
                nn.BatchNorm3d(32),                         # 32 * 16^3
                nn.ReLU(),                                  # 32 * 16^3
                nn.ConvTranspose3d(                         # 32 * 16^3 -> 1 * 32^3
                    in_channels=32, 
                    out_channels=1, 
                    kernel_size=4,
                    stride=2, 
                    padding=1
                ),
            )
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),                       # 512 -> 1024
                nn.BatchNorm1d(1024),                       # 1024
                nn.ReLU(),                                  # 1024
                nn.Linear(1024, 3 * self.n_point),          # 1024 -> 3 * self.n_point
            )
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = make_template(args.template, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            num_verts = int(self.mesh_pred.num_verts_per_mesh()[0])

            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),                       # 512 -> 1024
                nn.BatchNorm1d(1024),                       # 1024
                nn.ReLU(),                                  # 1024
                nn.Linear(1024, 3 * num_verts),             # 1024 -> 3 * num_verts
                nn.Unflatten(1, (num_verts, 3))             # (num_verts, 3)
            )
            # zero-init the offset layer so training starts from the undeformed sphere
            nn.init.zeros_(self.decoder[3].weight)
            nn.init.zeros_(self.decoder[3].bias)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

