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


class ResnetBlockFC(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc_0 = nn.Linear(size, size)
        self.fc_1 = nn.Linear(size, size)
        self.actvn = nn.ReLU()
        # zero-init the residual branch so each block starts as the identity
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        return x + dx


class ImplicitDecoder(nn.Module):
    def __init__(self, c_dim=512, hidden_size=128, n_blocks=5, grid_size=32):
        super().__init__()
        self.fc_p = nn.Linear(3, hidden_size)
        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_size, 1)

        # query grid in (-1, 1)^3, flattened in the voxel targets' [z, y, x] order so
        # grid point i lines up with target voxel i; each row is an (x, y, z) point
        self.grid_size = grid_size
        lin = torch.linspace(-1, 1, grid_size)
        z, y, x = torch.meshgrid(lin, lin, lin, indexing="ij")
        self.register_buffer("grid", torch.stack([x, y, z], dim=-1).reshape(-1, 3), persistent=False)

    def forward(self, c, query_idx=None):
        """
        c: (b, c_dim) image features. query_idx: (b, k) indices into the flattened grid,
        or None for the full grid. Returns occupancy logits (b, k), or (b, 1, D, D, D)
        on the full grid, the same layout as the voxel decoder.
        """
        b = c.shape[0]
        p = self.grid[query_idx] if query_idx is not None else self.grid.expand(b, -1, -1)
        net = self.fc_p(p)
        for fc_c, block in zip(self.fc_c, self.blocks):
            # the code is shared by every point of a shape: project it once, broadcast over points
            net = block(net + fc_c(c)[:, None, :])
        logits = self.fc_out(torch.relu(net)).squeeze(-1)
        if query_idx is not None:
            return logits
        d = self.grid_size
        return logits.reshape(b, 1, d, d, d)


class ParametricDecoder(nn.Module):
    def __init__(self, n_points, n_charts=10, c_dim=512, widths=(128, 256, 512, 512)):
        super().__init__()
        assert n_points % n_charts == 0, "n_points must be divisible by n_charts"
        self.n_charts = n_charts
        self.points_per_chart = n_points // n_charts

        def chart_mlp():
            layers = []
            for w_in, w_out in zip(widths[:-1], widths[1:]):
                layers += [nn.ReLU(), nn.Linear(w_in, w_out)]
            return nn.Sequential(*layers, nn.ReLU(), nn.Linear(widths[-1], 3, bias=False))

        self.fc_uv = nn.ModuleList([nn.Linear(2, widths[0]) for _ in range(n_charts)])
        self.fc_c = nn.ModuleList([nn.Linear(c_dim, widths[0]) for _ in range(n_charts)])
        self.mlps = nn.ModuleList([chart_mlp() for _ in range(n_charts)])

        # evaluation samples: a regular grid of cell centers when points_per_chart is a
        # square number, otherwise a fixed random set; the same for every chart
        side = round(self.points_per_chart ** 0.5)
        if side * side == self.points_per_chart:
            lin = (torch.arange(side) + 0.5) / side
            u, v = torch.meshgrid(lin, lin, indexing="ij")
            uv = torch.stack([u, v], dim=-1).reshape(-1, 2)
        else:
            uv = torch.rand(self.points_per_chart, 2, generator=torch.Generator().manual_seed(0))
        self.register_buffer("eval_uv", uv, persistent=False)

    def forward(self, c, uv=None):
        b = c.shape[0]
        if uv is None:
            if self.training:
                uv = torch.rand(b, self.n_charts, self.points_per_chart, 2, device=c.device)
            else:
                uv = self.eval_uv.expand(b, self.n_charts, -1, -1)
        points = []
        for k in range(self.n_charts):
            # the image code is shared by every sample of a chart: project it once, broadcast
            h = self.fc_uv[k](uv[:, k]) + self.fc_c[k](c)[:, None, :]
            points.append(self.mlps[k](h))
        return torch.cat(points, dim=1)


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox" and args.vox_decoder == "implicit":
            # Input: b x 512 (+ query points on the 32^3 grid)
            # Output: b x 1 x 32 x 32 x 32 occupancy logits on the full grid
            self.decoder = ImplicitDecoder(c_dim=512, hidden_size=128, n_blocks=5, grid_size=32)
        elif args.type == "vox":
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
        elif args.type == "point" and args.point_decoder == "parametric":
            # Input: b x 512 (+ 2D samples per chart)
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            self.decoder = ParametricDecoder(n_points=args.n_points, n_charts=args.n_charts)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),                       # 512 -> 1024
                nn.BatchNorm1d(1024),                       # 1024
                nn.ReLU(),                                  # 1024
                nn.Linear(1024, 3 * self.n_point),          # 1024 -> 3 * self.n_point
                nn.Unflatten(1, (self.n_point, 3))          # (self.n_point, 3)
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

    def forward(self, images, args, query_idx=None):
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
        if args.type == "vox" and args.vox_decoder == "implicit":
            # query_idx: (b, k) grid points to predict during training; None = full grid
            return self.decoder(encoded_feat, query_idx)

        elif args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

