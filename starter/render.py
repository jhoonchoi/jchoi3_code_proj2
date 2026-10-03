import matplotlib
import numpy as np
import torch
import pytorch3d.renderer
import pytorch3d.structures
from tqdm.auto import tqdm

from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    HardFlatShader,
)

import utils_vox
from starter.media import to_uint8, save_gif, hstack_frames

DEFAULT_COLOR = (0.7, 0.7, 1.0)


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None, flat=False):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    # prep shader
    if flat:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardFlatShader(device=device, lights=lights),
        )
    else:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=device, lights=lights),
        )
    return renderer


"""
Helper function to render reused turntable view & save gif
"""
@torch.no_grad()
def render_turntable(
    obj,
    dist=3.0,
    elev=0.0,
    n_frames=36,
    image_size=256,
    azim_start=0.0,
    up=((0.0, 1.0, 0.0),),
    at=((0.0, 0.0, 0.0),),
    fov=60.0,
    flat=False,
    lights=None,
    light_follows_camera=True,
    point_radius=0.01,
    background_color=(1.0, 1.0, 1.0),
    device=None,
    progress=True,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()
    obj = obj.to(device)

    # determine if object is pointcloud
    is_pc = isinstance(obj, pytorch3d.structures.Pointclouds)

    # get renderer
    if is_pc:
        renderer = get_points_renderer(
            image_size=image_size,
            radius=point_radius,
            background_color=background_color,
            device=device,
        )
    else:
        renderer = get_mesh_renderer(
            image_size=image_size,
            device=device,
            flat=flat
        )

    # generate azimuth angles for full 360 animation
    azims = torch.linspace(azim_start, azim_start + 360.0, n_frames + 1)[:-1]

    # render image frames
    frames = []
    for azim in tqdm(azims, desc="turntable", disable=not progress):
        # get R, T for azimuth
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist, elev=elev, azim=float(azim), up=up, at=at
        )

        # prep camera with R, T
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=fov, device=device
        )

        if is_pc:
            # if pointcloud no lights
            rend = renderer(obj, cameras=cameras)
        else:
            # set up lights
            if lights is not None:
                frame_lights = lights
            elif light_follows_camera:
                if flat:
                    frame_lights = pytorch3d.renderer.DirectionalLights(
                        direction=cameras.get_camera_center() + torch.tensor([[1.0,1.0,0.0]], device=device),
                        device=device
                    )
                else:
                    frame_lights = pytorch3d.renderer.PointLights(
                        location=cameras.get_camera_center(), device=device
                    )
            else:
                if flat:
                    frame_lights = pytorch3d.renderer.DirectionalLights(
                        device=device
                    )
                else:
                    frame_lights = pytorch3d.renderer.PointLights(
                        location=[[0.0, 0.0, -3.0]], device=device
                    )
            # if mesh, render with light
            rend = renderer(obj, cameras=cameras, lights=frame_lights)
        # stack image frames for gif making
        frames.append(to_uint8(rend[0, ..., :3]))
    return frames


"""
Converters from fit_data / eval_model outputs to renderable Pytorch3D structures.
Each builds a new object, so the caller's tensors and meshes are never modified.
"""
def color_mesh(mesh, color=DEFAULT_COLOR):
    """
    Returns the first mesh of a (possibly untextured) Meshes batch with a solid
    vertex color.
    """
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    texture = torch.ones_like(verts) * torch.tensor(color, device=verts.device)
    return pytorch3d.structures.Meshes(
        verts=[verts],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(verts_features=[texture]),
    )


def color_points(points, lo=-0.4, hi=0.4):
    """
    Builds a point cloud from (1, N, 3) points, colored by position.
    """
    color = ((points - lo) / (hi - lo)).clamp(0.0, 1.0)
    return pytorch3d.structures.Pointclouds(points=points, features=color)


def error_points(points, dists, vmax=0.1, cmap="viridis"):
    """
    Builds a point cloud from (N, 3) points colored by a per-point error (N,),
    on a fixed [0, vmax] colormap so colors mean the same distance in every render.
    """
    rgb = matplotlib.colormaps[cmap]((dists / vmax).clamp(0.0, 1.0).cpu().numpy())[:, :3]
    color = torch.tensor(rgb, dtype=points.dtype, device=points.device)
    return pytorch3d.structures.Pointclouds(points=[points], features=[color])


def voxels_to_mesh(voxels, thresh=0.5, color=DEFAULT_COLOR):
    """
    Converts a voxel grid to a mesh with one cube per occupied voxel, in the same
    world frame as the R2N2 ground-truth mesh.

    Args:
        voxels (torch.Tensor): (1, Z, Y, X) occupancy probabilities, indexed
            [z, y, x] as produced by utils_vox.voxelize_xyz.
        thresh (float): Voxels with occupancy >= thresh are drawn.

    Returns:
        Meshes, or None if no voxel is occupied.
    """
    Z, Y, X = voxels.shape[-3:]

    # cubify reads the grid as (D, H, W) = (Z, Y, X), emits (x, y, z) verts, and
    # with align="center" puts voxel centers on linspace(-1, 1) along each axis
    mesh = cubify(voxels.reshape(1, Z, Y, X), thresh, align="center")
    if mesh.isempty():
        return None
    verts = mesh.verts_list()[0]

    # [-1, 1] -> voxel index space -> world coordinates (inverse of the dataset voxelization)
    size = torch.tensor([X, Y, Z], dtype=verts.dtype, device=verts.device)
    verts = (verts + 1.0) * (size - 1.0) / 2.0
    verts = utils_vox.Mem2Ref(verts.unsqueeze(0), Z, Y, X, device=verts.device).squeeze(0)

    return color_mesh(
        pytorch3d.structures.Meshes(verts=[verts], faces=mesh.faces_list()), color
    )


@torch.no_grad()
def to_renderable(obj, obj_type, device=None):
    if device is None:
        device = get_device()
    obj = obj.to(device)

    if obj_type == "vox":
        return voxels_to_mesh(obj)
    elif obj_type == "point":
        return color_points(obj)
    elif obj_type == "mesh":
        return color_mesh(obj)
    elif obj_type == "textured_mesh":
        return obj  # already colored, e.g. ParametricDecoder.chart_meshes
    raise ValueError(f"unknown type {obj_type}")


"""
utils to help rendering from train_data outputs
"""
def render_frames(
        obj,
        obj_type,
        dist=1.5,
        elev=15,
        n_frames=72,
        image_size=256,
        device=None,
        flat_shading=False,
):
    if device is None:
        device = get_device()

    renderable = to_renderable(obj, obj_type, device=device)

    # nothing to draw (e.g. an empty voxel prediction): plain background frames
    if renderable is None:
        blank = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        return [blank] * n_frames

    return render_turntable(
        renderable,
        flat=flat_shading,
        dist=dist,
        elev=elev,
        n_frames=n_frames,
        image_size=image_size,
        device=device,
    )

def image_to_frame(image, height):
    """
    Converts an (H, W, 3) RGB image in [0, 1] to a uint8 frame of the given
    height (aspect ratio kept), so it can be stacked next to rendered frames.
    """
    image = image.detach().float().cpu().permute(2, 0, 1)[None]  # 1 x 3 x H x W
    width = round(image.shape[-1] * height / image.shape[-2])
    image = torch.nn.functional.interpolate(
        image, size=(height, width), mode="bilinear", align_corners=False
    )
    return to_uint8(image[0].permute(1, 2, 0))


def render_comparison(image, obj, obj_type, mesh_gt, output_file, fps=30, **render_kwargs):
    """
    Renders object and gt mesh on the same turntable and saves them side
    by side as one gif: input image | object | gt mesh.
    """
    obj_frames = render_frames(obj, obj_type, **render_kwargs)
    gt_frames = render_frames(mesh_gt, "mesh", **render_kwargs)

    # the input image doesn't move; repeat it once per turntable frame
    image_frame = image_to_frame(image, height=obj_frames[0].shape[0])
    image_frames = [image_frame] * len(obj_frames)

    return save_gif(hstack_frames(image_frames, obj_frames, gt_frames), output_file, fps=fps)

def render_side_by_side(objs, obj_type, output_file, fps=30, **render_kwargs):
    """
    Renders each object on the same turntable and saves them side by side
    (left to right in the order given) as one gif.
    """
    frame_lists = [render_frames(obj, obj_type, **render_kwargs) for obj in objs]
    return save_gif(hstack_frames(*frame_lists), output_file, fps=fps)


def render_model(obj, obj_type, output_file, fps=30, **render_kwargs):
    return render_side_by_side([obj], obj_type, output_file, fps=fps, **render_kwargs)
