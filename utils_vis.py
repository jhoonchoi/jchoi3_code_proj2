import imageio
import numpy as np
import pytorch3d.renderer
import pytorch3d.structures
import torch
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    HardFlatShader,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
)
from tqdm.auto import tqdm

import utils_vox

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
        device = get_device()
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
        flat (bool): Use flat shading instead of Phong shading.
    """
    if device is None:
        device = get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    # prep shader
    if flat:
        shader = HardFlatShader(device=device, lights=lights)
    else:
        shader = HardPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=shader,
    )
    return renderer


"""
Converters from the raw tensors used in fit_data / eval_model to renderable
Pytorch3D structures.
"""
def mesh_from_verts_faces(verts, faces, color=DEFAULT_COLOR):
    """
    Builds a single solid-colored mesh.

    Args:
        verts (torch.Tensor): (V, 3) vertices.
        faces (torch.Tensor): (F, 3) faces.
        color (tuple): RGB color applied to every vertex.
    """
    verts = verts.detach().float()
    faces = faces.detach().long().to(verts.device)
    textures = torch.ones_like(verts) * torch.tensor(color, device=verts.device)
    return pytorch3d.structures.Meshes(
        verts=[verts],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex([textures]),
    )


def colorize_mesh(mesh, color=DEFAULT_COLOR):
    """
    Returns a copy of the first mesh in a (possibly untextured) Meshes batch
    with a solid vertex color, ready for render_turntable.
    """
    return mesh_from_verts_faces(mesh.verts_list()[0], mesh.faces_list()[0], color)


def pointcloud_from_points(points, color=DEFAULT_COLOR):
    """
    Builds a single solid-colored point cloud.

    Args:
        points (torch.Tensor): (N, 3) or (1, N, 3) points.
        color (tuple): RGB color applied to every point.
    """
    points = points.detach().float().reshape(-1, 3)
    features = torch.ones_like(points) * torch.tensor(color, device=points.device)
    return pytorch3d.structures.Pointclouds(points=[points], features=[features])


def voxels_to_mesh(voxels, thresh=0.5, color=DEFAULT_COLOR):
    """
    Converts a voxel grid to a mesh with one cube per occupied voxel (cubify),
    expressed in the same world frame as the R2N2 ground-truth mesh.

    Args:
        voxels (torch.Tensor): (Z, Y, X) or (1, Z, Y, X) occupancy probabilities,
            indexed [z, y, x] as produced by utils_vox.voxelize_xyz. Pass
            sigmoid(logits) for network / optimized outputs.
        thresh (float): Voxels with occupancy >= thresh are drawn.
        color (tuple): RGB color applied to every vertex.

    Returns:
        Meshes, or None if no voxel is occupied.
    """
    voxels = voxels.detach().float().reshape(1, *voxels.shape[-3:])
    Z, Y, X = voxels.shape[1:]

    # cubify reads the grid as (D, H, W) = (Z, Y, X), emits (x, y, z) verts, and
    # with align="center" puts voxel centers on linspace(-1, 1) along each axis
    mesh = cubify(voxels, thresh, align="center")
    verts, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    if len(faces) == 0:
        return None

    # [-1, 1] -> voxel index space -> world coordinates (inverse of the dataset voxelization)
    size = torch.tensor([X, Y, Z], dtype=verts.dtype, device=verts.device)
    verts = (verts + 1.0) * (size - 1.0) / 2.0
    verts = utils_vox.Mem2Ref(verts.unsqueeze(0), Z, Y, X, device=verts.device).squeeze(0)

    return mesh_from_verts_faces(verts, faces, color)


"""
Helper function to render reused turntable view & save gif
"""
@torch.no_grad()
def render_turntable(
    obj,
    dist=1.5,
    elev=20.0,
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
                        direction=cameras.get_camera_center() + torch.tensor([[1.0, 1.0, 0.0]], device=device),
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


def to_uint8(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def save_gif(frames, path, fps=15, loop=0):
    duration = 1000 // fps  # ms per frame
    imageio.mimsave(path, frames, duration=duration, loop=loop)
    return path


def hstack_frames(*frame_lists, pad=8, pad_value=255):
    n = len(frame_lists[0])
    assert all(len(f) == n for f in frame_lists), "frame lists must match"
    out = []
    for i in range(n):
        parts = []
        for j, frames in enumerate(frame_lists):
            if j > 0:
                h = frames[i].shape[0]
                parts.append(np.full((h, pad, 3), pad_value, dtype=np.uint8))
            parts.append(frames[i])
        out.append(np.concatenate(parts, axis=1))
    return out


def render_side_by_side(objs, path, fps=15, **turntable_kwargs):
    """
    Renders each object on the same turntable and saves them side by side
    (left to right in the order given) as one gif.
    """
    frame_lists = [render_turntable(obj, **turntable_kwargs) for obj in objs]
    return save_gif(hstack_frames(*frame_lists), path, fps=fps)
