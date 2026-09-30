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

from starter.media import to_uint8, save_gif


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
utils to help rendering from train_data outputs
"""
def render_model(
        obj,
        obj_type,
        output_file,
        dist=1.5,
        elev=15,
        n_frames=72,
        image_size=256,
        device=None,
        flat_shading=False,
        vox_is_logits=True,
):
    if device is None:
        device = get_device()
    obj = obj.to(device)
    
    # color variables
    lo, hi = -0.4, 0.4

    # detach just in case
    obj = obj.detach()

    if obj_type == "vox":
        if vox_is_logits:
            obj = torch.sigmoid(obj)

        # marching cubes
        mesh = cubify(
            voxels=obj,
            thresh=0.5,
        )

        # scale
        mesh.scale_verts_(0.5)
        
        # generate color texture
        texture = torch.ones_like(mesh.verts_padded()) * torch.tensor([0.7, 0.7, 1], device=device)
        mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=texture)
        obj_renderable = mesh

    elif obj_type == "point":
        color = ((obj - lo) / (hi - lo)).clamp(0.0, 1.0)
        # build pointcloud
        obj_renderable = pytorch3d.structures.Pointclouds(
            points=obj, features=color,
        )

    elif obj_type == "mesh":
        # generate color texture
        texture = torch.ones_like(obj.verts_padded()) * torch.tensor([0.7, 0.7, 1], device=device)
        obj.textures = pytorch3d.renderer.TexturesVertex(verts_features=texture)
        obj_renderable = obj

    else:
        raise ValueError(f"unknown type {obj_type}")

    # render
    frames = render_turntable(
        obj_renderable,
        flat=flat_shading,
        dist=dist,
        elev=elev,
        n_frames=n_frames,
        image_size=image_size,
    )

    # save gif
    save_gif(frames, output_file, fps=30)