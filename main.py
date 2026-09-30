"""Entry point for the assignment code submission.

Running this file should reproduce every result referenced in the webpage
writeup (projX/assignment.md). Add whatever CLI flags/subcommands you need.
"""
import argparse
import torch
import fit_data
from starter.render import render_model
from starter.cache import get_cache_filename


"""
1.1. Fitting a voxel grid (4 points)
"""
def fit_voxel(
        max_iter=None,
        n_frames=72,
        image_size=256,
        output_file="output/fit_voxel.gif",
        cache_input_file=None,
):
    # load or fit points
    if cache_input_file:
        voxels = torch.load(cache_input_file)
    else: 
        voxels = fit_model(
            type="vox",
            max_iter=max_iter,
            cache_output_file=get_cache_filename(output_file)
        )

    # render & save gif
    render_model(
        obj=voxels,
        obj_type="vox",
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file,
        flat_shading=True
    )


"""
1.2. Fitting a point cloud (3 points)
"""
def fit_pointcloud(
        max_iter=None,
        n_frames=72,
        image_size=256,
        output_file="output/fit_pointcloud.gif",
        cache_input_file=None,
):
    # load or fit points
    if cache_input_file:
        points = torch.load(cache_input_file)
    else: 
        points = fit_model(
            type="point",
            max_iter=max_iter,
            cache_output_file=get_cache_filename(output_file)
        )

    # render & save gif
    render_model(
        obj=points,
        obj_type="point",
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file
    )


"""
1.3. Fitting a mesh (3 points)
"""
def fit_mesh(
        max_iter=None,
        n_frames=72,
        image_size=256,
        output_file="output/fit_mesh.gif",
        cache_input_file=None,
):
    # load or fit mesh
    if cache_input_file:
        mesh = torch.load(cache_input_file, weights_only=False)
    else: 
        mesh = fit_model(
            type="mesh",
            max_iter=max_iter,
            cache_output_file=get_cache_filename(output_file)
        )

    # render & save gif
    render_model(
        obj=mesh,
        obj_type="mesh",
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file,
        flat_shading=True
    )    


"""
Helper functions
"""
def test_visuals(type="vox"):
    parser = argparse.ArgumentParser(parents=[fit_data.get_args_parser()])
    args = parser.parse_args(["--type", type, "--test"])
    gt = fit_data.train_model(args)

    render_model(
        obj=gt, 
        obj_type=type, 
        output_file="output/test_"+type+".gif", 
        vox_is_logits=False,
        flat_shading=True,
    )


def fit_model(
        type="vox",
        max_iter=None,
        cache_output_file="output/model.pt",
):
    # fit model
    parser = argparse.ArgumentParser(parents=[fit_data.get_args_parser()])
    arg_list = ["--type", type]
    if max_iter:
        arg_list += ["--max_iter", str(max_iter)]
    args = parser.parse_args(arg_list)
    model = fit_data.train_model(args)

    # cache model
    torch.save(model, get_cache_filename(cache_output_file))

    return model


def main():
    # fit_voxel(max_iter=100)
    # fit_pointcloud(max_iter=10000)
    # fit_mesh(max_iter=10000)
    test_visuals(type="vox")
    return


if __name__ == "__main__":
    main()