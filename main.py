"""Entry point for the assignment code submission.

Running this file should reproduce every result referenced in the webpage
writeup (projX/assignment.md). Add whatever CLI flags/subcommands you need.
"""
import argparse
import fit_data
from starter.render import render_model, render_side_by_side
from starter.cache import save_cache, load_cache


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
    fit_and_render(
        type="vox",
        max_iter=max_iter,
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file,
        cache_input_file=cache_input_file,
        flat_shading=True,
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
    fit_and_render(
        type="point",
        max_iter=max_iter,
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file,
        cache_input_file=cache_input_file,
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
    fit_and_render(
        type="mesh",
        max_iter=max_iter,
        n_frames=n_frames,
        image_size=image_size,
        output_file=output_file,
        cache_input_file=cache_input_file,
        flat_shading=True,
    )


"""
Helper functions
"""
def fit_and_render(
        type,
        output_file,
        max_iter=None,
        cache_input_file=None,
        n_frames=72,
        image_size=256,
        flat_shading=False,
):
    # load or fit (fitted, ground truth) pair
    if cache_input_file:
        fitted, gt = load_cache(cache_input_file)
    else:
        fitted, gt = fit_model(
            type=type,
            max_iter=max_iter,
            cache_output_file=output_file,
        )

    # render fitted (left) and ground truth (right) & save gif
    render_side_by_side(
        [fitted, gt],
        obj_type=type,
        output_file=output_file,
        n_frames=n_frames,
        image_size=image_size,
        flat_shading=flat_shading,
    )


def fit_model(
        type="vox",
        max_iter=None,
        cache_output_file="output/model.pt",
        test=False,
):
    # fit model
    parser = argparse.ArgumentParser(parents=[fit_data.get_args_parser()])
    arg_list = ["--type", type]
    if max_iter is not None:
        arg_list += ["--max_iter", str(max_iter)]
    if test:
        arg_list += ["--test"]
    args = parser.parse_args(arg_list)
    fitted, gt = fit_data.train_model(args)

    # cache (fitted, ground truth) so renders can be redone without refitting
    if cache_output_file:
        save_cache((fitted, gt), cache_output_file)

    return fitted, gt


def main():
    # fit_voxel(max_iter=10000)
    # fit_pointcloud(max_iter=20000)
    # fit_mesh(max_iter=15000)
    return


if __name__ == "__main__":
    main()
