"""Entry point for the assignment code submission.

Running this file should reproduce every result referenced in the webpage
writeup (projX/assignment.md). Add whatever CLI flags/subcommands you need.
"""
import argparse
import os
import fit_data
import train_model
import eval_model
from starter.render import render_side_by_side
from starter.cache import save_cache, load_cache, get_checkpoint_path


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
2.1. Image to voxel grid (20 points)
"""
def train_voxel(
        max_iter=5000,
        pos_weight=None,
        retrain=False,
):
    model_args = [] if pos_weight is None else ["--pos_weight", str(pos_weight)]
    train_and_evaluate(
        type="vox",
        max_iter=max_iter,
        model_args=model_args,
        tag="vox" if pos_weight is None else f"vox_pw{pos_weight:g}",
        retrain=retrain,
    )


"""
2.2. Image to point cloud (20 points)
"""
def train_pointcloud(
        max_iter=5000,
        n_points=1000,
        retrain=False,
):
    train_and_evaluate(
        type="point",
        max_iter=max_iter,
        model_args=["--n_points", str(n_points)],
        tag="point" if n_points == 1000 else f"point_n{n_points}",
        retrain=retrain,
    )


"""
2.3. Image to mesh (20 points)
"""
def train_mesh(
        max_iter=5000,
        template="ico4",
        retrain=False,
):
    train_and_evaluate(
        type="mesh",
        max_iter=max_iter,
        model_args=["--template", template],
        tag="mesh" if template == "ico4" else f"mesh_{template}",
        retrain=retrain,
    )


"""
2.4. Analyse effects of hyperparams variations (10 points)
"""
def template_study(
        max_iter=5000,
        retrain=False,
):
    # mesh decoder initial shape: ico4 sphere (2.3 baseline) vs Q1.3 fitted chair vs torus
    for template in ["chair", "torus"]:
        train_mesh(max_iter=max_iter, template=template, retrain=retrain)


def class_imbalance_study(
        max_iter=5000,
        pos_weight=3,
        retrain=False,
):
    # voxel BCE with occupied voxels weighted pos_weight x (2.1 is the unweighted baseline)
    train_voxel(max_iter=max_iter, pos_weight=pos_weight, retrain=retrain)


"""
3.1. Implicit network (10 points)
"""
def train_implicit(
        max_iter=5000,
        n_query=2048,
        pos_weight=None,
        retrain=False,
):
    # occupancy MLP on (image feature, 3D point), queried on the 32^3 grid; its output
    # is a voxel grid, so it trains and evaluates through the vox pipeline
    model_args = ["--vox_decoder", "implicit"]
    if pos_weight is not None:
        model_args += ["--pos_weight", str(pos_weight)]
    train_and_evaluate(
        type="vox",
        max_iter=max_iter,
        model_args=model_args,
        train_only_args=["--n_query", str(n_query)],
        tag="vox_implicit" if pos_weight is None else f"vox_implicit_pw{pos_weight:g}",
        retrain=retrain,
    )


"""
3.2. Parametric network (10 points)
"""
def train_parametric(
        max_iter=5000,
        n_charts=10,
        n_points=1000,
        retrain=False,
):
    # atlas of 2D -> 3D chart MLPs conditioned on the image feature; its output is a
    # point cloud, so it trains and evaluates through the point pipeline
    train_and_evaluate(
        type="point",
        max_iter=max_iter,
        model_args=["--point_decoder", "parametric", "--n_charts", str(n_charts),
                    "--n_points", str(n_points)],
        tag="point_parametric" if n_charts == 10 else f"point_parametric_k{n_charts}",
        retrain=retrain,
    )


"""
Helper functions
"""
def train_and_evaluate(
        type,
        max_iter,
        model_args=(),
        train_only_args=(),
        tag=None,
        retrain=False,
        vis_freq=200,
):
    # model_args define the model (template, n_points, pos_weight, vox_decoder), so
    # training and evaluation both get them and agree on the model and its checkpoint
    # file; train_only_args are training-only settings (e.g. --n_query)
    common_args = ["--type", type, "--load_feat", *model_args]

    # train, unless this model's checkpoint already exists
    train_parser = argparse.ArgumentParser(parents=[train_model.get_args_parser()])
    train_args = train_parser.parse_args(
        common_args + ["--max_iter", str(max_iter), "--save_freq", "500", *train_only_args]
    )
    if retrain or not os.path.exists(get_checkpoint_path(train_args)):
        train_model.train_model(train_args)

    # evaluate: F1 plot to eval_{tag}.png, example GIFs to vis/{step}_{tag}.gif
    eval_parser = argparse.ArgumentParser(parents=[eval_model.get_args_parser()])
    eval_args = eval_parser.parse_args(
        common_args + ["--load_checkpoint", "--vis_freq", str(vis_freq), "--tag", tag or type]
    )
    os.makedirs("vis", exist_ok=True)
    eval_model.evaluate_model(eval_args)


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
    # 1. Exploring loss functions
    fit_voxel(max_iter=10000)
    fit_pointcloud(max_iter=20000)
    fit_mesh(max_iter=15000)  # also the 2.4 chair template (output/fit_mesh.pt)

    # 2.1-2.3 Reconstructing 3D from single view (all with --load_feat, CPU)
    train_voxel()
    train_pointcloud()
    train_mesh()

    # 2.4 Hyperparameter analysis
    template_study()
    class_imbalance_study()

    # 3.1 Implicit network
    train_implicit()

    # 3.2 Parametric network
    train_parametric()
    return


if __name__ == "__main__":
    main()
