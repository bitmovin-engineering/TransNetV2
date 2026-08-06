import os
import torch
import argparse
import numpy as np
import tensorflow as tf

import transnetv2_pytorch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def remap_name(x):
    x = x.replace("TransNet/", "")
    l = []
    for a in x.split("/"):
        if a.startswith("SDDCNN") or a.startswith("DDCNN"):
            a = a.split("_")
            a = a[0] + "." + str(int(a[1]) - 1)
        elif a == "conv_spatial":
            a = "layers.0"
        elif a == "conv_temporal":
            a = "layers.1"
        elif a == "kernel:0" or a == "gamma:0":
            a = "weight"
        elif a == "bias:0" or a == "beta:0":
            a = "bias"
        elif a == "dense":
            a = "fc1"
        elif a == "dense_1":
            a = "cls_layer1"
        elif a == "dense_2":
            a = "cls_layer2"
        elif a == "dense_3":
            a = "frame_sim_layer.projection"
        elif a == "dense_4":
            a = "frame_sim_layer.fc"
        elif a == "dense_5":
            a = "color_hist_layer.fc"
        elif a == "FrameSimilarity" or a == "ColorHistograms":
            a = ""
        elif a == "moving_mean:0":
            a = "running_mean"
        elif a == "moving_variance:0":
            a = "running_var"
        l.append(a)
    x = ".".join([a for a in l if a != ""])
    return x


def remap_tensor(x):
    x = x.numpy()
    if len(x.shape) == 5:
        x = np.transpose(x, [0, 1, 2, 4, 3])
        x = np.transpose(x, [3, 4, 0, 1, 2])
    elif len(x.shape) == 2:
        x = np.transpose(x)
    return torch.from_numpy(x).clone()


def check_and_fix_dicts(tf_dict, torch_dict):
    error = False

    for k in torch_dict.keys():
        if k not in tf_dict:
            if k.endswith("num_batches_tracked"):
                tf_dict[k] = torch.tensor(1., dtype=torch.float32)
            else:
                print("!", k, "missing in TF")
                error = True

    for k in tf_dict.keys():
        if k not in torch_dict:
            print("!", k, "missing in TORCH")
            error = True
        if tuple(tf_dict[k].shape) != torch_dict[k]:
            print("!", k, f"has wrong shape (TF: {tuple(tf_dict[k].shape)}, TORCH: {torch_dict[k]})")
            error = True

    return not error


def fuse_spatial_weights(tf_dict):
    """Combine the four branch spatial kernels while preserving branch order."""
    dilations = (1, 2, 4, 8)
    fused_dict = dict(tf_dict)

    for stack_index in range(3):
        for block_index in range(2):
            prefix = f"SDDCNN.{stack_index}.DDCNN.{block_index}"
            spatial_keys = [
                f"{prefix}.Conv3D_{dilation}.layers.0.weight" for dilation in dilations
            ]
            fused_dict[f"{prefix}.conv_spatial.weight"] = torch.cat(
                [tf_dict[key] for key in spatial_keys], dim=0
            )
            for branch_index, dilation in enumerate(dilations):
                for parameter in ("weight", "bias"):
                    old_key = f"{prefix}.Conv3D_{dilation}.layers.1.{parameter}"
                    new_key = f"{prefix}.conv_temporal.{branch_index}.{parameter}"
                    if old_key in tf_dict:
                        fused_dict[new_key] = tf_dict[old_key]
            for key in spatial_keys:
                del fused_dict[key]
            for dilation in dilations:
                for parameter in ("weight", "bias"):
                    key = f"{prefix}.Conv3D_{dilation}.layers.1.{parameter}"
                    if key in fused_dict:
                        del fused_dict[key]

    return fused_dict


def convert_weights(tf_weights_dir, fuse_spatial_convolutions=False):
    tf_model = tf.saved_model.load(tf_weights_dir)
    tf_dict = {remap_name(v.name): remap_tensor(v) for v in tf_model.variables}
    if fuse_spatial_convolutions:
        tf_dict = fuse_spatial_weights(tf_dict)

    torch_model = transnetv2_pytorch.TransNetV2(
        fuse_spatial_convolutions=fuse_spatial_convolutions,
    )
    torch_dict = {k: tuple(v.shape) for k, v in list(torch_model.named_parameters()) + list(torch_model.named_buffers())}

    assert check_and_fix_dicts(tf_dict, torch_dict), "some errors occurred when converting"
    torch_model.load_state_dict(tf_dict)

    return torch_model, tf_model


def test_models(torch_model, tf_model):
    random = np.random.default_rng(0)
    input_tensors = [random.integers(0, 255, size=(2, 100, 27, 48, 3), dtype=np.uint8) for _ in range(10)]

    print("Tests: computing forward passes...")
    with torch.no_grad():
        torch_outputs = [torch_model(torch.from_numpy(x)) for x in input_tensors]
    tf_outputs = [tf_model(tf.cast(x, tf.float32)) for x in input_tensors]

    for i, ((torch_single, torch_many), (tf_single, tf_many)) in enumerate(zip(torch_outputs, tf_outputs)):
        single = np.isclose(torch_single.numpy(), tf_single.numpy()).mean()
        many = np.isclose(torch_many["many_hot"].numpy(), tf_many["many_hot"].numpy()).mean()

        print(f"Test {i:2d}: "
              f"{single * 100:5.1f}% of 'single' predictions matching, "
              f"{many * 100:5.1f}% of 'many' predictions matching")
        np.testing.assert_allclose(torch_single.numpy(), tf_single.numpy(), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            torch_many["many_hot"].numpy(), tf_many["many_hot"].numpy(), rtol=1e-5, atol=1e-8
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_weights", type=str, help="path to TransNet V2 weights",
                        default="../inference/transnetv2-weights/")
    parser.add_argument('--test', action="store_true", help="run tests")
    parser.add_argument('--fuse-spatial-convolutions', action="store_true",
                        help="fuse the four parallel spatial convolutions in each DDCNN block")
    args = parser.parse_args()

    torch_model, tf_model = convert_weights(
        args.tf_weights,
        fuse_spatial_convolutions=args.fuse_spatial_convolutions,
    )
    output_path = (
        "./transnetv2-pytorch-weights-fused.pth"
        if args.fuse_spatial_convolutions
        else "./transnetv2-pytorch-weights.pth"
    )

    if args.test:
        test_models(torch_model, tf_model)

    print(f"Saving model to {output_path}")
    torch.save(torch_model.state_dict(), output_path)


if __name__ == "__main__":
    main()
