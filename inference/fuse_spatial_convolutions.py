"""Create an opt-in SavedModel with fused DDCNN spatial convolutions.

The deployed TransNetV2 graph has four parallel (1, 3, 3) convolutions in
each DDCNN block. Their filters can be concatenated on the output-channel
axis, evaluated by one wider convolution, and split back before the four
temporal branches. This preserves the branch tensors up to floating-point
accumulation order while reducing the number of spatial Conv3D operations.

This is an export-time rewrite. The input model directory is copied to a new
output directory; variables are not changed.
"""

import argparse
import re
import shutil
from pathlib import Path

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import saved_model_pb2


_SPATIAL_NODE = re.compile(
    r"(.*/SDDCNN_[1-3]/DDCNN_[1-2])/Conv3D_(1|2|4|8)/conv_spatial/Conv3D$")
_DILATIONS = (1, 2, 4, 8)


def _scalar_int(value):
    tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_INT32)
    tensor.int_val.append(value)
    return tensor


def _type_attr(data_type):
    return attr_value_pb2.AttrValue(type=data_type)


def _int_attr(value):
    return attr_value_pb2.AttrValue(i=value)


def _replace_input(value, replacements):
    for old_name, new_name in replacements.items():
        if value in (old_name, old_name + ":output:0", old_name + ":0"):
            return new_name
    return value


def _rewired(node, replacements):
    result = node_def_pb2.NodeDef()
    result.CopyFrom(node)
    del result.input[:]
    result.input.extend(_replace_input(value, replacements) for value in node.input)
    return result


def _fused_nodes(prefix, branches):
    first = branches[1]
    filter_inputs = [branches[dilation].input[1] for dilation in _DILATIONS]
    axis_name = prefix + "/fused_spatial/axis"
    weights_name = prefix + "/fused_spatial/weights"
    split_name = prefix + "/fused_spatial/split"

    axis = node_def_pb2.NodeDef(name=axis_name, op="Const")
    axis.attr["dtype"].CopyFrom(_type_attr(types_pb2.DT_INT32))
    axis.attr["value"].tensor.CopyFrom(_scalar_int(4))

    weights = node_def_pb2.NodeDef(
        name=weights_name,
        op="ConcatV2",
        input=filter_inputs + [axis_name + ":output:0"],
    )
    weights.attr["N"].CopyFrom(_int_attr(4))
    weights.attr["T"].CopyFrom(_type_attr(types_pb2.DT_FLOAT))
    weights.attr["Tidx"].CopyFrom(_type_attr(types_pb2.DT_INT32))

    fused = node_def_pb2.NodeDef()
    fused.CopyFrom(first)
    del fused.input[:]
    fused.input.extend([first.input[0], weights_name + ":output:0"])
    # The original node records the narrow branch shape. TensorFlow Python
    # validates this metadata against the widened filter at import time.
    fused.attr.pop("_output_shapes", None)

    split = node_def_pb2.NodeDef(
        name=split_name,
        op="Split",
        input=[axis_name + ":output:0", first.name + ":output:0"],
    )
    split.attr["T"].CopyFrom(_type_attr(types_pb2.DT_FLOAT))
    split.attr["num_split"].CopyFrom(_int_attr(4))

    return [axis, weights, fused, split], split_name


def _fuse_function(function_def):
    conv3d_count = sum(node.op == "Conv3D" for node in function_def.node_def)
    spatial = {}
    for node in function_def.node_def:
        match = _SPATIAL_NODE.fullmatch(node.name)
        if match and node.op == "Conv3D":
            spatial.setdefault(match.group(1), {})[int(match.group(2))] = node

    if (conv3d_count != 48 or len(spatial) != 6 or
            any(set(branches) != set(_DILATIONS) for branches in spatial.values())):
        return function_def, False

    replacements = {}
    removed = set()
    insertions = {}
    nodes = list(function_def.node_def)
    for prefix, branches in spatial.items():
        for dilation in _DILATIONS:
            if len(branches[dilation].input) < 2:
                raise ValueError("spatial Conv3D has fewer than two inputs: " +
                                 branches[dilation].name)

        additions, split_name = _fused_nodes(prefix, branches)
        first_name = branches[1].name
        first_index = next(index for index, node in enumerate(nodes)
                           if node.name == first_name)
        insertions.setdefault(first_index, []).extend(additions)
        for branch_index, dilation in enumerate(_DILATIONS):
            branch_name = branches[dilation].name
            replacements[branch_name] = split_name + ":output:" + str(branch_index)
            removed.add(branch_name)

    result = function_def.__class__()
    result.CopyFrom(function_def)
    del result.node_def[:]
    for index, node in enumerate(nodes):
        for addition in insertions.get(index, ()):
            # New nodes already contain their final inputs. Applying the
            # replacement map to them would make Split consume itself.
            result.node_def.add().CopyFrom(addition)
        if node.name not in removed:
            result.node_def.add().CopyFrom(_rewired(node, replacements))

    for key, value in list(result.ret.items()):
        result.ret[key] = _replace_input(value, replacements)
    return result, True


def fuse_saved_model(model):
    changed = 0
    for meta_graph in model.meta_graphs:
        functions = list(meta_graph.graph_def.library.function)
        del meta_graph.graph_def.library.function[:]
        for function_def in functions:
            transformed, was_changed = _fuse_function(function_def)
            meta_graph.graph_def.library.function.add().CopyFrom(transformed)
            changed += int(was_changed)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="source SavedModel directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory to create for the fused SavedModel")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not (input_dir / "saved_model.pb").is_file():
        parser.error("--input must contain saved_model.pb")
    if output_dir.exists():
        parser.error("--output already exists; choose a new directory")

    model = saved_model_pb2.SavedModel()
    with (input_dir / "saved_model.pb").open("rb") as source:
        model.ParseFromString(source.read())
    changed = fuse_saved_model(model)
    if changed != 1:
        raise RuntimeError("expected exactly one deployed function to change")

    shutil.copytree(input_dir, output_dir)
    with (output_dir / "saved_model.pb").open("wb") as destination:
        destination.write(model.SerializeToString())
    print("Changed functions: {}".format(changed))


if __name__ == "__main__":
    main()
