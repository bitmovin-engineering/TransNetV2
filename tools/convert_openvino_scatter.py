"""Convert TF SavedModel to FP32 OV and replace its single histogram segment sum.

The integer scatter rewrite needs exact count validation on the converted graph.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import openvino as ov

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();assert not a.destination.exists()
    start=time.perf_counter();unpatched=a.destination.with_suffix('.unpatched.xml')
    if unpatched.exists():m=ov.Core().read_model(unpatched)
    else:
        m=ov.convert_model(str(a.source));ov.save_model(m,unpatched,compress_to_fp16=False)
    matches=[n for n in m.get_ops() if n.get_type_name()=='EmbeddingSegmentsSum'];assert len(matches)==1
    old=matches[0]
    print('embedding_inputs',[(v.get_node().get_friendly_name(),v.get_node().get_type_name(),str(v.get_partial_shape()),v.get_node().get_data().tolist() if v.get_node().get_type_name()=='Constant' and v.get_node().get_data().size<16 else None) for v in old.input_values()],flush=True)
    # Target TransNet's known integer histogram, before the frontend's TopK sort.
    # Scattering already sorted embedding inputs would retain that expensive sort.
    prefix=old.get_friendly_name().removesuffix('UnsortedSegmentSum');assert prefix!=old.get_friendly_name()
    nodes={n.get_friendly_name():n for n in m.get_ops()};ids_node=nodes[prefix+'add_2'];ones=nodes[prefix+'ones_like'];segments=nodes[prefix+'mul_4']
    assert ids_node.get_output_partial_shape(0).rank.get_length()==2 and ones.get_output_partial_shape(0).rank.get_length()==2
    assert ids_node.get_output_element_type(0)==ones.get_output_element_type(0)==ov.Type.i32
    assert segments.get_output_partial_shape(0).rank.get_length()==0
    ops=ov.opset15;ids=ops.reshape(ids_node,ops.constant([-1],np.int64),False);updates=ops.reshape(ones,ops.constant([-1],np.int64),False)
    count=ops.reshape(ops.convert(segments,'i64'),ops.constant([1],np.int64),False);zero=ops.broadcast(ops.constant(0,np.int32),count)
    scatter=ops.scatter_elements_update(zero,ids,updates,ops.constant(0,np.int64),reduction='sum',use_init_val=True);scatter.set_friendly_name('research/histogram_scatter_sum');scatter.output(0).get_tensor().set_names(old.output(0).get_names());old.output(0).replace(scatter.output(0));m.validate_nodes_and_infer_types()
    ov.save_model(m,a.destination,compress_to_fp16=False);print(json.dumps({'seconds':time.perf_counter()-start,'replaced':old.get_friendly_name(),'output_keys':[sorted(x.get_names())for x in m.outputs]},indent=2))

if __name__=='__main__':main()
