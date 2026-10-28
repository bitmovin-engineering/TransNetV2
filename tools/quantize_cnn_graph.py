"""Quantize only the graph upstream of three CNN stage AvgPool boundaries.

Uses topology rather than semantic names, which can disappear during conversion.
Bias correction is disabled. Quantizer placement is audited after calibration.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np
import nncf
import openvino as ov

def cnn_region(model):
    pools=[n for n in model.get_ops() if n.get_type_name()=='AvgPool'];assert len(pools)==3
    allowed=set();stack=list(pools)
    while stack:
        node=stack.pop()
        if node.get_name() in allowed:continue
        allowed.add(node.get_name());stack.extend(v.get_node() for v in node.input_values())
    return allowed

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('rgb',type=Path,nargs='+');p.add_argument('--frames',type=int,default=100);p.add_argument('--offsets',type=int,nargs='+');p.add_argument('--restore-blocks',type=int,nargs='+',default=[]);a=p.parse_args();assert not a.destination.exists()
    samples=[];sources=[]
    for path in a.rgb:
        raw=np.fromfile(path,np.uint8).reshape(-1,27,48,3);assert len(raw)>=a.frames
        offsets=a.offsets or np.linspace(0,len(raw)-a.frames,4,dtype=int).tolist()
        for i in offsets:assert 0<=i and i+a.frames<=len(raw);samples.append(raw[i:i+a.frames].astype(np.float32)[None])
        sources.append({'path':str(path),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'offsets':offsets})
    m=ov.Core().read_model(a.source);allowed=cnn_region(m)
    restored=set();previous=set();blocks=[]
    for boundary in m.get_ordered_ops():
        if boundary.get_type_name()!='Concat' or boundary.get_input_size()!=4 or boundary.get_output_partial_shape(0).rank.get_length()!=5 or boundary.get_name() not in allowed:continue
        ancestors=set();stack=[boundary]
        while stack:
            node=stack.pop()
            if node.get_name() in ancestors:continue
            ancestors.add(node.get_name());stack.extend(v.get_node() for v in node.input_values())
        current=ancestors-previous;blocks.append(current);previous=ancestors
    assert len(blocks)==6
    for index in a.restore_blocks:assert 0<=index<6;restored.update(blocks[index])
    restored_convs={n.get_friendly_name():n.input_value(1).get_node().get_data().copy() for n in m.get_ops() if n.get_name() in restored and n.get_type_name()=='Convolution'}
    forbidden=[n.get_friendly_name() for n in m.get_ops() if (n.get_name() not in allowed or n.get_name() in restored) and n.get_type_name() not in ['Constant','Parameter','Result']]
    patterns=[r'(^|\n)'+re.escape(name)+r'(\n|$)' for name in forbidden]
    protocol={'scope':__doc__,'source_xml_sha256':hashlib.file_digest(a.source.open('rb'),'sha256').hexdigest(),'source_bin_sha256':hashlib.file_digest(a.source.with_suffix('.bin').open('rb'),'sha256').hexdigest(),'calibration':sources,'frames':a.frames,'disable_bias_correction':True,'restore_blocks':a.restore_blocks,'restored_convs':list(restored_convs),'forbidden_aux_nodes':forbidden,'preset':'MIXED','nncf':nncf.__version__,'openvino':ov.__version__}
    a.destination.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');start=time.perf_counter()
    q=nncf.quantize(m,nncf.Dataset(samples),preset=nncf.QuantizationPreset.MIXED,target_device=nncf.TargetDevice.CPU,subset_size=len(samples),ignored_scope=nncf.IgnoredScope(patterns=patterns,types=['MatMul'],validate=False),advanced_parameters=nncf.AdvancedQuantizationParameters(disable_bias_correction=True))
    region=cnn_region(q);quantizers=[];bad=[]
    checked=set()
    for n in q.get_ops():
        if n.get_type_name()=='Convolution' and n.get_friendly_name() in restored_convs:
            w=n.input_value(1).get_node();assert w.get_type_name()=='Constant' and np.array_equal(w.get_data().view(np.uint32),restored_convs[n.get_friendly_name()].view(np.uint32));checked.add(n.get_friendly_name())
    assert checked==set(restored_convs),'Restored convolution weights changed or disappeared'
    for n in q.get_ops():
        if n.get_type_name()!='FakeQuantize':continue
        row={'name':n.get_friendly_name(),'source':n.input_value(0).get_node().get_friendly_name(),'in_CNN_region':n.get_name() in region};quantizers.append(row)
        if not row['in_CNN_region']:bad.append(row)
    audit={'quantizers':quantizers,'outside_CNN':bad,'seconds':time.perf_counter()-start};a.destination.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n');assert not bad,'Quantizers escaped CNN boundary'
    ov.save_model(q,a.destination,compress_to_fp16=False);print(json.dumps({'quantizers':len(quantizers),'outside_CNN':len(bad),'seconds':audit['seconds']},indent=2))

if __name__=='__main__':main()
