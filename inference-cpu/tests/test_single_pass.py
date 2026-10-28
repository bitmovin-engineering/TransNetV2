import sys
from types import SimpleNamespace
import numpy as np
import pytest
from transnetv2_cpu.frames import FrameSpool
from transnetv2_cpu.models import FP32_POLICY, load_manifest, write_fp32_manifest
from transnetv2_cpu.single_pass import SinglePassVOD


@pytest.mark.parametrize('n',[1,73,949,950,951,1000,1900,1901])
def test_one_model_one_pass_and_both_head_ownership(tmp_path,monkeypatch,n):
    compiled=[];executions=[]
    class Request:
        def set_input_tensor(self,tensor):self.values=tensor
        def infer(self):executions.append(1)
        def get_tensor(self,name):
            ids=self.values[0,:,0,0,0]+256*self.values[0,:,0,0,1]
            return SimpleNamespace(data=ids if name=='output_1'else-ids-1)
    request=Request()
    class Core:
        def compile_model(self,path,device,config):
            compiled.append(path)
            return SimpleNamespace(create_infer_request=lambda:request)
    monkeypatch.setitem(sys.modules,'openvino',SimpleNamespace(Core=Core,Tensor=lambda x,shared_memory:x))
    for ext in ['.xml','.bin']:(tmp_path/('first-composed-f32'+ext)).write_bytes(ext.encode())
    write_fp32_manifest(tmp_path)
    pixels=np.zeros((n,27,48,3),np.uint8);pixels[:,0,0,0]=np.arange(n)%256;pixels[:,0,0,1]=np.arange(n)//256
    path=tmp_path/'input.rgb';pixels.tofile(path);spool=FrameSpool(path)
    engine=SinglePassVOD(tmp_path/'manifest.json',capture_windows=True)
    result=engine.predict(spool,tmp_path/'output');spool.close()
    actual=np.fromfile(tmp_path/'output/logits.f32',np.float32).reshape(2,n)
    np.testing.assert_array_equal(actual[0],np.arange(n));np.testing.assert_array_equal(actual[1],-np.arange(n)-1)
    assert len(compiled)==1 and len(executions)==(n+949)//950
    assert result['pilotExecutedFrames']==result['fastExecutedFrames']==result['refinementExecutedFrames']==0
    assert result['referenceExecutedFrames']==1000*len(executions)
    assert (tmp_path/'output/windows.f32').stat().st_size==8000*len(executions)
    assert not list((tmp_path/'output').glob('*.part'))


def test_fp32_contract_cannot_be_loaded_as_guarded_and_checks_bytes(tmp_path):
    for ext in ['.xml','.bin']:(tmp_path/('first-composed-f32'+ext)).write_bytes(ext.encode())
    write_fp32_manifest(tmp_path)
    with pytest.raises(ValueError,match='policy'):load_manifest(tmp_path/'manifest.json')
    assert load_manifest(tmp_path/'manifest.json',policy=FP32_POLICY)['policy']==FP32_POLICY
    (tmp_path/'first-composed-f32.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):load_manifest(tmp_path/'manifest.json',policy=FP32_POLICY)
