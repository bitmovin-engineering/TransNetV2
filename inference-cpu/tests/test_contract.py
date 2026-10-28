import io
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
import pytest
from transnetv2_cpu.engine import hard_regions, soft_regions, merge, merged_stream
from transnetv2_cpu.frames import FrameSpool, FRAME_BYTES
from transnetv2_cpu.models import load_manifest, write_manifest


def original_soft(values):
    n = len(values)
    p = 1/(1+np.exp(-values.astype(np.float64)))
    above = p >= .5
    onsets = np.flatnonzero(above & ~np.pad(above[:-1], (1,0)))
    exclusion = np.zeros(n, bool)
    for i in onsets:
        if i != 0:
            exclusion[max(0,i-30):min(n,i+31)] = True
    rolling = np.convolve(p,np.ones(31),'full')[15:15+n]
    from transnetv2_cpu.engine import runs
    return merge((max(0,a-64),min(n,b+64))for a,b in runs((rolling>.03)&(p<.5)&~exclusion)if np.max(p[a:b])>=.015)


@pytest.mark.parametrize('n',[1,49,50,73,949,950,951,975,1000,1900,1901])
def test_window_ownership_and_eof(tmp_path,n):
    raw = np.arange(n,dtype=np.uint16)
    pixels = np.zeros((n,27,48,3),np.uint8)
    pixels[:,0,0,0] = raw%256
    pixels[:,0,0,1] = raw//256
    path=tmp_path/'frames.rgb';pixels.tofile(path)
    spool=FrameSpool(path)
    assert spool.digest.hexdigest()==hashlib.sha256(pixels.tobytes()).hexdigest()
    tail=25+950-(n%950 if n%950 else 950)
    expected=np.pad(pixels,((25,tail),(0,0),(0,0),(0,0)),mode='edge')
    owned=[]
    try:
        for i in range((n+949)//950):
            window=spool.window(i)[0]
            np.testing.assert_array_equal(window,expected[i*950:i*950+1000])
            owned.append(window[25:25+min(950,n-i*950)])
        np.testing.assert_array_equal(np.concatenate(owned),pixels)
    finally:spool.close()


@pytest.mark.parametrize('chunk',[1,7,31,64,128,65536])
def test_query_streams_preserve_global_decisions(chunk):
    rng=np.random.default_rng(42)
    x=rng.normal(-6,2,(2,1003)).astype(np.float32)
    x[0,0:4]=-1;x[0,64:81]=-3;x[0,125:130]=2;x[0,999:]=-.1
    from transnetv2_cpu.engine import runs
    expected=[(max(0,a-2),min(x.shape[1],b+2))for a,b in runs(np.any(np.abs(x)<=1.5,axis=0))]
    assert list(hard_regions(x,x.shape[1],chunk))==expected
    assert list(merged_stream(soft_regions(x[0],x.shape[1],chunk)))==original_soft(x[0])


def test_stream_receiver_validates_truncation_and_quota(tmp_path):
    for name,data,limit in [('short',b'x',FRAME_BYTES*2),('quota',bytes(FRAME_BYTES*3),FRAME_BYTES*2)]:
        spool=FrameSpool(tmp_path/name,io.BytesIO(data),max_bytes=limit)
        with pytest.raises(ValueError):spool.await_frames(1000)
        spool.close()
        assert not spool.thread.is_alive()


def test_manifest_binds_model_bytes_and_policy(tmp_path):
    for stem in ['folded8','first-composed-f32']:
        for suffix in ['.xml','.bin']:(tmp_path/(stem+suffix)).write_bytes((stem+suffix).encode())
    manifest=write_manifest(tmp_path)
    assert load_manifest(tmp_path/'manifest.json')==manifest
    (tmp_path/'folded8.bin').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='checksum'):load_manifest(tmp_path/'manifest.json')


def test_stream_checksum_and_eof_are_committed_together(tmp_path):
    data=bytes(np.random.default_rng(3).integers(0,256,FRAME_BYTES*1001,np.uint8))
    spool=FrameSpool(tmp_path/'stream.rgb',io.BytesIO(data))
    assert not spool.await_frames(1002)
    assert spool.frames==1001 and spool.digest.hexdigest()==hashlib.sha256(data).hexdigest()
    spool.close()
    assert not spool.thread.is_alive()
