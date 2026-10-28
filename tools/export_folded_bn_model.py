"""Inference-only BN folding + direct dilated temporal Conv3D prototype.

This changes floating-point arithmetic. Boundary/quality checks are mandatory.
Targets only the SHA-pinned original architecture (F16/L3/S2, residuals,
stochastic depth0, avg pooling). ReLU commutes with channel concatenation.
"""
import argparse, hashlib, json, os
from pathlib import Path
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import tensorflow as tf
ORIGINAL='8ac2a52c5719690d512805b6eaf5ce12097c1d8860b3d9de245dcbbc3100f554'

class Folded(tf.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
        variables={v.name:v for v in model.variables}; self.kernels=[]
        for stage in range(1,4):
            for block in range(1,3):
                prefix=f'TransNet/SDDCNN_{stage}/DDCNN_{block}/'; channels=16*2**(stage-1)
                gamma=variables[prefix+'bn/gamma:0']; beta=variables[prefix+'bn/beta:0']
                mean=variables[prefix+'bn/moving_mean:0']; variance=variables[prefix+'bn/moving_variance:0']
                assert gamma.shape==[channels*4]
                scale=gamma*tf.math.rsqrt(variance+tf.constant(1e-3,tf.float32))
                offset=beta-mean*scale; branches=[]
                for i,dilation in enumerate([1,2,4,8]):
                    spatial=variables[prefix+f'Conv3D_{dilation}/conv_spatial/kernel:0']
                    temporal=variables[prefix+f'Conv3D_{dilation}/conv_temporal/kernel:0']
                    assert spatial.shape[:3]==[1,3,3] and temporal.shape[:3]==[3,1,1] and temporal.shape[-1]==channels
                    s=scale[i*channels:(i+1)*channels]; b=offset[i*channels:(i+1)*channels]
                    branches.append((tf.constant(spatial.numpy()),tf.constant((temporal*s).numpy()),tf.constant(b.numpy())))
                self.kernels.append(branches)
    @tf.function(input_signature=[tf.TensorSpec([None,None,27,48,3],tf.float32,name='input_1')],autograph=False)
    def serve(self,input_1):
        x=input_1/255.; features=[]
        for stage in range(3):
            shortcut=None
            for block in range(2):
                branches=[]
                for dilation,(spatial,temporal,bias) in zip([1,2,4,8],self.kernels[2*stage+block]):
                    y=tf.nn.conv3d(x,spatial,[1]*5,'SAME',data_format='NDHWC')
                    y=tf.nn.conv3d(y,temporal,[1]*5,'SAME',data_format='NDHWC',dilations=[1,dilation,1,1,1])
                    y=tf.nn.relu(tf.nn.bias_add(y,bias))
                    branches.append(y)
                x=tf.concat(branches,axis=-1)
                if shortcut is None: shortcut=x
            x=x+shortcut
            x=tf.nn.avg_pool3d(x,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='VALID')
            features.append(x)
        flat=tf.reshape(x,[tf.shape(input_1)[0],tf.shape(input_1)[1],3*6*256])
        hidden=tf.concat([self.model.color_hist_layer(input_1),self.model.frame_sim_layer(features),flat],axis=2)
        hidden=self.model.fc1(hidden)
        return {'output_1':self.model.cls_layer1(hidden),'output_2':self.model.cls_layer2(hidden)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('destination',type=Path); a=p.parse_args(); assert not a.destination.exists()
    assert hashlib.file_digest((a.source/'saved_model.pb').open('rb'),'sha256').hexdigest()==ORIGINAL
    m=tf.saved_model.load(str(a.source)); folded=Folded(m)
    tf.saved_model.save(folded,str(a.destination),signatures={'serving_default':folded.serve.get_concrete_function()})
    print(json.dumps({'folded_bn_layers':6,'direct_temporal_convolutions':24,'relu_before_concat':True,'learned_arithmetic_changed':True,'model_sha256':hashlib.file_digest((a.destination/'saved_model.pb').open('rb'),'sha256').hexdigest()},indent=2))

if __name__=='__main__': main()
