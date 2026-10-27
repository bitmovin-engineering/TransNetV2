"""Compose each linear spatial+temporal pair into one 3x3x3 convolution.

No activation lies between the source pair. Spatial-only then temporal-only
SAME zero padding is mathematically the same as the composed kernel's padding.
Floating-point evaluation changes; measured quality gates remain mandatory.
This trades extra multiply-adds for fewer large intermediate tensors/passes.
"""
import argparse,hashlib,json
from pathlib import Path
from export_folded_bn_model import Folded,ORIGINAL,tf
import numpy as np

class Composed(Folded):
    def __init__(self,model,first_only=False):
        super().__init__(model)
        self.first_only=first_only
        self.composed=[]
        for block in self.kernels:
            branches=[]
            for spatial,temporal,bias in block:
                s=spatial.numpy()[0]; t=temporal.numpy()[:,0,0]
                kernel=np.einsum('hwim,tmo->thwio',s,t,optimize=True,dtype=np.float32)
                branches.append((tf.constant(kernel),bias))
            self.composed.append(branches)
    @tf.function(input_signature=[tf.TensorSpec([None,None,27,48,3],tf.float32,name='input_1')],autograph=False)
    def serve(self,input_1):
        x=input_1/255.; features=[]
        for stage in range(3):
            shortcut=None
            for block in range(2):
                branches=[]
                for branch,dilation in enumerate([1,2,4,8]):
                    if not self.first_only or stage==0 and block==0:
                        kernel,bias=self.composed[2*stage+block][branch]
                        y=tf.nn.conv3d(x,kernel,[1]*5,'SAME',data_format='NDHWC',dilations=[1,dilation,1,1,1])
                    else:
                        spatial,temporal,bias=self.kernels[2*stage+block][branch]
                        y=tf.nn.conv3d(x,spatial,[1]*5,'SAME',data_format='NDHWC')
                        y=tf.nn.conv3d(y,temporal,[1]*5,'SAME',data_format='NDHWC',dilations=[1,dilation,1,1,1])
                    branches.append(tf.nn.relu(tf.nn.bias_add(y,bias)))
                x=tf.concat(branches,axis=-1)
                if shortcut is None:shortcut=x
            x=x+shortcut
            x=tf.nn.avg_pool3d(x,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='VALID');features.append(x)
        flat=tf.reshape(x,[tf.shape(input_1)[0],tf.shape(input_1)[1],3*6*256])
        hidden=tf.concat([self.model.color_hist_layer(input_1),self.model.frame_sim_layer(features),flat],axis=2)
        hidden=self.model.fc1(hidden)
        return {'output_1':self.model.cls_layer1(hidden),'output_2':self.model.cls_layer2(hidden)}

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--first-only',action='store_true');a=p.parse_args();assert not a.destination.exists()
    assert hashlib.file_digest((a.source/'saved_model.pb').open('rb'),'sha256').hexdigest()==ORIGINAL
    original=tf.saved_model.load(str(a.source)); composed=Composed(original,a.first_only)
    tf.saved_model.save(composed,str(a.destination),signatures={'serving_default':composed.serve.get_concrete_function()})
    print(json.dumps({'source_pb_sha256':ORIGINAL,'composed_convolutions':4 if a.first_only else 24,'BN_folded':True,'full_model_sha256':hashlib.file_digest((a.destination/'saved_model.pb').open('rb'),'sha256').hexdigest()},indent=2))

if __name__=='__main__':main()
