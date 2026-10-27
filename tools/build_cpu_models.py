"""Reproduce the guarded CPU model pair using explicit, local calibration inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'inference-cpu/src'))
from transnetv2_cpu.models import SOURCE, write_manifest, write_fp32_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--calibration-rgb',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for name,expected in SOURCE.items():
        with (args.weights/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Unexpected source weights: '+name)
    args.output.mkdir(parents=True,exist_ok=False)
    def run(script,*parameters):
        subprocess.run([sys.executable,str(ROOT/'tools'/script),*map(str,parameters)],check=True)
    run('export_folded_bn_model.py',args.weights,args.output/'folded')
    run('convert_openvino_scatter.py',args.output/'folded',args.output/'folded-f32.xml')
    run('export_composed_conv_model.py',args.weights,args.output/'first-composed','--first-only')
    run('convert_openvino_scatter.py',args.output/'first-composed',args.output/'first-composed-f32.xml')
    run('quantize_cnn_graph.py',args.output/'folded-f32.xml',args.output/'folded8.xml',*args.calibration_rgb)
    write_fp32_manifest(args.output)
    (args.output/'manifest.json').rename(args.output/'fp32-manifest.json')
    write_manifest(args.output)
    print(args.output/'manifest.json')


if __name__=='__main__':main()
