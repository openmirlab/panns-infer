"""Compare extracted CNN14 stages to the committed unchanged-upstream fixtures.

Normalized waveform is compared independently of model execution. Float results
use explicit tolerances across Torch versions and exact top-ten class ordering.
Reads: CC0 fixtures, original activation fixtures and a verified checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from panns_infer._impl.model import Cnn14

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--exact',action='store_true')
a=p.parse_args()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model=Cnn14();model.load_checkpoint(a.checkpoint);model.to(a.device).eval()
root=Path(__file__).resolve().parents[1]/'tests/fixtures'
for f in json.loads((root/'manifest.json').read_text()):
    original=np.load(root/('baseline_'+a.device.split(':')[0])/(f['id']+'.npz'))
    wave=torch.from_numpy(original['waveform']).to(a.device)
    with torch.inference_mode():
        actual=model(wave)
        actual['logmel']=model.logmel_extractor(model.spectrogram_extractor(wave))
    for field, value in actual.items():
        x=value.cpu().numpy();y=original[field]
        if a.exact: np.testing.assert_array_equal(x,y)
        else: np.testing.assert_allclose(x,y,rtol=1e-4,atol=1e-4)
        print(f['id'],field,float(np.max(np.abs(x-y))))
    np.testing.assert_array_equal(np.argsort(-actual['clipwise_output'].cpu().numpy()[0])[:10],np.argsort(-original['clipwise_output'][0])[:10])
