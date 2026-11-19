#!/usr/bin/env python3
"""Check that an ONNX file has the inputs and outputs DeepCoreSeedGeneratorONNX expects, and (optionally)
that it gives the same outputs as the PyTorch model on the same input.

  python3 check_onnx_phase1.py DeepCore_phase1.onnx
  python3 check_onnx_phase1.py DeepCore_phase1.onnx --torch-script model.pt     (a torch.jit-saved model)

Expected (as the TensorFlow model in CMSSW):
  inputs   input_1 [N,1] jet eta, input_2 [N,1] jet pT, input_3 [N,30,30,4] pixel image (NHWC)
  outputs  output_node0 [N,30,30,3,>=5] track parameters, output_node1 [N,30,30,3,>=1] probability
"""
import argparse
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("--inputs", nargs=3, default=["input_1", "input_2", "input_3"])
ap.add_argument("--outputs", nargs=2, default=["output_node0", "output_node1"])
ap.add_argument("--torch-script", default=None, help="torch.jit model taking (eta, pt, image) to compare with")
a = ap.parse_args()

s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
print("inputs :", [(i.name, i.shape) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in s.get_outputs()])
ok = True
names_in = [i.name for i in s.get_inputs()]
names_out = [o.name for o in s.get_outputs()]
for n in a.inputs:
    if n not in names_in:
        print("MISSING input", n); ok = False
for n in a.outputs:
    if n not in names_out:
        print("MISSING output", n); ok = False
if not ok:
    raise SystemExit("FAIL: set inputTensorName / outputTensorName of the module to the names above, "
                     "in the order (eta, pt, image) and (parameters, probability)")

rng = np.random.default_rng(1)
N = 2
eta = rng.uniform(-1.4, 1.4, (N, 1)).astype(np.float32)
pt = rng.uniform(500, 1500, (N, 1)).astype(np.float32)
img = np.zeros((N, 30, 30, 4), np.float32)
m = rng.random((N, 30, 30, 4)) < 0.03
img[m] = rng.exponential(1.0, m.sum()).astype(np.float32)
par, prob = s.run(a.outputs, dict(zip(a.inputs, [eta, pt, img])))
print("output shapes:", par.shape, prob.shape)
if par.shape[:4] != (N, 30, 30, 3) or prob.shape[:4] != (N, 30, 30, 3) or par.shape[4] < 5:
    raise SystemExit("FAIL: outputs must be [N,30,30,3,>=5] and [N,30,30,3,>=1]")
print("probability range %.3g .. %.3g, cells above 0.7: %d" % (prob[..., 0].min(), prob[..., 0].max(),
                                                             int((prob[..., 0] > 0.7).sum())))
if a.torch_script:
    import torch
    mdl = torch.jit.load(a.torch_script).eval()
    with torch.no_grad():
        tp = mdl(torch.from_numpy(eta), torch.from_numpy(pt), torch.from_numpy(img))
    tpar, tprob = [t.numpy() for t in tp]
    d1, d2 = np.abs(tpar - par).max(), np.abs(tprob - prob).max()
    print("max |torch - onnx|: parameters %.2e, probability %.2e" % (d1, d2))
    if max(d1, d2) > 1e-4:
        raise SystemExit("FAIL: ONNX and PyTorch differ")
print("PASS")
