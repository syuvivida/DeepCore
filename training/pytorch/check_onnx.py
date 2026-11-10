#!/usr/bin/env python3
"""Check that the exported ONNX model gives the same outputs as the PyTorch checkpoint.

  python check_onnx.py --weights DeepCore_ckpt_ep088.pt --onnx DeepCore_phase2.onnx \
                       --input NTUPLE.root --maxEvents 20

Needs: torch, uproot, onnxruntime (pip install onnxruntime).
"""
import os
import sys
import argparse

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepcore_common as dc

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--onnx", required=True)
p.add_argument("--input", required=True, help="ntuple (ROOT, glob allowed)")
p.add_argument("--maxEvents", type=int, default=20)
p.add_argument("--atol", type=float, default=1e-4)
args = p.parse_args()

ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
net = dc.DeepCoreNet(lay_num=cfg["layNum"])
net.load_state_dict(ckpt["model"])
ref = dc.DeepCoreExport(net).eval()

src = dc.tree_sources(args.input)
data = dc.load_arrays(src, cfg["layNum"], with_target=False, max_events=args.maxEvents)
eta = data["jet_eta"][:, None].astype(np.float32)
pt = data["jet_pt"][:, None].astype(np.float32)
img = data["cluster_measured"].astype(np.float32)          # [N, X, Y, L], as CMSSW fills it
print("events:", len(eta), " window:", img.shape[1:])

with torch.no_grad():
    t_par, t_prob = (o.numpy() for o in ref(torch.from_numpy(eta), torch.from_numpy(pt), torch.from_numpy(img)))

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
print("ONNX inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
print("ONNX outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

# run one jet at a time, like DeepCoreSeedGenerator does
o_par, o_prob = [], []
for i in range(len(eta)):
    a, b = sess.run(["output_node0", "output_node1"],
                    {"input_1": eta[i:i + 1], "input_2": pt[i:i + 1], "input_3": img[i:i + 1]})
    o_par.append(a)
    o_prob.append(b)
o_par, o_prob = np.concatenate(o_par), np.concatenate(o_prob)

ok = True
for name, t, o in (("par", t_par, o_par), ("prob", t_prob, o_prob)):
    d = np.abs(t - o)
    print("{:5s} shape {}  max|diff| {:.2e}  mean|diff| {:.2e}".format(name, o.shape, d.max(), d.mean()))
    ok &= bool(d.max() < args.atol)

thr = 0.85
n_t, n_o = int((t_prob[..., 0] > thr).sum()), int((o_prob[..., 0] > thr).sum())
print("pixels above prob {}: torch {}  onnx {}".format(thr, n_t, n_o))
print("PASS" if ok and n_t == n_o else "FAIL")
sys.exit(0 if ok and n_t == n_o else 1)
