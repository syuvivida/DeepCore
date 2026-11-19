#!/usr/bin/env python3
"""
Check the CMSSW DeepCoreSeedGeneratorPhase2 against the training chain, using the windows it dumps
(run_DeepCoreSeedGeneratorPhase2_cfg.py dumpDir=dump).

  1) image check: the generator image must equal cluster_measured of DeepCoreNtuplizerPhase2 for the same event,
     jet and direction (run the ntuplizer on the same input file):
        python3 compare_images.py --dump dump --ntuple DeepCoreTrainingSample.root
  2) network check: the CMSSW (ONNX Runtime) outputs must equal the PyTorch checkpoint on the same image:
        python3 compare_images.py --dump dump --weights DeepCore_ckpt_ep088.pt --pytorchDir /path/to/training/pytorch

Both checks can be run together. Needs numpy (+ uproot for 1, torch for 2).
"""
import argparse
import glob
import os
import sys

import numpy as np

NOVER, NPAR1, NLAY = 3, 6, 4

p = argparse.ArgumentParser()
p.add_argument("--dump", required=True, help="directory with the .bin files")
p.add_argument("--ntuple", default="", help="DeepCoreNtuplizerPhase2 output (TFileService file)")
p.add_argument("--tree", default="DeepCoreNtuplizerTest/DeepCoreNtuplizerPhase2Tree")
p.add_argument("--weights", default="", help="PyTorch checkpoint (.pt) for the network check")
p.add_argument("--pytorchDir", default="", help="directory with deepcore_common.py")
p.add_argument("--tol", type=float, default=1e-4)
p.add_argument("--probThr", type=float, default=0.85, help="threshold for the impact check")
args = p.parse_args()


def read(fn):
    with open(fn, "rb") as f:
        head = np.frombuffer(f.read(64), np.float64)
        X, Y = int(head[6]), int(head[7])
        img = np.frombuffer(f.read(4 * X * Y * NLAY), np.float32).reshape(X, Y, NLAY)
        prob = np.frombuffer(f.read(4 * X * Y * NOVER), np.float32).reshape(X, Y, NOVER)
        par = np.frombuffer(f.read(4 * X * Y * NOVER * NPAR1), np.float32).reshape(X, Y, NOVER, NPAR1)
    return dict(run=int(head[0]), event=int(head[1]), jet=int(head[2]), dir=int(head[3]), eta=head[4], pt=head[5],
                img=img, prob=prob, par=par)


files = sorted(glob.glob(os.path.join(args.dump, "*.bin")))
if not files:
    sys.exit("no .bin files in " + args.dump)
W = [read(f) for f in files]
print("%d windows from %d events" % (len(W), len({(w['run'], w['event']) for w in W})))

# ------------------------------------------------------------------------------------------------
if args.ntuple:
    import uproot
    t = uproot.open(args.ntuple)[args.tree]
    branches = ["event_ID", "jet_eta", "jet_pt", "cluster_measured"]
    has_truth = all(b in t.keys() for b in ("trackPar", "trackProb"))
    nt = t.arrays(branches + (["trackPar", "trackProb"] if has_truth else []), library="np")
    # the ntuplizer writes the directions of one jet consecutively, in the same order as the generator
    groups = {}
    for i in range(len(nt["event_ID"])):
        key = (int(nt["event_ID"][i]), round(float(nt["jet_eta"][i]), 4), round(float(nt["jet_pt"][i]), 2))
        groups.setdefault(key, []).append(i)
    n_ok = n_bad = n_missing = 0
    worst = []
    pairs = []      # (window, ntuple image) for the windows that differ
    for iw, w in enumerate(W):
        key = (w["event"] & 0x7fffffff, round(w["eta"], 4), round(w["pt"], 2))
        idx = groups.get(key)
        if idx is None or w["dir"] >= len(idx):
            n_missing += 1
            continue
        ref = np.asarray(nt["cluster_measured"][idx[w["dir"]]], np.float32).reshape(w["img"].shape)
        d = np.abs(ref - w["img"]).max()
        if d < args.tol:
            n_ok += 1
        else:
            n_bad += 1
            pairs.append((w, ref, idx[w["dir"]]))
            worst.append((d, os.path.basename(files[iw]), ref.sum(axis=(0, 1)), w["img"].sum(axis=(0, 1))))
    print("\nIMAGE CHECK: %d identical, %d different, %d without a ntuple entry" % (n_ok, n_bad, n_missing))
    for d, fn, sref, sgen in sorted(worst, reverse=True)[:10]:
        print("  %-35s max|diff| %.4g   charge per layer ntuple %s  generator %s"
              % (fn, d, np.round(sref, 3), np.round(sgen, 3)))
    if n_bad:
        nt_empty = sum(1 for w, ref, _ in pairs for l in range(NLAY) if ref[..., l].sum() == 0 and w["img"][..., l].sum() > 0)
        gen_empty = sum(1 for w, ref, _ in pairs for l in range(NLAY) if w["img"][..., l].sum() == 0 and ref[..., l].sum() > 0)
        same = sum(1 for w, ref, _ in pairs for l in range(NLAY)
                   if ref[..., l].sum() > 0 and w["img"][..., l].sum() > 0 and np.abs(ref[..., l] - w["img"][..., l]).max() >= args.tol)
        print("  differing layers: %d empty in the ntuple only, %d empty in the generator only, %d filled in both but different"
              % (nt_empty, gen_empty, same))
        print("  (empty in the ntuple only = the ntuplizer chose a module with SimHits but no clusters in the window, while the"
              " generator chose the module with the clusters)")
    if n_missing:
        print("  (windows without an entry: different jets/directions, e.g. the ntuplizer picks modules with SimHits,"
              " the generator modules with clusters, or a direction had no module on layer 2)")

# ------------------------------------------------------------------------------------------------
if args.weights:
    import torch
    sys.path.insert(0, args.pytorchDir or os.getcwd())
    import deepcore_common as dc
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    net = dc.DeepCoreNet(lay_num=ck.get("config", {}).get("layNum", NLAY))
    net.load_state_dict(ck["model"])
    exp = dc.DeepCoreExport(net).eval()
    dpar = dprob = 0.
    with torch.no_grad():
        for w in W:
            par, prob = exp(torch.tensor([[w["eta"]]], dtype=torch.float32), torch.tensor([[w["pt"]]], dtype=torch.float32),
                            torch.from_numpy(w["img"].copy())[None])
            dpar = max(dpar, float(np.abs(par[0].numpy() - w["par"]).max()))
            dprob = max(dprob, float(np.abs(prob[0, ..., 0].numpy() - w["prob"]).max()))
    print("\nNETWORK CHECK (CMSSW ONNX vs PyTorch on the same image): max|d par| = %.2e, max|d prob| = %.2e  -> %s"
          % (dpar, dprob, "OK" if dpar < 1e-3 and dprob < 1e-4 else "MISMATCH"))

    # impact of the image differences on the predictions (needs --ntuple as well)
    if args.ntuple and pairs:
        thr = args.probThr
        n_nt = n_gen = n_common = 0
        P = {"ntuple": [], "generator": []}
        T = {"par": [], "prob": []}
        n_same2 = 0
        with torch.no_grad():
            for w, ref, ient in pairs:
                eta = torch.tensor([[w["eta"]]], dtype=torch.float32); pt = torch.tensor([[w["pt"]]], dtype=torch.float32)
                o_nt = exp(eta, pt, torch.from_numpy(ref.astype(np.float32))[None])
                o_gen = exp(eta, pt, torch.from_numpy(w["img"].copy())[None])
                p_nt, p_gen = o_nt[1][0, ..., 0].numpy(), o_gen[1][0, ..., 0].numpy()
                n_nt += int((p_nt > thr).sum()); n_gen += int((p_gen > thr).sum()); n_common += int(((p_nt > thr) & (p_gen > thr)).sum())
                # truth comparison only where both chose the same layer-2 module (same seed surface, same targets)
                if has_truth and np.abs(ref[..., 1] - w["img"][..., 1]).max() < args.tol and ref[..., 1].sum() > 0:
                    n_same2 += 1
                    P["ntuple"].append((o_nt[0][0].numpy(), p_nt)); P["generator"].append((o_gen[0][0].numpy(), p_gen))
                    T["par"].append(np.asarray(nt["trackPar"][ient], np.float32).reshape(o_nt[0][0].shape))
                    T["prob"].append(np.asarray(nt["trackProb"][ient], np.float32).reshape(p_nt.shape))
        print("\nIMPACT on the %d differing windows (prob > %.2f): %d crossings predicted from the ntuple image, %d from the"
              " generator image, %d in common" % (len(pairs), thr, n_nt, n_gen, n_common))
        if has_truth and n_same2:
            tpar, tprob = np.stack(T["par"]), np.stack(T["prob"])
            print("TRUTH CHECK on the %d differing windows with the same layer-2 module (true crossings: %d):"
                  % (n_same2, int((tprob == 1).sum())))
            for name in ("ntuple", "generator"):
                par = np.stack([a for a, _ in P[name]]); prob = np.stack([b for _, b in P[name]])
                r = dc.eff_fake(par, prob, tpar, tprob, thr)
                print("   %-9s image: efficiency %.3f / %.3f, fake rate %.3f / %.3f  (chi2 <= 4 / 8), %d predictions"
                      % (name, r["eff_4"], r["eff_8"], r["fake_4"], r["fake_8"], int((prob > thr).sum())))
        elif not has_truth:
            print("(no trackPar/trackProb in the ntuple: no truth check)")
