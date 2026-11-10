#!/usr/bin/env python3
"""
Compare DeepCore checkpoints on the SAME validation events.

Inputs, either (or mixed):
  --weights ep086=DeepCore_ckpt_ep086.pt ep088=DeepCore_ckpt_ep088.pt ...   (runs the network)
  --pred    ep086=pred_ep086.npz ...                                         (DeepCore_testing_pytorch.py output,
                                                                              made with the same --input/--maxEvents)
Example:
  python compare_checkpoints.py --input "/eos/.../validation/*.root" --maxEvents 3000 \
      --weights ep086=DeepCore_ckpt_ep086.pt ep088=DeepCore_ckpt_ep088.pt ep095=DeepCore_ckpt_ep095.pt

For every checkpoint it computes, with the definitions of deepcore_common.eff_fake (chi2 <= 4 and 8):
  * efficiency, fake rate and duplicates vs the probability threshold (an efficiency-vs-fake curve)
  * the validation losses (par loss, ROI and ROIsoft prob losses) on these events
  * residual mean / width of (dx, dy, deta, dphi, pt) for matched crossings
and the event-bootstrap uncertainty of the DIFFERENCE between checkpoints (paired: same events), which is what
decides whether one epoch is really better than another.
Outputs: compare_checkpoints.csv, compare_checkpoints.pdf, and a summary on stdout.
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepcore_common as dc

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="validation ntuple(s), glob allowed (NOT used in training)")
p.add_argument("--weights", nargs="*", default=[], help="label=checkpoint.pt")
p.add_argument("--pred", nargs="*", default=[], help="label=prediction.npz from DeepCore_testing_pytorch.py")
p.add_argument("--maxEvents", type=int, default=2000)
p.add_argument("--batchSize", type=int, default=32)
p.add_argument("--layNum", type=int, default=4)
p.add_argument("--probThr", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
p.add_argument("--refThr", type=float, default=0.85, help="threshold used for the summary / bootstrap (CMSSW probThr)")
p.add_argument("--nBoot", type=int, default=200)
p.add_argument("--normFrom", default="", help="checkpoint whose track-parameter normalization is used for the par loss "
                                          "of --pred files (default: the first --weights checkpoint)")
p.add_argument("--cpu", action="store_true")
p.add_argument("--out", default="compare_checkpoints")
args = p.parse_args()

CHI2 = (4, 8)
NEAR = 2
LAY_DIST = 3
thrs = sorted(set(args.probThr) | {args.refThr})

# ------------------------------------------------------------------------------------------------
# data (read once, shared by all checkpoints)
# ------------------------------------------------------------------------------------------------
src = dc.tree_sources(args.input)
if not src:
    sys.exit("no {}/{} in {}".format(dc.INPUT_MODULE, dc.INPUT_TREE, args.input))
data = dc.load_arrays(src, args.layNum, with_target=True, max_events=args.maxEvents)
N = len(data["jet_pt"])
print("events:", N, " window:", data["cluster_measured"].shape[1:])


# ------------------------------------------------------------------------------------------------
# per-event counts (same logic as dc.eff_fake, kept per event so they can be bootstrapped)
# ------------------------------------------------------------------------------------------------
def event_counts(par_pred, prob_pred, tpar, tprob):
    """returns dict[(thr, c)] -> int array [n_events, 4] = (n_true, found, n_pred, n_good)"""
    d = par_pred[..., :4] - tpar[..., :4]
    scale = np.arctan(2 / float(LAY_DIST * 3))
    chi2 = d[..., 0] ** 2 + d[..., 1] ** 2 + (d[..., 2] ** 2 + d[..., 3] ** 2) / scale
    flag = tpar[..., -1] > 0
    is_t = tprob == 1
    n_true = is_t.sum(axis=(1, 2, 3))
    X, Y = is_t.shape[1], is_t.shape[2]
    out = {}
    for thr in thrs:
        is_p = prob_pred > thr
        n_pred = is_p.sum(axis=(1, 2, 3))
        for c in CHI2:
            good = is_p & flag & (chi2 <= c)
            g = good.any(axis=-1)
            gp = np.pad(g, ((0, 0), (NEAR, NEAR), (NEAR, NEAR)))
            dil = np.zeros_like(g)
            for dx in range(-NEAR, NEAR + 1):
                for dy in range(-NEAR, NEAR + 1):
                    dil |= gp[:, NEAR + dx:NEAR + dx + X, NEAR + dy:NEAR + dy + Y]
            found = (is_t & dil[..., None]).sum(axis=(1, 2, 3))
            out[(thr, c)] = np.stack([n_true, found, n_pred, good.sum(axis=(1, 2, 3))], 1)
    return out


def metrics(cnt):
    n_true, found, n_pred, n_good = cnt.sum(0)
    return dict(eff=found / max(n_true, 1), fake=(n_pred - n_good) / n_pred if n_pred else 1.,
                dup=n_good / max(found, 1), n_pred_per_event=n_pred / len(cnt))


def loss_sums(par, prob, tpar, tprob, norm):
    """numerators/denominators of the training losses (prob = sigmoid output)"""
    mean, sd = np.asarray(norm["mean"]), np.asarray(norm["sd"])
    wei = tpar[..., -1:]
    pp, tt = (par[..., :-1] - mean) / sd, (tpar[..., :-1] - mean) / sd
    lp = (np.clip(pp - tt, -5, 5) ** 2 * wei).sum()
    p_ = np.clip(prob[..., None], 1e-7, 1 - 1e-7)
    t_ = tprob[..., None]
    bce = -(10. * t_ * np.log(p_) + (1 - t_) * np.log(1 - p_))      # pos_weight = 10
    w = tpar[..., -1:]
    return np.array([lp, wei.sum() * 5, (bce * w).sum(), w.sum(), (bce * (w + 0.01)).sum(), (w + 0.01).sum()])


# ------------------------------------------------------------------------------------------------
# run every model / prediction file
# ------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
models = [tuple(s.split("=", 1)) if "=" in s else (os.path.basename(s), s) for s in args.weights]
preds = [tuple(s.split("=", 1)) if "=" in s else (os.path.basename(s), s) for s in args.pred]
if not models and not preds:
    sys.exit("give --weights and/or --pred")

norm_pred = None
for f in ([args.normFrom] if args.normFrom else []) + [m[1] for m in models]:
    norm_pred = torch.load(f, map_location="cpu", weights_only=False).get("norm")
    if norm_pred:
        break
if preds and norm_pred is None:
    print("WARNING: no checkpoint given for the normalization (--normFrom): the par loss of the prediction files "
          "uses the Phase-1 constants and is not comparable to the training plots")
    norm_pred = dict(dc.PHASE1_NORM)

results = {}
for label, path in models + preds:
    is_model = path.endswith(".pt")
    norm = norm_pred
    counts = {k: [] for k in [(t, c) for t in thrs for c in CHI2]}
    lsum = np.zeros(6)
    resid = []
    if is_model:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        norm = ckpt.get("norm", norm)
        model = dc.DeepCoreNet(lay_num=ckpt.get("config", {}).get("layNum", args.layNum)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    else:
        z = np.load(path)
        vpar, vprob = z["validation_par"], z["validation_prob"]
        if vprob.ndim == 5:
            vprob = vprob[..., 0]
        if len(vpar) < N:
            sys.exit("%s has %d events, the input has %d: use the same --input/--maxEvents" % (path, len(vpar), N))
    for i in range(0, N, args.batchSize):
        sl = slice(i, i + args.batchSize)
        tpar, tprob = data["trackPar"][sl], data["trackProb"][sl]
        if is_model:
            with torch.no_grad():
                clus = torch.from_numpy(data["cluster_measured"][sl]).permute(0, 3, 1, 2).to(device)
                par, logit = model(clus, torch.from_numpy(data["jet_eta"][sl]).to(device),
                                   torch.from_numpy(data["jet_pt"][sl]).to(device))
                par = par.cpu().numpy()
                prob = torch.sigmoid(logit[..., 0]).cpu().numpy()
        else:
            par, prob = vpar[sl], vprob[sl]
        for k, v in event_counts(par, prob, tpar, tprob).items():
            counts[k].append(v)
        lsum += loss_sums(par, prob, tpar, tprob, norm)
        sel = (prob > args.refThr) & (tprob == 1)
        resid.append(par[sel][:, :5] - tpar[sel][:, :5])
    counts = {k: np.concatenate(v) for k, v in counts.items()}
    resid = np.concatenate(resid)
    results[label] = dict(counts=counts, loss=dict(par=lsum[0] / lsum[1], prob_ROI=lsum[2] / lsum[3],
                                                    prob_ROIsoft=lsum[4] / lsum[5]),
                          resid_mean=resid.mean(0) if len(resid) else np.full(5, np.nan),
                          resid_sd=resid.std(0) if len(resid) else np.full(5, np.nan),
                          resid_iqr=(np.subtract(*np.percentile(resid, [75, 25], axis=0)) if len(resid)
                                     else np.full(5, np.nan)),
                          n_resid=len(resid), source="checkpoint" if is_model else "prediction file")
    m = metrics(counts[(args.refThr, 4)])
    print("%-10s eff(chi2<=4)=%.4f fake=%.4f  | loss par %.4f ROI %.4f ROIsoft %.5f  [%s]"
          % (label, m["eff"], m["fake"], results[label]["loss"]["par"], results[label]["loss"]["prob_ROI"],
             results[label]["loss"]["prob_ROIsoft"], results[label]["source"]), flush=True)

labels = list(results)

# ------------------------------------------------------------------------------------------------
# paired bootstrap of the differences at the reference threshold
# ------------------------------------------------------------------------------------------------
rng = np.random.default_rng(12345)
boot_idx = rng.integers(0, N, size=(args.nBoot, N))


def boot(lbl, key, what):
    c = results[lbl]["counts"][key]
    return np.array([metrics(c[ix])[what] for ix in boot_idx])


print("\n=== at probThr = %.2f (bootstrap over %d events, %d replicas) ===" % (args.refThr, N, args.nBoot))
summary = {}
for c in CHI2:
    key = (args.refThr, c)
    for what in ("eff", "fake"):
        b = {l: boot(l, key, what) for l in labels}
        for l in labels:
            summary[(l, what, c)] = (metrics(results[l]["counts"][key])[what], b[l].std())
            print("  %-10s %-4s chi2<=%d : %.4f +- %.4f" % (l, what, c, *summary[(l, what, c)]))
        ref = labels[0]
        for l in labels[1:]:
            d = b[l] - b[ref]
            dv = metrics(results[l]["counts"][key])[what] - metrics(results[ref]["counts"][key])[what]
            print("     %s - %s : %+.4f +- %.4f  (%.1f sigma)" % (l, ref, dv, d.std(), dv / d.std() if d.std() else 0))

# ------------------------------------------------------------------------------------------------
# outputs
# ------------------------------------------------------------------------------------------------
import csv
with open(args.out + ".csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["label", "probThr", "chi2", "eff", "fake", "dup", "n_pred_per_event"])
    for l in labels:
        for t in thrs:
            for c in CHI2:
                m = metrics(results[l]["counts"][(t, c)])
                w.writerow([l, t, c, m["eff"], m["fake"], m["dup"], m["n_pred_per_event"]])
    w.writerow([])
    w.writerow(["label", "loss_par", "loss_prob_ROI", "loss_prob_ROIsoft", "n_matched"]
               + ["resid_mean_%s" % n for n in ("dx", "dy", "deta", "dphi", "pt")]
               + ["resid_sd_%s" % n for n in ("dx", "dy", "deta", "dphi", "pt")]
               + ["resid_iqr_%s" % n for n in ("dx", "dy", "deta", "dphi", "pt")])
    for l in labels:
        r = results[l]
        w.writerow([l, r["loss"]["par"], r["loss"]["prob_ROI"], r["loss"]["prob_ROIsoft"], r["n_resid"]]
                   + list(r["resid_mean"]) + list(r["resid_sd"]) + list(r["resid_iqr"]))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(args.out + ".pdf") as pdf:
    for c in CHI2:
        fig, ax = plt.subplots()
        for l in labels:
            ms = [metrics(results[l]["counts"][(t, c)]) for t in thrs]
            ax.plot([m["fake"] for m in ms], [m["eff"] for m in ms], marker="o", label=l)
            mr = metrics(results[l]["counts"][(args.refThr, c)])
            ax.errorbar(mr["fake"], mr["eff"], xerr=summary[(l, "fake", c)][1], yerr=summary[(l, "eff", c)][1],
                        color=ax.lines[-1].get_color(), capsize=3)
        ax.set_xlabel("fake rate")
        ax.set_ylabel("efficiency")
        ax.set_title(r"efficiency vs fake rate, $\chi^2 \leq %d$ (points: probThr %s; bars at %.2f)"
                     % (c, ", ".join("%g" % t for t in thrs), args.refThr), fontsize=8)
        ax.grid(True)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for what in ("eff", "fake"):
            fig, ax = plt.subplots()
            for l in labels:
                ax.plot(thrs, [metrics(results[l]["counts"][(t, c)])[what] for t in thrs], marker="o", label=l)
            ax.set_xlabel("probability threshold")
            ax.set_ylabel({"eff": "efficiency", "fake": "fake rate"}[what] + r" ($\chi^2\leq%d$)" % c)
            ax.grid(True)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.2))
    names = [r"$\Delta x$", r"$\Delta y$", r"$\Delta\eta$", r"$\Delta\phi$", r"$p_T$"]
    for i, ax in enumerate(axs):
        ax.bar(range(len(labels)), [results[l]["resid_iqr"][i] for l in labels])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_title("IQR of residual " + names[i], fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
print("\nwrote %s.csv and %s.pdf" % (args.out, args.out))
