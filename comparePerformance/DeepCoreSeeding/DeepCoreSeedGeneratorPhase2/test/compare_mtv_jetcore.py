#!/usr/bin/env python3
"""
Compare MultiTrackValidator results of two (or more) step3/step4 DQM files,
e.g. baseline jet-core (pixel pairs) vs DeepCore Phase-2.

Efficiency and fake rate are recomputed from the numerator/denominator
histograms (num_assoc(simToReco)_X / num_simul_X, 1 - num_assoc(recoToSim)_X /
num_reco_X), so it works on harvested (DQM_V0001_*.root) files.

Usage:
  python3 compare_mtv_jetcore.py \
      --files DQM_V0001_R000000001__base.root DQM_V0001_R000000001__deepcore.root \
      --names "pixel pairs" DeepCore --folder JetCore \
      --labels generalTracks jetCoreRegionalStepTracks cutsRecoTracksJetCoreRegionalStepByOriginalAlgo \
      --out mtv_jetcore.pdf

--folder: JetCore  (trackValidatorJetCore, chi2 associator, from customiseJetCoreValidation)
          Track    (standard trackValidatorTrackingOnly, hit associator)
"""
import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VARS = [("drj", r"$\Delta R$(track, jet)", True),
        ("pT", r"$p_T$ [GeV]", True),
        ("eta", r"$\eta$", False),
        ("hit", "number of hits", False)]


def find_dirs(f, folder):
    base = None
    for k in f.keys(recursive=True, cycle=False):
        if k.endswith("Tracking/Run summary/" + folder) or k.endswith("Tracking/Run summary/" + folder + "/"):
            base = k
            break
    if base is None:
        raise RuntimeError("no Tracking/Run summary/%s in %s" % (folder, f.file_path))
    return base, [k.split("/")[-1] for k in f[base].keys(recursive=False, cycle=False)]


def mtv_dir(label):
    """MultiTrackValidator folder prefix: it removes the first 'Tracks' from the label
    (generalTracks -> general, jetCoreRegionalStepTracks -> jetCoreRegionalStep,
     cutsRecoTracksX -> cutsRecoX)."""
    i = label.find("Tracks")
    return label if i < 0 else label[:i] + label[i + 6:]


def get(f, path):
    try:
        h = f[path]
    except Exception:
        return None
    v, e = h.to_numpy()
    return v.astype(float), e


def ratio(num, den, fake=False):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
        err = np.where(den > 0, np.sqrt(np.clip(r * (1 - r), 0, None) / den), np.nan)
    return (1 - r, err) if fake else (r, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--folder", default="JetCore")
    ap.add_argument("--labels", nargs="+",
                    default=["generalTracks", "jetCoreRegionalStepTracks",
                             "cutsRecoTracksJetCoreRegionalStepByOriginalAlgo"])
    ap.add_argument("--out", default="mtv_jetcore.pdf")
    a = ap.parse_args()
    names = a.names or [f.split("/")[-1] for f in a.files]
    files = [uproot.open(f) for f in a.files]

    rows = []
    with PdfPages(a.out) as pdf:
        for lab in a.labels:
            for var, xt, logx in VARS:
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
                drawn = False
                for f, nm in zip(files, names):
                    base, dirs = find_dirs(f, a.folder)
                    d = [x for x in dirs if x.startswith(mtv_dir(lab) + "_") or x.startswith(lab + "_")]
                    if not d:
                        continue
                    p = base + "/" + d[0] + "/"
                    ns = get(f, p + "num_assoc(simToReco)_" + var)
                    ds = get(f, p + "num_simul_" + var)
                    nr = get(f, p + "num_assoc(recoToSim)_" + var)
                    dr = get(f, p + "num_reco_" + var)
                    if ns is None or ds is None:
                        continue
                    edges = ns[1]
                    x = 0.5 * (edges[1:] + edges[:-1])
                    eff, ee = ratio(ns[0], ds[0])
                    ax[0].errorbar(x, eff, yerr=ee, fmt="o-", ms=3, label=nm)
                    if nr is not None and dr is not None:
                        fk, fe = ratio(nr[0], dr[0], fake=True)
                        ax[1].errorbar(x, fk, yerr=fe, fmt="o-", ms=3, label=nm)
                    drawn = True
                    if var == "drj":
                        S, Ns = ds[0].sum(), ns[0].sum()
                        R, Nr = (dr[0].sum(), nr[0].sum()) if dr is not None else (0, 0)
                        rows.append((lab, nm, Ns / S if S else np.nan, 1 - Nr / R if R else np.nan, int(S), int(R)))
                if not drawn:
                    plt.close(fig)
                    continue
                for i, yt in enumerate(["efficiency", "fake rate"]):
                    ax[i].set_xlabel(xt)
                    ax[i].set_ylabel(yt)
                    ax[i].set_ylim(0, 1.05)
                    ax[i].grid(alpha=.3)
                    if logx:
                        ax[i].set_xscale("log")
                    ax[i].legend()
                fig.suptitle("%s  (%s)" % (lab, a.folder))
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    if not rows:
        print("Nothing matched. Folders in %s:" % a.files[0])
        try:
            base, dirs = find_dirs(files[0], a.folder)
            for x in sorted(dirs):
                print("   ", x)
        except Exception as e:
            print("   ", e)
        print("Pass labels with --labels (track labels like generalTracks, or the folder prefix shown above).")
    print("%-50s %-14s %8s %8s %9s %9s" % ("label", "config", "eff", "fake", "N_sim", "N_reco"))
    for r in rows:
        print("%-50s %-14s %8.4f %8.4f %9d %9d" % r)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
