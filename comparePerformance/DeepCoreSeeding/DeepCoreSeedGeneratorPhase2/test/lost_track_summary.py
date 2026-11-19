#!/usr/bin/env python3
"""Where do the lost high-pT tracks go?  Reads the LostTrackAnalyzer tree (customiseLostTracks)
and sorts every simulated track that has no generalTracks match into the first stage that failed:

  merge   found in earlyGeneralTracks / preDuplicateMergingGeneralTracks, lost afterwards
  select  built by an iteration (raw track collection), dropped by its classifier / the merger
            (split into the jet-core step and the other iterations)
  fit     jet-core track candidate, but no fitted jet-core track
  build   jet-core seed (pixel pair hit-matched, or DeepCore seed within --dc-dr), no candidate
  seed    no jet-core seed at all

Usage:
  python3 lost_track_summary.py lostTracks.root [--name deepcore] [--pt-min 100] [--dc-dr 0.02] [--out lost.pdf]
"""
import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CATS = ["found", "merge", "select (jet-core)", "select (other iter.)", "fit", "build (pixel-pair seed)",
        "build (DeepCore seed only)", "seed"]
COLORS = ["#4daf4a", "#984ea3", "#ff7f00", "#ffd92f", "#a65628", "#377eb8", "#80b1d3", "#e41a1c"]


def is_deepcore_seed(name):
    return name.startswith("seedTracks") and (name.endswith("SeedsBarrel") or name.endswith("SeedsDeepCore"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--tree", default="lostTrackAnalyzer/tp")
    ap.add_argument("--name", default="")
    ap.add_argument("--pt-min", type=float, default=100.)
    ap.add_argument("--dc-dr", type=float, default=0.02, help="DeepCore seed match: Delta R(seed, TP) below this")
    ap.add_argument("--dc-dpt", type=float, default=1e9, help="DeepCore seed match: |dpT/pT| below this")
    ap.add_argument("--out", default="lost_tracks.pdf")
    a = ap.parse_args()

    t = uproot.open(a.file)[a.tree]
    keys = set(t.keys())
    d = t.arrays(library="np")
    sel = d["pt"] > a.pt_min
    d = {k: v[sel] for k, v in d.items()}
    n = int(sel.sum())
    if n == 0:
        raise SystemExit("no entries above pt-min")

    colls = sorted(k[:-len("_nMatch")] for k in keys if k.endswith("_nMatch"))
    cands = [c for c in colls if c + "_pt" not in keys]
    trks = [c for c in colls if c not in cands]
    seeds = [c for c in trks if c.startswith("seedTracks")]
    dcSeeds = [c for c in seeds if is_deepcore_seed(c)]
    hitSeeds = [c for c in seeds if c not in dcSeeds]
    post = [c for c in ("earlyGeneralTracks", "preDuplicateMergingGeneralTracks") if c in trks]
    jcTrk = [c for c in trks if c.startswith("jetCoreRegionalStep") and c not in seeds]
    iterTrk = [c for c in trks if c not in seeds + post + jcTrk + ["generalTracks"]]
    if "generalTracks" not in trks:
        raise SystemExit("generalTracks not in the tree")

    m = lambda c: d[c + "_nMatch"] > 0
    anyOf = lambda cs: np.any([m(c) for c in cs], axis=0) if cs else np.zeros(n, bool)
    dcMatch = {c: (d[c + "_dR"] < a.dc_dr) & (np.abs(d[c + "_dPtRel"]) < a.dc_dpt) for c in dcSeeds}
    anyDC = np.any(list(dcMatch.values()), axis=0) if dcSeeds else np.zeros(n, bool)

    found = m("generalTracks")
    lost = ~found
    hasPost, hasJC, hasIter = anyOf(post), anyOf(jcTrk), anyOf(iterTrk)
    hasCand, hasHitSeed = anyOf(cands), anyOf(hitSeeds)

    cat = np.full(n, len(CATS) - 1)
    rules = [(found, 0), (lost & hasPost, 1), (lost & ~hasPost & hasJC, 2), (lost & ~hasPost & ~hasJC & hasIter, 3),
             (lost & ~hasPost & ~hasJC & ~hasIter & hasCand, 4),
             (lost & ~hasPost & ~hasJC & ~hasIter & ~hasCand & hasHitSeed, 5),
             (lost & ~hasPost & ~hasJC & ~hasIter & ~hasCand & ~hasHitSeed & anyDC, 6)]
    for mask, k in reversed(rules):
        cat[mask] = k

    tag = (" [%s]" % a.name) if a.name else ""
    print("file %s%s:  %d simulated tracks with pT > %g GeV in %d events" %
          (a.file, tag, n, a.pt_min, len(set(zip(d["run"], d["lumi"], d["event"])))))
    print("\nfraction of simulated tracks matched (hit associator; DeepCore seeds: Delta R < %g)" % a.dc_dr)
    for c in seeds + cands + jcTrk + iterTrk + post + ["generalTracks"]:
        f = dcMatch[c] if c in dcMatch else m(c)
        print("   %-48s %7d  %.3f%s" % (c, f.sum(), f.mean(), "   (geometric)" if c in dcMatch else ""))

    print("\nfirst stage that failed, simulated tracks without a generalTracks match (%d = %.1f %% of all)" %
          (lost.sum(), 100 * lost.mean()))
    for k in range(1, len(CATS)):
        s = (cat == k).sum()
        print("   %-30s %6d   %5.1f %% of lost   %5.1f %% of all" % (CATS[k], s, 100 * s / max(lost.sum(), 1), 100 * s / n))

    # what distinguishes lost from found
    def row(label, x):
        print("   %-40s %8.3f %8.3f" % (label, np.mean(x[found]) if found.any() else np.nan,
                                         np.mean(x[lost]) if lost.any() else np.nan))
    print("\nproperties                                  found     lost")
    if (d["nPixClus"] >= 0).all():
        row("<# pixel clusters>", d["nPixClus"])
        row("<# pixel clusters shared with other TP>", d["nPixClusShared"])
        row("frac. with >=1 shared pixel cluster", d["nPixClusShared"] >= 1)
        row("frac. with >=2 shared pixel clusters", d["nPixClusShared"] >= 2)
        row("frac. >=1 shared with TP pT>1 GeV", d["nPixClusSharedHighPt"] >= 1)
    if (d["nPixLay"] >= 0).all():
        row("<# pixel layers with sim hits>", d["nPixLay"])
    row("frac. with other TP within dR < 0.002", d["tpDR"] < 0.002)
    row("frac. with other TP within dR < 0.005", d["tpDR"] < 0.005)
    row("frac. within dR < 0.1 of a jet (pT>100)", d["jetDR"] < 0.1)
    row("frac. |eta| < 1.4", np.abs(d["eta"]) < 1.4)
    close = d["tpDR"] < 0.001
    print("\nefficiency for tracks with another simulated track within dR < 0.001: %d / %d = %.3f" %
          (found[close].sum(), close.sum(), found[close].mean() if close.any() else np.nan))
    for c in dcSeeds:
        cand = [x for x in cands if "Barrel" in x or "DeepCore" in x]
        s = dcMatch[c]
        print("\n   %s: <# seeds within dR < nearDR of a track> %.1f (lost %.1f, found %.1f)" %
              (c, d[c + "_nNear"].mean(), d[c + "_nNear"][lost].mean() if lost.any() else np.nan,
               d[c + "_nNear"][found].mean()))
        if cand and s.any():
            cm = anyOf(cand)
            print("      P(%s | DeepCore seed) = %.3f;  for tracks with dR(other TP) < 0.001: %.3f" %
                  ("+".join(cand), cm[s].mean(), cm[s & close].mean() if (s & close).any() else np.nan))
    for c in dcSeeds:
        q = d[c + "_dR"][hasJC & (d[c + "_dR"] < 900)]
        if q.size:
            print("\n   %s: Delta R to nearest seed for tracks that the jet-core step did build:" % c)
            print("      quantiles 50/68/90/95 %%: %s" % "  ".join("%.4f" % x for x in np.quantile(q, [.5, .68, .9, .95])))

    # plots
    with PdfPages(a.out) as pdf:
        for var, edges, xl, logx in [("pt", np.geomspace(a.pt_min, 2000, 13), r"simulated track $p_T$ [GeV]", True),
                                     ("tpDR", np.geomspace(2e-4, 0.5, 13), r"min $\Delta R$ to other TP", True),
                                     ("eta", np.linspace(-2.5, 2.5, 11), r"$\eta$", False)]:
            x = d[var]
            idx = np.digitize(x, edges) - 1
            ok = (idx >= 0) & (idx < len(edges) - 1)
            tot = np.bincount(idx[ok], minlength=len(edges) - 1).astype(float)
            fig, ax = plt.subplots(figsize=(7.5, 5))
            bottom = np.zeros(len(edges) - 1)
            for k in range(len(CATS)):
                h = np.bincount(idx[ok & (cat == k)], minlength=len(edges) - 1) / np.maximum(tot, 1)
                ax.bar(edges[:-1], h, width=np.diff(edges), bottom=bottom, align="edge", color=COLORS[k],
                       label=CATS[k], edgecolor="none")
                bottom += h
            if logx:
                ax.set_xscale("log")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, 1)
            ax.set_xlabel(xl)
            ax.set_ylabel("fraction of simulated tracks")
            ax.set_title("first stage that failed%s" % tag, fontsize=11)
            ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
            pdf.savefig(fig)
            plt.close(fig)
        if (d["nPixClus"] >= 0).all():
            fig, ax = plt.subplots(figsize=(7, 5))
            b = np.arange(-0.5, 10.5)
            for msk, lab, col in [(found, "found", COLORS[0]), (lost, "lost", COLORS[-1])]:
                if msk.any():
                    ax.hist(d["nPixClusShared"][msk], bins=b, density=True, histtype="step", lw=2, color=col, label=lab)
            ax.set_xlabel("# pixel clusters shared with another TP")
            ax.set_ylabel("normalised")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        for c in dcSeeds:
            fig, ax = plt.subplots(figsize=(7, 5))
            b = np.geomspace(1e-4, 1, 41)
            for msk, lab, col in [(found, "found in generalTracks", COLORS[0]), (lost, "lost", COLORS[-1])]:
                if msk.any():
                    ax.hist(np.clip(d[c + "_dR"][msk], 1e-4, 0.999), bins=b, histtype="step", lw=2, color=col, label=lab)
            ax.axvline(a.dc_dr, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\Delta R$(nearest %s, TP)" % c)
            ax.set_ylabel("simulated tracks")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
    print("\nplots: %s" % a.out)


if __name__ == "__main__":
    main()
