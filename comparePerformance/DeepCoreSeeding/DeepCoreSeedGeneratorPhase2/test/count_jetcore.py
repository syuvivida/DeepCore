#!/usr/bin/env python3
"""
Per-event counts of jet-core seeds and tracks in step3 files (FWLite, run after cmsenv).

  python3 count_jetcore.py step3_base.root step3_deepcore.root [--max 20]

A collection that is not in a file is printed as '-' (check with edmDumpEventContent).
"DC in ..." counts tracks whose seed has no hits, i.e. tracks built from DeepCore seeds
(needs the TrackExtras and the seed collections in the file).
"""
import argparse
import ROOT
from DataFormats.FWLite import Events, Handle

ROOT.gROOT.SetBatch(True)

COLLS = [  # (label, C++ type, short name)
    ("jetCoreRegionalStepSeeds", "std::vector<TrajectorySeed>", "seeds"),
    ("jetCoreRegionalStepSeedsBarrel", "std::vector<TrajectorySeed>", "seedsDC"),
    ("jetCoreRegionalStepSeedsDeepCore", "std::vector<TrajectorySeed>", "seedsDC"),
    ("jetCoreRegionalStepSeedsEndcap", "std::vector<TrajectorySeed>", "seedsEC"),
    ("jetCoreRegionalStepBarrelTracks", "std::vector<reco::Track>", "trkDC"),
    ("jetCoreRegionalStepEndcapTracks", "std::vector<reco::Track>", "trkEC"),
    ("jetCoreRegionalStepTracks", "std::vector<reco::Track>", "trkJC"),
    ("generalTracks", "std::vector<reco::Track>", "general"),
]


def count(ev, label, typ, cache):
    h = cache.setdefault(label, Handle(typ))
    ev.getByLabel(label, h)
    return h.product().size() if h.isValid() else None


def jetcore_in_general(ev, cache):
    h = cache.setdefault("generalTracks", Handle("std::vector<reco::Track>"))
    ev.getByLabel("generalTracks", h)
    if not h.isValid():
        return None
    algo = ROOT.reco.TrackBase.algoByName("jetCoreRegionalStep")
    return sum(1 for t in h.product() if t.originalAlgo() == algo)


def hitless_seeded(ev, label, cache, jetcore_only=False):
    """tracks whose seed has no hits (= DeepCore seeds); needs TrackExtras and the seeds in the file"""
    h = cache.setdefault(label, Handle("std::vector<reco::Track>"))
    ev.getByLabel(label, h)
    if not h.isValid():
        return None
    algo = ROOT.reco.TrackBase.algoByName("jetCoreRegionalStep")
    n = 0
    for t in h.product():
        if jetcore_only and t.originalAlgo() != algo:
            continue
        try:
            ref = t.seedRef()
            # duplicate-merged tracks (algo duplicateMerge) have no seed: skip them
            if ref.isNull() or not ref.isAvailable():
                continue
            if ref.get().nHits() == 0:
                n += 1
        except Exception:
            continue
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max", type=int, default=20)
    a = ap.parse_args()
    per_file = []
    for f in a.files:
        cache, rows = {}, {}
        for i, ev in enumerate(Events(f)):
            if i >= a.max:
                break
            aux = ev.eventAuxiliary()
            key = (aux.run(), aux.luminosityBlock(), aux.event())
            r = {}
            for lab, typ, short in COLLS:
                n = count(ev, lab, typ, cache)
                if n is not None:
                    r[lab] = n
            r["jetCore in generalTracks"] = jetcore_in_general(ev, cache)
            r["DC in jcTracks"] = hitless_seeded(ev, "jetCoreRegionalStepTracks", cache)
            r["DC in general"] = hitless_seeded(ev, "generalTracks", cache, jetcore_only=True)
            rows[key] = r
        per_file.append((f, rows))

    keys = sorted(set.intersection(*[set(r.keys()) for _, r in per_file]))
    labels = []
    for _, rows in per_file:
        for r in rows.values():
            for k in r:
                if k not in labels:
                    labels.append(k)
    for f, rows in per_file:
        print("\n== %s" % f)
        print("%-16s" % "run:lumi:event" + "".join("%12s" % l.replace("jetCoreRegionalStep", "jc")[:11] for l in labels))
        tot = {l: 0 for l in labels}
        for k in keys:
            r = rows[k]
            print("%-16s" % ("%d:%d:%d" % k) + "".join("%12s" % ("-" if r.get(l) is None else r[l]) for l in labels))
            for l in labels:
                tot[l] += r.get(l) or 0
        n = max(len(keys), 1)
        print("%-16s" % "mean" + "".join("%12.1f" % (tot[l] / n) for l in labels))


if __name__ == "__main__":
    main()
