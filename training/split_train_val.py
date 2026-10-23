#!/usr/bin/env python3
"""
Split DeepCore ntuples into training / validation files by number of entries.

For every input file, the first X% of entries (default 80%) go to
  <outdir>/training/DeepCoreTrainingSample_<name>.root
and the rest go to
  <outdir>/validation/DeepCoreTrainingSample_<name>.root

- Only the HIGHEST cycle of the tree is read. Lower cycles (e.g. ;2 next to ;3)
  are AutoSave backups that contain a subset of the same entries.
- The cut is moved to the nearest event boundary (event_ID), so jets from the
  same event never end up in both training and validation. Disable with --no-event-boundary.
- Data is copied in chunks, so memory use stays small even for large files.
- Output keeps the same directory/tree name (DeepCoreNtuplizerTest/DeepCoreNtuplizerTree)
  and branch types, so DeepCore_GPU.py reads it unchanged.

Usage:
  python split_train_val.py "/eos/user/.../DeepCoreTrainingSample_*.root" -o /eos/user/.../TrainingSamples
  python split_train_val.py file1.root file2.root -o out --train-frac 0.8
Then:
  python DeepCore_GPU.py --training --epochs N \
      --trainPath "out/training/*.root" --valPath "out/validation/*.root"
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

DIR_NAME = "DeepCoreNtuplizerTest"
TREE_NAME = "DeepCoreNtuplizerTree"


def latest_tree_path(f, dir_name, tree_name):
    """Return 'dir/tree;N' for the highest cycle of the tree in file f."""
    cycles = []
    for k in f[dir_name].keys(cycle=True, recursive=False):
        name, _, c = k.partition(";")
        if name == tree_name and c:
            cycles.append(int(c))
    if not cycles:
        return None
    return "{}/{};{}".format(dir_name, tree_name, max(cycles))


def find_cut(tree, n_entries, train_frac, use_event_boundary):
    """Entry index where validation starts."""
    cut = int(round(n_entries * train_frac))
    if not use_event_boundary or "event_ID" not in tree.keys():
        return cut
    ev = tree["event_ID"].array(library="np")
    # indices where a new event starts
    starts = np.flatnonzero(np.diff(ev) != 0) + 1
    if len(starts) == 0:
        return cut
    # only meaningful if each event's jets are stored contiguously
    if len(np.unique(ev)) != len(starts) + 1:
        print("  note: event_ID entries are not contiguous, splitting on entries only")
        return cut
    return int(starts[np.argmin(np.abs(starts - cut))])


def entries_per_chunk(tree, branches, basket_mb):
    """Entries per chunk so the largest branch's UNCOMPRESSED basket stays below basket_mb.
    (uproot's "200 MB" step_size is based on compressed size; these ntuples are mostly zeros and
    compress ~100x, which produced >2 GiB baskets that ROOT cannot store.)"""
    max_bytes = 1
    for b in branches:
        try:
            max_bytes = max(max_bytes, tree[b].interpretation.numpy_dtype.itemsize)
        except AttributeError:  # jagged/object branch: fall back to a conservative guess
            max_bytes = max(max_bytes, 8 * 1024)
    return max(1, int(basket_mb * 1024 * 1024 // max_bytes))


def copy_range(tree, branches, start, stop, out_path, step_size, compression):
    """Copy entries [start, stop) of tree into a new file with the same dir/tree name."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with uproot.recreate(out_path, compression=compression) as fout:
        created = False
        for chunk in tree.iterate(branches, entry_start=start, entry_stop=stop,
                                  step_size=step_size, library="np"):
            # convert big-endian ROOT arrays to native byte order for writing
            chunk = {b: np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("=")) for b, a in chunk.items()}
            if not created:
                # Explicit TTree via mktree. Fixed-size array branches (cluster_measured[30][30][4], ...)
                # are declared as numpy subarray dtypes, the form uproot uses internally; the plain
                # (dtype, shape) tuple is rejected by some uproot versions (e.g. LCG_110a).
                # Do NOT use fout[name] = chunk: recent uproot writes an RNTuple that way, not a TTree.
                fout.mktree("{}/{}".format(DIR_NAME, TREE_NAME),
                            {b: (np.dtype((a.dtype, a.shape[1:])) if a.ndim > 1 else a.dtype)
                             for b, a in chunk.items()})
                created = True
            fout["{}/{}".format(DIR_NAME, TREE_NAME)].extend(chunk)
    return stop - start


def main():
    p = argparse.ArgumentParser(description="Split DeepCore ntuples into training/validation by entries")
    p.add_argument("inputs", nargs="+", help="input ROOT files or glob patterns (quote globs)")
    p.add_argument("-o", "--outdir", default="TrainingSamples", help="output base directory")
    p.add_argument("--train-frac", type=float, default=0.8, help="fraction of entries for training (default 0.8)")
    p.add_argument("--prefix", default="DeepCoreTrainingSample_", help="output file name prefix")
    p.add_argument("--branches", nargs="*", default=None,
                   help="branches to keep (default: all). The training only needs: "
                        "cluster_measured jet_eta jet_pt trackPar trackProb")
    p.add_argument("--no-event-boundary", action="store_true",
                   help="cut exactly at train-frac instead of the nearest event boundary")
    p.add_argument("--basket-mb", type=float, default=100.0,
                   help="max UNCOMPRESSED size (MB) of one written basket for the largest branch (default 100). "
                        "ROOT forbids baskets above 2 GiB; the number of entries per chunk is derived from this.")
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        matched = sorted(glob.glob(os.path.expanduser(pat)))
        files += matched if matched else ([pat] if os.path.exists(pat) else [])
    files = list(dict.fromkeys(files))  # drop duplicates, keep order
    if not files:
        sys.exit("No input files found for: {}".format(args.inputs))
    if not 0 < args.train_frac < 1:
        sys.exit("--train-frac must be between 0 and 1")


    train_dir = os.path.join(args.outdir, "training")
    val_dir = os.path.join(args.outdir, "validation")
    tot_train = tot_val = 0

    for fn in files:
        base = os.path.splitext(os.path.basename(fn))[0]
        if base.startswith(args.prefix):
            base = base[len(args.prefix):]
        with uproot.open(fn) as f:
            path = latest_tree_path(f, DIR_NAME, TREE_NAME) if DIR_NAME in f else None
            if path is None:
                print("SKIP {}: no {}/{}".format(fn, DIR_NAME, TREE_NAME))
                continue
            tree = f[path]
            n = tree.num_entries
            branches = args.branches or list(tree.keys())
            step = entries_per_chunk(tree, branches, args.basket_mb)
            cut = find_cut(tree, n, args.train_frac, not args.no_event_boundary)
            print("{}  [{}]  entries={}  -> training {} ({:.1%}), validation {}  (chunk = {} entries)".format(
                fn, path, n, cut, cut / n if n else 0, n - cut, step))

            # compression: same algorithm/level as the input when possible
            try:
                comp = uproot.compression.Compression.from_code(f.file.fCompress)
            except Exception:
                comp = uproot.ZLIB(1)

            tot_train += copy_range(tree, branches, 0, cut,
                                    os.path.join(train_dir, "{}{}.root".format(args.prefix, base)), step, comp)
            tot_val += copy_range(tree, branches, cut, n,
                                  os.path.join(val_dir, "{}{}.root".format(args.prefix, base)), step, comp)

    tot = tot_train + tot_val
    if tot:
        print("\nTotal: {} entries -> training {} ({:.1%}), validation {} ({:.1%})".format(
            tot, tot_train, tot_train / tot, tot_val, tot_val / tot))
        print("Outputs in {}/ and {}/".format(train_dir, val_dir))


if __name__ == "__main__":
    main()
