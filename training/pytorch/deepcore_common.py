"""
Common code for the PyTorch DeepCore training and testing scripts.

  * window geometry (Phase-2 by default; dimensions are read from the ntuple itself)
  * DeepCore 2.2 network in PyTorch (fully convolutional: works for 120x45 and 30x30 windows)
  * losses (same maths as the TensorFlow version)
  * uproot-5 readers: streaming IterableDataset (multi-file, multi-cycle) and in-memory loader

Data layout (as written by DeepCoreNtuplizer, Phase-2 version):
  cluster_measured [N][X][Y][Nlayer]           X = 120, Y = 45 for Phase-2 (30x30 for Phase-1)
  trackPar         [N][X][Y][Nover][Npar+1]    (dx, dy, deta, dphi, pt, zero_flag); the flag is the loss weight
  trackProb        [N][X][Y][Nover]
  jet_eta, jet_pt  [N]
The network keeps the same channels-last layout at its outputs:
  par  [N][X][Y][Nover][Npar+1],  prob [N][X][Y][Nover][2]   (only [...,0] of prob is used)
"""
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uproot
# fork-safe file access: uproot's default (fsspec) source uses a background thread that does not survive fork(),
# which deadlocks the DataLoader workers.  The memory-map source has no threads.
UPROOT_OPTS = dict(handler=uproot.source.file.MemmapSource)

INPUT_MODULE = "DeepCoreNtuplizerTest"
INPUT_TREE = os.environ.get("DEEPCORE_TREE", "DeepCoreNtuplizerPhase2Tree")
BRANCHES_IN = ["cluster_measured", "jet_eta", "jet_pt"]
BRANCHES_TGT = ["trackPar", "trackProb"]

OVERLAP_NUM = 3
PAR_NUM = 5

# pixel pitch (cm) and approximate barrel-layer radii (cm, plotting only)
GEOMETRY = {
    "phase2": dict(pitchX=0.0025, pitchY=0.010, layerR=[3.0, 6.1, 10.4, 14.6]),   # TBPX, 25x100 um
    "phase1": dict(pitchX=0.0100, pitchY=0.015, layerR=[3.0, 6.8, 10.2, 16.0]),   # BPix, 100x150 um
}

# Phase-1 standardization of the track parameters used in the TF loss (kept as a fallback)
PHASE1_NORM = dict(mean=[0., 0., 0., 0., 95.], sd=[0.404, 0.478, 1.9, 2., 150.])


# ------------------------------------------------------------------------------------------------
# network
# ------------------------------------------------------------------------------------------------
def _conv(cin, cout, k):
    return nn.Conv2d(cin, cout, k, padding=k // 2)  # "same" padding for odd kernels


class DeepCoreNet(nn.Module):
    """DeepCore 2.2 architecture (same layers as the Keras model), channels-first internally."""

    def __init__(self, lay_num=4, overlap_num=OVERLAP_NUM, par_num=PAR_NUM):
        super().__init__()
        self.lay_num, self.overlap_num, self.par_num = lay_num, overlap_num, par_num
        self.trunk = nn.Sequential(
            _conv(lay_num + 2, 50, 7), nn.ReLU(),
            _conv(50, 40, 5), nn.ReLU(),
            _conv(40, 40, 5), nn.ReLU(),
            _conv(40, 30, 5), nn.ReLU(),
            _conv(30, 30, 3), nn.ReLU(),
        )
        self.par_head = nn.Sequential(
            _conv(30, 18, 3), nn.ReLU(),
            _conv(18, 18, 3), nn.ReLU(),
            _conv(18, 18, 3), nn.ReLU(),
            _conv(18, overlap_num * (par_num + 1), 3),
        )
        self.prob_head = nn.Sequential(
            _conv(30, 30, 3), nn.ReLU(),
            _conv(30, 30, 3), nn.ReLU(),
            _conv(30, 30, 3), nn.ReLU(),
            _conv(30, overlap_num * 2, 3),  # logits; sigmoid applied in the loss / at inference
        )

    def forward(self, clusters, jet_eta, jet_pt):
        """clusters [N, L, X, Y]; jet_eta, jet_pt [N].
        Returns par [N, X, Y, ov, par+1] and prob *logits* [N, X, Y, ov, 2] (Keras Reshape ordering)."""
        n, _, nx, ny = clusters.shape
#        jet = torch.stack([jet_eta, jet_pt], dim=1)[:, :, None, None].expand(n, 2, nx, ny)  # = UpSampling2D
        jet = torch.stack([jet_eta, jet_pt / 1000.], dim=1)[:, :, None, None].expand(n, 2, nx, ny)
        h = self.trunk(torch.cat([clusters, jet], dim=1))
        par = self.par_head(h).permute(0, 2, 3, 1).reshape(n, nx, ny, self.overlap_num, self.par_num + 1)
        logit = self.prob_head(h).permute(0, 2, 3, 1).reshape(n, nx, ny, self.overlap_num, 2)
        return par, logit


class DeepCoreExport(nn.Module):
    """Wrapper with the interface of the TensorFlow graph used by CMSSW DeepCoreSeedGenerator:
    inputs  input_1 = jet eta [N,1], input_2 = jet pt [N,1], input_3 = clusters [N,X,Y,L] (channels last)
    outputs output_node0 = par [N,X,Y,ov,par+1], output_node1 = probability [N,X,Y,ov,2] (after sigmoid)"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_1, input_2, input_3):
        par, logit = self.net(input_3.permute(0, 3, 1, 2), input_1[:, 0], input_2[:, 0])
        return par, torch.sigmoid(logit)


# ------------------------------------------------------------------------------------------------
# losses (same maths as the TF/Keras version)
# ------------------------------------------------------------------------------------------------
def loss_par(pred, target, mean, inv_sd):
    """loss_mse_select_clipped: standardized, clipped at +-5 sigma, weighted by the zero_flag column."""
    wei = target[..., -1:]
    p = (pred[..., :-1] - mean) * inv_sd
    t = (target[..., :-1] - mean) * inv_sd
    out = torch.clamp(p - t, -5., 5.) ** 2 * wei
    return out.sum() / (wei.sum() * (pred.shape[-1] - 1) + 1e-5)  # (pred.shape[-1]-1) = parNum = 5


def loss_prob(logit, target_prob, wei, soft=False, pos_weight=10.):
    """loss_ROI_crossentropy (soft=False) / loss_ROIsoft_crossentropy (soft=True).
    weighted_cross_entropy_with_logits(pos_weight) == BCEWithLogits(pos_weight)."""
    pw = torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit[..., 0:1], target_prob, pos_weight=pw, reduction="none")
    if soft:
        w = wei + 0.01
        return (bce * w).sum() / w.sum()
    return (bce * wei).sum() / (wei.sum() + 1e-5)


# ------------------------------------------------------------------------------------------------
# input files
# ------------------------------------------------------------------------------------------------
def tree_cycles(filename):
    with uproot.open(filename, **UPROOT_OPTS) as f:
        if INPUT_MODULE not in f:
            return []
        cyc = []
        for k in f[INPUT_MODULE].keys(cycle=True, recursive=False):
            name, _, c = k.partition(";")
            if name == INPUT_TREE and c:
                cyc.append(int(c))
    return sorted(set(cyc))


def tree_sources(pattern, cycles_mode="latest"):
    """glob pattern (or comma-separated list) -> list of dict(file, tree, start, stop, n)."""
    files = []
    for p in pattern.split(","):
        files += sorted(glob.glob(os.path.expandvars(os.path.expanduser(p.strip()))))
    out = []
    for fn in files:
        cyc = tree_cycles(fn)
        if not cyc:
            print("WARNING: no {}/{} in {}, skipped".format(INPUT_MODULE, INPUT_TREE, fn))
            continue
        if cycles_mode == "latest":
            cyc = cyc[-1:]
        for c in cyc:
            obj = "{}/{};{}".format(INPUT_MODULE, INPUT_TREE, c)
            with uproot.open(fn, **UPROOT_OPTS) as f:
                n = f[obj].num_entries
            out.append(dict(file=fn, tree=obj, start=0, stop=n, n=n))
    return out


def split_sources(sources, val_frac):
    """Use the last val_frac of the entries of each tree for validation (like Keras validation_split)."""
    train, val = [], []
    for s in sources:
        cut = s["start"] + int(round((s["stop"] - s["start"]) * (1 - val_frac)))
        train.append(dict(s, stop=cut, n=cut - s["start"]))
        val.append(dict(s, start=cut, n=s["stop"] - cut))
    return [s for s in train if s["n"] > 0], [s for s in val if s["n"] > 0]


def n_entries(sources):
    return sum(s["n"] for s in sources)


def window_shape(sources):
    """(jetDimX, jetDimY, nLayers) read from the first entry."""
    s = sources[0]
    with uproot.open(s["file"], **UPROOT_OPTS) as f:
        a = f[s["tree"]]["cluster_measured"].array(library="np", entry_start=s["start"], entry_stop=s["start"] + 1)
    a = np.asarray(a[0]) if a.dtype == object else a[0]
    return tuple(a.shape)


def _as_regular(arr):
    if arr.dtype == object:
        arr = np.stack([np.asarray(a) for a in arr])
    return arr


def read_chunks(source, branches, step_size, lay_num):
    """yield dicts of float32 numpy arrays for one source."""
    for chunk in uproot.iterate({source["file"]: source["tree"]}, branches, step_size=step_size,
                                library="np", entry_start=source["start"], entry_stop=source["stop"], **UPROOT_OPTS):
        out = {b: _as_regular(chunk[b]) for b in branches}
        if "cluster_measured" in out:
            out["cluster_measured"] = out["cluster_measured"][..., :lay_num]
        yield {b: v.astype(np.float32, copy=False) for b, v in out.items()}


class DeepCoreDataset(torch.utils.data.IterableDataset):
    """Streams events from ROOT files; each DataLoader worker reads a different subset of the trees.
    Shuffling is done within a buffer of `shuffle_buffer` events (0 = no shuffling)."""

    def __init__(self, sources, lay_num, with_target=True, step_size=256, shuffle_buffer=0, seed=0):
        super().__init__()
        self.sources, self.lay_num, self.with_target = sources, lay_num, with_target
        self.step_size, self.shuffle_buffer, self.seed = step_size, shuffle_buffer, seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def _events(self, sources):
        branches = BRANCHES_IN + (BRANCHES_TGT if self.with_target else [])
        for s in sources:
            for c in read_chunks(s, branches, self.step_size, self.lay_num):
                for i in range(len(c["jet_pt"])):
                    ev = (c["cluster_measured"][i].transpose(2, 0, 1),  # [L, X, Y]
                          c["jet_eta"][i], c["jet_pt"][i])
                    if self.with_target:
                        ev += (c["trackPar"][i], c["trackProb"][i][..., None], c["trackPar"][i][..., -1:])
                    yield ev

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        srcs = self.sources
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + (info.id if info else 0))
        if self.shuffle_buffer:
            srcs = [srcs[i] for i in rng.permutation(len(srcs))]
        if info is not None:
            srcs = srcs[info.id::info.num_workers]
        if not self.shuffle_buffer:
            yield from self._events(srcs)
            return
        buf = []
        for ev in self._events(srcs):
            buf.append(ev)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.integers(len(buf)))
        rng.shuffle(buf)
        yield from buf


def to_device(batch, device):
    return [torch.as_tensor(np.asarray(b) if not torch.is_tensor(b) else b).to(device, non_blocking=True)
            for b in batch]


def load_arrays(sources, lay_num, with_target=True, max_events=-1):
    """Read everything into memory (testing). Returns dict of numpy arrays, channels-last as in the ntuple."""
    branches = BRANCHES_IN + (BRANCHES_TGT if with_target else [])
    parts = {b: [] for b in branches}
    n = 0
    for s in sources:
        for c in read_chunks(s, branches, 256, lay_num):
            k = len(c["jet_pt"]) if max_events < 0 else min(len(c["jet_pt"]), max_events - n)
            for b in branches:
                parts[b].append(c[b][:k])
            n += k
            if 0 <= max_events <= n:
                break
        if 0 <= max_events <= n:
            break
    return {b: np.concatenate(v) for b, v in parts.items()}


# ------------------------------------------------------------------------------------------------
# normalization of the track parameters (computed on the Phase-2 sample, saved in the checkpoint)
# ------------------------------------------------------------------------------------------------
def compute_norm(sources, lay_num, n_events=300):
    """mean and standard deviation of (dx, dy, deta, dphi, pt) over the pixels with weight (zero_flag) = 1."""
    s1 = np.zeros(PAR_NUM)
    s2 = np.zeros(PAR_NUM)
    cnt = 0
    seen = 0
    for s in sources:
        for c in read_chunks(s, ["trackPar"], 64, lay_num):
            tp = c["trackPar"].reshape(-1, PAR_NUM + 1)
            sel = tp[:, -1] > 0
            v = tp[sel, :PAR_NUM].astype(np.float64)
            s1 += v.sum(0)
            s2 += (v ** 2).sum(0)
            cnt += len(v)
            seen += len(c["trackPar"])
            if seen >= n_events:
                break
        if seen >= n_events:
            break
    if cnt == 0:
        print("WARNING: no weighted pixels found, using the Phase-1 normalization")
        return dict(PHASE1_NORM)
    mean = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    mean[:4] = 0.  # position/angle offsets are centred by construction, as in the original loss
    return dict(mean=mean.tolist(), sd=sd.tolist())


def norm_tensors(norm, device):
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    inv_sd = 1. / torch.tensor(norm["sd"], dtype=torch.float32, device=device)
    return mean, inv_sd


# ------------------------------------------------------------------------------------------------
# helpers shared by training (extra validation) and testing
# ------------------------------------------------------------------------------------------------
def prop_on_layer(x1, y1, eta, phi, eta_jet, lay, geom):
    """linear propagation from layer 2 to layer `lay` (0-3), in pixel units; plotting only."""
    g = GEOMETRY[geom]
    theta_jet = 2 * math.atan(math.exp(-eta_jet))
    theta = 2 * math.atan(math.exp(-(eta + eta_jet)))
    dist = g["layerR"][lay] - g["layerR"][1]
    distx, disty = dist / g["pitchX"], dist / g["pitchY"]
    y_out = disty * math.sin(theta - theta_jet) / (math.sin(theta_jet) * math.sin(theta)) + y1
    x_out = distx * math.tan(-phi) + x1
    return x_out, y_out

"""
def eff_fake(par_pred, prob_pred, tpar, tprob, prob_thr, lay_dist=3):
    d = par_pred[..., :4] - tpar[..., :4]
    scale = math.atan(2 / float(lay_dist * 3))
    chi2 = d[..., 0] ** 2 + d[..., 1] ** 2 + (d[..., 2] ** 2 + d[..., 3] ** 2) / scale
    is_t = tprob == 1
    is_p = prob_pred > prob_thr
    n_tot_eff = max(1, len(tprob) * 3)
    n_pred = is_p.sum()
    res = {}
    for c in (4, 8):
        res["eff_%d" % c] = float(((chi2 <= c) & is_t & is_p).sum()) / n_tot_eff
        res["fake_%d" % c] = float(((chi2 >= c) & is_t & is_p).sum()) / n_pred if n_pred else 1.
    return res
"""


def eff_fake(par_pred, prob_pred, tpar, tprob, prob_thr, lay_dist=3, near=2):
    """Track-level efficiency and fake rate.

    A prediction (pixel, slot with prob > prob_thr) is GOOD if it lies in the region of interest of a true track
    (flag column of trackPar = 1, i.e. within `near` (=distThr) pixels of a true crossing) and its predicted
    parameters match that track's parameters stored at that pixel (chi2 <= cut).  Neighbouring pixels of the same
    track therefore count as good (duplicates of one seed), not as fakes.
      efficiency = true crossings with at least one good prediction within +-near pixels / true crossings
      fake rate  = predictions that are not good / all predictions
      dup        = good predictions per found track (how many seeds one track gets before any cleaning)
    """
    d = par_pred[..., :4] - tpar[..., :4]
    scale = math.atan(2 / float(lay_dist * 3))
    chi2 = d[..., 0] ** 2 + d[..., 1] ** 2 + (d[..., 2] ** 2 + d[..., 3] ** 2) / scale
    flag = tpar[..., -1] > 0                        # region of interest of a true track
    is_t = tprob == 1                               # exact crossing pixel (label)
    is_p = prob_pred > prob_thr
    n_true, n_pred = max(1, int(is_t.sum())), int(is_p.sum())
    res = {}
    for c in (4, 8):
        good = is_p & flag & (chi2 <= c)                         # [N, X, Y, slot]
        g = good.any(axis=-1)                                    # any slot, [N, X, Y]
        # dilate by +-near pixels: is there a good prediction within the neighbourhood of each pixel?
        gp = np.pad(g, ((0, 0), (near, near), (near, near)))
        X, Y = g.shape[1], g.shape[2]
        dil = np.zeros_like(g)
        for dx in range(-near, near + 1):
            for dy in range(-near, near + 1):
                dil |= gp[:, near + dx:near + dx + X, near + dy:near + dy + Y]
        found = (is_t & dil[..., None]).sum()
        res["eff_%d" % c] = float(found) / n_true
        res["fake_%d" % c] = float((is_p & ~good).sum()) / n_pred if n_pred else 1.
        res["dup_%d" % c] = float(good.sum()) / max(1, int(found))
    return res


