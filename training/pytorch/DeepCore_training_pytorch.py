#!/usr/bin/env python3
##############################################################################################
##  DeepCore TRAINING, PyTorch version, for the Phase-2 DeepCoreNtuplizer output
##  (120 x 45 window of 25 x 100 um pixels; the window size is read from the ntuple, so
##   Phase-1 30 x 30 ntuples also work)
##
##  Environment (lxplus-gpu, EL9), e.g.:
##      source /cvmfs/sft.cern.ch/lcg/views/LCG_xxx_cuda/x86_64-el9-gcc*-opt/setup.sh   # has torch + uproot
##   or a venv:  pip install torch uproot awkward pandas matplotlib onnx
##
##  Usage:
##   central input (recommended):
##     python DeepCore_training_pytorch.py --trainPath "/eos/.../training/*.root" \
##                                         --valPath   "/eos/.../validation/*.root" --epochs 20
##   local single file (last 20% used for validation):
##     python DeepCore_training_pytorch.py --input ntuple.root --epochs 5
##   continue a training:
##     python DeepCore_training_pytorch.py ... --continueTraining --weights DeepCore_ckpt_ep010.pt \
##                                         --csv loss_plots_0_10.csv --lr 1e-4 --loss ROIsoft
##   export for CMSSW (ONNX, same input/output layout as the TF graph):
##     python DeepCore_training_pytorch.py --weights DeepCore_ckpt_ep020.pt --exportONNX DeepCore_phase2.onnx
##
##  Training recipe of the Run-2 barrel network (for orientation, Phase-2 needs its own tuning):
##     LR 2e-4 -> 1e-4 -> 7e-5 with loss ROI, then ROIsoft with LR 7e-5 -> 1e-6 -> 1e-7
##
##  Prediction and validation plots: DeepCore_testing_pytorch.py
##############################################################################################
import os
import sys
import json
import math
import time
import argparse

_nthreads = str(min(8, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", _nthreads)
os.environ.setdefault("MKL_NUM_THREADS", _nthreads)

import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepcore_common as dc

parser = argparse.ArgumentParser(description="DeepCore NN training (PyTorch)")
parser.add_argument("--input", dest="Input", default="", help="local input ROOT file (last valSplit used for validation)")
parser.add_argument("--trainPath", dest="TrainPath", default="", help='glob of training files, e.g. "/eos/.../training/*.root"')
parser.add_argument("--valPath", dest="ValPath", default="", help="glob of validation files")
parser.add_argument("--cycles", dest="Cycles", default="latest", choices=["all", "latest"],
                    help="TTree cycles to read in each file (default: only the highest cycle)")
parser.add_argument("--valSplit", type=float, default=0.2, help="validation fraction for --input")
parser.add_argument("--geometry", default="phase2", choices=["phase2", "phase1"], help="pixel geometry (plots only)")
parser.add_argument("--layNum", type=int, default=4, help="number of layers used from cluster_measured")
parser.add_argument("--epochs", dest="Epochs", type=int, default=2, help="number of epochs to train")
parser.add_argument("--continueTraining", dest="continueTraining", action="store_true", help="continue from --weights")
parser.add_argument("--weights", dest="Weights", default="", help="checkpoint (.pt) to continue from / to export")
parser.add_argument("--epochsstart", dest="EpochsStart", type=int, default=-1,
                    help="first epoch number (default: taken from the checkpoint, 0 for a new training)")
parser.add_argument("--csv", dest="Csv", default="", help="csv of previous losses, to plot all epochs")
parser.add_argument("--batchSize", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (Adam)")
parser.add_argument("--loss", default="ROI", choices=["ROI", "ROIsoft"],
                    help="probability loss: ROI (early training) or ROIsoft (late training)")
parser.add_argument("--posWeight", type=float, default=10.,
                    help="pos_weight of the probability cross entropy (10 in the TF code); the 120x45 window "
                         "has 6x more empty pixels than 30x30, so a larger value may be needed")
parser.add_argument("--norm", default="auto", choices=["auto", "phase1"],
                    help="standardization of the track parameters in the loss: computed from the training "
                         "sample (auto) or the Phase-1 constants")
parser.add_argument("--normEvents", type=int, default=300, help="events used to compute the normalization")
parser.add_argument("--stepsPerEpoch", type=int, default=0, help="limit the number of batches per epoch (0 = full pass)")
parser.add_argument("--valSteps", type=int, default=0, help="limit the number of validation batches (0 = full pass)")
parser.add_argument("--shuffleBuffer", type=int, default=2000, help="events in the shuffling buffer (0 = no shuffle)")
parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--amp", action="store_true", help="mixed precision on GPU")
parser.add_argument("--cpu", action="store_true", help="force CPU")
parser.add_argument("--deb1ev", dest="deb1ev", action="store_true", help="debug: train on a single event")
parser.add_argument("--checkSample", dest="checkSample", action="store_true", help="check for NaN/inf in the inputs before training")
parser.add_argument("--extraValidation", dest="extraValidation", action="store_true",
                    help="chi2-based efficiency / fake rate on validation events after each epoch")
parser.add_argument("--extraValEvents", type=int, default=200, help="events used by --extraValidation")
parser.add_argument("--probThr", type=float, default=0.85, help="probability threshold for --extraValidation")
parser.add_argument("--exportONNX", default="", help="write the model in --weights (or the trained one) to ONNX")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
print("device:", device, (torch.cuda.get_device_name(0) if device.type == "cuda" else ""))

# ------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------
if args.Input:
    print("WARNING: using local data (also for training!)")
    all_src = dc.tree_sources(args.Input, args.Cycles)
    if not all_src:
        sys.exit("ERROR: no {}/{} in {}".format(dc.INPUT_MODULE, dc.INPUT_TREE, args.Input))
    train_src, val_src = dc.split_sources(all_src, args.valSplit)
elif args.TrainPath:
    train_src = dc.tree_sources(args.TrainPath, args.Cycles)
    val_src = dc.tree_sources(args.ValPath, args.Cycles) if args.ValPath else []
    if not train_src:
        sys.exit("ERROR: no training trees found with --trainPath " + args.TrainPath)
elif not args.exportONNX:
    sys.exit("ERROR: give --input FILE.root or --trainPath/--valPath")
else:
    train_src, val_src = [], []

if args.deb1ev and train_src:
    train_src = [dict(train_src[0], stop=train_src[0]["start"] + 1, n=1)]
    val_src = list(train_src)

ckpt = None
if args.Weights:
    ckpt = torch.load(args.Weights, map_location="cpu", weights_only=False)
    print("loaded checkpoint", args.Weights, "(epoch {})".format(ckpt.get("epoch", "?")))

if train_src:
    jetDimX, jetDimY, nLayFile = dc.window_shape(train_src)
elif ckpt is not None:
    jetDimX, jetDimY, nLayFile = ckpt["config"]["jetDimX"], ckpt["config"]["jetDimY"], ckpt["config"]["layNum"]
else:
    sys.exit("ERROR: nothing to do")
layNum = min(args.layNum, nLayFile)
print("window: {} x {} pixels, {} layers (file has {})".format(jetDimX, jetDimY, layNum, nLayFile))
if args.geometry == "phase2" and (jetDimX, jetDimY) == (30, 30):
    print("WARNING: 30x30 window with --geometry phase2: is this a Phase-1 ntuple? (only plots are affected)")

n_train, n_val = dc.n_entries(train_src), dc.n_entries(val_src)
print("training events =", n_train, ", validation events =", n_val)

# ------------------------------------------------------------------------------------------------
# model, normalization, optimizer
# ------------------------------------------------------------------------------------------------
model = dc.DeepCoreNet(lay_num=layNum).to(device)
if ckpt is not None:
    model.load_state_dict(ckpt["model"])
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

if ckpt is not None and "norm" in ckpt:
    norm = ckpt["norm"]
elif args.norm == "phase1" or not train_src:
    norm = dict(dc.PHASE1_NORM)
else:
    print("computing the track-parameter normalization on {} events ...".format(args.normEvents))
    norm = dc.compute_norm(train_src, layNum, args.normEvents)
print("normalization: mean =", np.round(norm["mean"], 4), " sd =", np.round(norm["sd"], 4))
mean_t, inv_sd_t = dc.norm_tensors(norm, device)

config = dict(jetDimX=jetDimX, jetDimY=jetDimY, layNum=layNum, overlapNum=dc.OVERLAP_NUM, parNum=dc.PAR_NUM,
              geometry=args.geometry)


def export_onnx(path):
    import torch.onnx
    exp = dc.DeepCoreExport(model).to("cpu").eval()
    model.to("cpu")
    dummy = (torch.zeros(1, 1), torch.full((1, 1), 500.), torch.zeros(1, jetDimX, jetDimY, layNum))
    kw = dict(input_names=["input_1", "input_2", "input_3"], output_names=["output_node0", "output_node1"],
              dynamic_axes={n: {0: "batch"} for n in ["input_1", "input_2", "input_3", "output_node0", "output_node1"]},
              opset_version=17)
    try:  # classic TorchScript exporter (newer torch defaults to dynamo, which needs onnxscript)
        torch.onnx.export(exp, dummy, path, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(exp, dummy, path, **kw)
    with open(os.path.splitext(path)[0] + "_config.json", "w") as f:
        json.dump(dict(config, norm=norm), f, indent=1)
    print("ONNX model written to", path)
    model.to(device)


if args.exportONNX and not train_src:
    export_onnx(args.exportONNX)
    sys.exit(0)

if args.continueTraining and ckpt is None:
    sys.exit("ERROR: --continueTraining needs --weights CHECKPOINT.pt")
if args.EpochsStart >= 0:
    start_epoch = args.EpochsStart
elif args.continueTraining:
    start_epoch = int(ckpt.get("epoch", 0))
else:
    start_epoch = 0
epochs = args.Epochs

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.continueTraining and ckpt is not None and "optimizer" in ckpt:
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        for g in optimizer.param_groups:  # the learning rate on the command line wins
            g["lr"] = args.lr
    except ValueError as e:
        print("WARNING: optimizer state not restored:", e)
scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
soft = (args.loss == "ROIsoft")
print("loss: par = mse_select_clipped, prob = {}_crossentropy (pos_weight={}), lr = {}".format(args.loss, args.posWeight, args.lr))

# ------------------------------------------------------------------------------------------------
# data loaders
# ------------------------------------------------------------------------------------------------
def make_loader(sources, shuffle):
    ds = dc.DeepCoreDataset(sources, layNum, with_target=True, shuffle_buffer=(args.shuffleBuffer if shuffle else 0))
    nw = min(args.workers, len(sources))
    # "fork" workers on Linux (Python >= 3.14 defaults to "forkserver", which re-runs this script)
    ctx = "fork" if (nw > 0 and sys.platform.startswith("linux")) else None
    return ds, torch.utils.data.DataLoader(ds, batch_size=args.batchSize, num_workers=nw,
                                           pin_memory=(device.type == "cuda"), persistent_workers=False,
                                           multiprocessing_context=ctx)


if args.checkSample:
    print("checksample")
    for s in train_src + val_src:
        for c in dc.read_chunks(s, dc.BRANCHES_IN + dc.BRANCHES_TGT, 256, layNum):
            for b, v in c.items():
                if not np.all(np.isfinite(v)):
                    print("file", s["file"], s["tree"], b, ": NaN/inf found")
    print("checksample: done")

train_ds, train_loader = make_loader(train_src, shuffle=not args.deb1ev)
val_ds, val_loader = make_loader(val_src, shuffle=False) if val_src else (None, None)


def run_epoch(loader, train, max_steps):
    model.train(train)
    tot = np.zeros(3)
    nb = 0
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if max_steps and i >= max_steps:
                break
            clus, eta, pt, tpar, tprob, wei = dc.to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                par, logit = model(clus, eta, pt)
            par, logit = par.float(), logit.float()
            lp = dc.loss_par(par, tpar, mean_t, inv_sd_t)
            lc = dc.loss_prob(logit, tprob, wei, soft=soft, pos_weight=args.posWeight)
            loss = lp + lc  # loss_weights = [1, 1]
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot += [loss.item(), lp.item(), lc.item()]
            nb += 1
    return tot / max(nb, 1), nb


def extra_validation():
    data = dc.load_arrays(val_src or train_src, layNum, True, args.extraValEvents)
    model.eval()
    pars, probs = [], []
    with torch.no_grad():
        for i in range(0, len(data["jet_pt"]), args.batchSize):
            sl = slice(i, i + args.batchSize)
            clus = torch.from_numpy(data["cluster_measured"][sl]).permute(0, 3, 1, 2).to(device)
            par, logit = model(clus, torch.from_numpy(data["jet_eta"][sl]).to(device),
                               torch.from_numpy(data["jet_pt"][sl]).to(device))
            pars.append(par.cpu().numpy())
            probs.append(torch.sigmoid(logit[..., 0]).cpu().numpy())
    return dc.eff_fake(np.concatenate(pars), np.concatenate(probs), data["trackPar"], data["trackProb"], args.probThr)


# ------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------
history = {k: [] for k in ["epoch", "loss", "val_loss", "par_loss", "val_par_loss", "prob_loss", "val_prob_loss"]}
extra = {k: [] for k in ["eff_4", "fake_4", "eff_8", "fake_8"]}
print("training: start, epochs {} -> {}".format(start_epoch, start_epoch + epochs))
for ep in range(start_epoch, start_epoch + epochs):
    t0 = time.time()
    train_ds.set_epoch(ep)
    tr, nb = run_epoch(train_loader, True, args.stepsPerEpoch)
    if val_loader is not None:
        va, nvb = run_epoch(val_loader, False, args.valSteps)
    else:
        va, nvb = np.full(3, np.nan), 0
    for k, v in zip(history, [ep + 1, tr[0], va[0], tr[1], va[1], tr[2], va[2]]):
        history[k].append(v)
    print("epoch {:4d}  {:5.0f}s  loss {:.5f} (par {:.5f}, prob {:.5f})  val_loss {:.5f} (par {:.5f}, prob {:.5f})  [{} / {} batches]"
          .format(ep + 1, time.time() - t0, tr[0], tr[1], tr[2], va[0], va[1], va[2], nb, nvb))
    if args.extraValidation:
        r = extra_validation()
        for k in extra:
            extra[k].append(r[k])
        print("   extra validation:", {k: round(v, 4) for k, v in r.items()})
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=ep + 1, norm=norm,
                    config=config, val_loss=float(va[0])),
               "DeepCore_ckpt_ep{:03d}.pt".format(ep + 1))

final_name = "DeepCore_train_ev{ev}_ep{ep}.pt".format(ev=n_train, ep=start_epoch + epochs)
torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=start_epoch + epochs,
                norm=norm, config=config), final_name)
print("training: completed, model saved in", final_name)

# ------------------------------------------------------------------------------------------------
# loss plots (csv keeps the same columns as the TF version)
# ------------------------------------------------------------------------------------------------
arr = np.array([history[k] for k in history]).T
np.savetxt("loss_plots_{}_{}.csv".format(start_epoch, start_epoch + epochs), arr, delimiter=",")
if args.continueTraining and args.Csv:
    arr_full = np.vstack((np.loadtxt(args.Csv, delimiter=",", ndmin=2), arr))
    np.savetxt("loss_plots_full_0_{}.csv".format(start_epoch + epochs), arr_full, delimiter=",")
else:
    arr_full = arr

with PdfPages("loss_file_ep{}_ev{}.pdf".format(start_epoch + epochs, n_train)) as pdf_loss:
    for cols, title in [((1, 2), "Model Loss"), ((3, 4), "TP Loss"), ((5, 6), "TCP Loss")]:
        plt.figure()
        plt.yscale("log")
        plt.plot(arr_full[:, 0], arr_full[:, cols[0]], marker="o")
        plt.plot(arr_full[:, 0], arr_full[:, cols[1]], marker="o")
        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(["train", "validation"], loc="upper right")
        pdf_loss.savefig(bbox_inches="tight")
        plt.close()
    if args.extraValidation:
        for keys, title in [(("eff_4", "eff_8"), "Efficiency of track finding"), (("fake_4", "fake_8"), "Fake Rate")]:
            plt.figure()
            for k in keys:
                plt.plot(history["epoch"], extra[k], marker="o")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.legend(["thr=4", "thr=8"])
            pdf_loss.savefig(bbox_inches="tight")
            plt.close()
print("loss plots written")

if args.exportONNX:
    try:
        export_onnx(args.exportONNX)
    except Exception as e:
        print("WARNING: ONNX export failed ({}); the checkpoint {} is saved, export later with "
              "--weights {} --exportONNX FILE.onnx".format(e, final_name, final_name))
