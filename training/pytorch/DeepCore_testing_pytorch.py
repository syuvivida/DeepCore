#!/usr/bin/env python3
#################################################################################################
##
##  DeepCore: PREDICTION and VALIDATION, PyTorch version, Phase-2 DeepCoreNtuplizer output
##  (window size read from the ntuple: 120 x 45 for Phase-2, 30 x 30 for Phase-1)
##
##  Environment: an LCG view with torch + uproot (+ ROOT for --output), e.g.
##     source /cvmfs/sft.cern.ch/lcg/views/LCG_xxx_cuda/x86_64-el9-gcc*-opt/setup.sh
##
##  Workflow:
##   1) python DeepCore_testing_pytorch.py --predict --input NTUPLE.root --weights DeepCore_ckpt_ep020.pt
##        -> runs the NN and writes DeepCore_prediction_ev<N>.npz
##   2) python DeepCore_testing_pytorch.py --output  --input NTUPLE.root
##        -> validation ROOT file + PDF plots (needs PyROOT)
##   or in one go:  --predict --output
##
##   Extra options: --onData, --drawOnly, --averageValueTest, --testSampleBuild, --rgb, --maxEvents
##
##  Memory: a Phase-2 event is ~1 MB of input+target and ~0.5 MB of prediction; use --maxEvents.
#################################################################################################
import os
import sys
import math
import argparse
from array import array as array2

import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepcore_common as dc

parser = argparse.ArgumentParser(description="DeepCore NN prediction and validation (PyTorch)")
parser.add_argument("--predict", dest="Predict", action="store_true", help="run the NN prediction on --input using --weights")
parser.add_argument("--output", dest="Output", action="store_true", help="validation ROOT file and PDF plots (needs ROOT)")
parser.add_argument("--input", dest="Input", required=True, help="input ntuple (ROOT file, glob allowed)")
parser.add_argument("--weights", dest="Weights", default="", help="checkpoint (.pt) written by the training script")
parser.add_argument("--cycles", dest="Cycles", default="latest", choices=["all", "latest"])
parser.add_argument("--maxEvents", type=int, default=-1, help="read at most this many windows (-1 = all)")
parser.add_argument("--batchSize", type=int, default=64)
parser.add_argument("--geometry", default="phase2", choices=["phase2", "phase1"])
parser.add_argument("--layNum", type=int, default=4)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--onData", dest="onData", action="store_true", help="running on data (no target available)")
parser.add_argument("--drawOnly", dest="drawOnly", action="store_true", help="draw input/target only, without prediction")
parser.add_argument("--averageValueTest", dest="averageValueTest", action="store_true", help="average ADC / occupancy and fraction of 1 in the target")
parser.add_argument("--testSampleBuild", dest="testSampleBuild", action="store_true", help="save a small test sample (.npz)")
parser.add_argument("--rgb", dest="rgb", action="store_true", help="RGB colour scheme plots")
args = parser.parse_args()

ON_DATA, DRAW_ONLY, RGB = args.onData, args.drawOnly, args.rgb
geo = dc.GEOMETRY[args.geometry]
pitchX, pitchY = geo["pitchX"], geo["pitchY"]

#------------------------------------------------------------------------------------------#
#----------------------------- INTERNAL CONFIGURATION PARAMETERS --------------------------#
#------------------------------------------------------------------------------------------#
overlapNum, parNum = dc.OVERLAP_NUM, dc.PAR_NUM
valSplit = 0.2      # fraction treated as "validation" (event offset of the maps)
prob_thr = 0.7      # DeepCore 2.2 thresholds
dth = [0.0, 0.15, 0.3]
dth1 = [0.1, 0.05, 0]
numPrint = 5        # events saved in the ROOT file
outEvent = 30       # complete RGB plots for this event only
jetNum_test = 50    # events in the test sample (--testSampleBuild)

#-----------------------------------------------------------------------------#
#----------------------------------INPUT -------------------------------------#
#-----------------------------------------------------------------------------#
sources = dc.tree_sources(args.Input, args.Cycles)
if not sources:
    sys.exit("ERROR: no {}/{} in {}".format(dc.INPUT_MODULE, dc.INPUT_TREE, args.Input))
jetDimX, jetDimY, nLayFile = dc.window_shape(sources)
layNum = min(args.layNum, nLayFile)
halfX, halfY = jetDimX / 2., jetDimY / 2.
nTot = dc.n_entries(sources)
nRead = nTot if args.maxEvents < 0 else min(nTot, args.maxEvents)
print("window {} x {} pixels, {} layers; reading {} of {} windows (~{:.1f} GB)".format(
    jetDimX, jetDimY, layNum, nRead, nTot, nRead * jetDimX * jetDimY * (layNum + overlapNum * 7) * 4 / 1e9))

data = dc.load_arrays(sources, layNum, with_target=not ON_DATA, max_events=args.maxEvents)
input_ = data["cluster_measured"]                 # [N, X, Y, L]
input_jeta, input_jpt = data["jet_eta"], data["jet_pt"]
if not ON_DATA:
    target_ = data["trackPar"]                    # [N, X, Y, ov, par+1]
    target_prob = data["trackProb"]               # [N, X, Y, ov]
    target_wei = target_[..., -1]                 # [N, X, Y, ov]
del data
print("loading data: completed, events =", len(input_jeta))

jetNum = int(len(input_jeta) * (1 - valSplit))  # used only to name the output files

if args.testSampleBuild and not ON_DATA:
    start = max(0, len(input_) - jetNum_test - 5)
    sel = slice(start, start + jetNum_test)
    np.savez("DeepCore_ev{}_test".format(jetNum_test), input_=input_[sel], input_jeta=input_jeta[sel],
             input_jpt=input_jpt[sel], target_=target_[sel], target_prob=target_prob[sel])
    print("test sample saved")

if args.averageValueTest:
    if not ON_DATA:
        n_wei = (target_wei[..., 0] != 0).sum(axis=(1, 2))
        n_one = ((target_prob[..., 0] == 1) & (target_wei[..., 0] != 0)).sum(axis=(1, 2))
        aver1 = np.where(n_wei > 0, n_one / np.maximum(n_wei, 1), 0.).mean()
        print("average of the number of 1", aver1, ", multiplicative factor to 1", 1 / aver1 if aver1 else float("inf"))
    nz = input_[input_ != 0]
    print("average value of input=", nz.mean() if nz.size else 0, ", average occupancy=", nz.size / float(input_.size))
    plt.figure()
    plt.hist(nz, 100, facecolor="green")
    plt.title("Non-zero pixel charge distribution")
    plt.xlabel("charge [/chargeNorm]")
    plt.ylabel("entries")
    plt.grid(True)
    plt.savefig("ADC_count.pdf")
    plt.close()

#--------------------------------------------------------------------------------------------------------#
#----------------------------------------------- PREDICTION ---------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
pred_file = "DeepCore_prediction_ev{}.npz".format(jetNum)
if args.Predict:
    if not args.Weights:
        sys.exit("ERROR: --predict needs --weights CHECKPOINT.pt")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.Weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if cfg and (cfg.get("jetDimX"), cfg.get("jetDimY")) != (jetDimX, jetDimY):
        print("WARNING: model trained on {}x{} windows, input is {}x{}".format(cfg.get("jetDimX"), cfg.get("jetDimY"), jetDimX, jetDimY))
    model = dc.DeepCoreNet(lay_num=cfg.get("layNum", layNum)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("prediction: start (device {}, weights {})".format(device, args.Weights))
    validation_par = np.zeros(target_.shape if not ON_DATA else (len(input_), jetDimX, jetDimY, overlapNum, parNum + 1), np.float32)
    validation_prob = np.zeros(validation_par.shape[:-1], np.float32)
    with torch.no_grad():
        for i in range(0, len(input_), args.batchSize):
            sl = slice(i, i + args.batchSize)
            clus = torch.from_numpy(input_[sl]).permute(0, 3, 1, 2).to(device)
            par, logit = model(clus, torch.from_numpy(input_jeta[sl]).to(device), torch.from_numpy(input_jpt[sl]).to(device))
            validation_par[sl] = par.cpu().numpy()
            validation_prob[sl] = torch.sigmoid(logit[..., 0]).cpu().numpy()
    np.savez(pred_file, validation_par=validation_par, validation_prob=validation_prob)
    print("prediction: completed ->", pred_file)

#------------------------------------------------------------------------------------------#
#------------------------------------- OUTPUT ROOT FILE -----------------------------------#
#------------------------------------------------------------------------------------------#
if args.Output:
    if (not args.Predict) and (not DRAW_ONLY):
        loadpred = np.load(pred_file)
        validation_par, validation_prob = loadpred["validation_par"], loadpred["validation_prob"]
        if validation_prob.ndim == 5:  # file written by the TF script
            validation_prob = validation_prob[..., 0]
        print("prediction loaded from", pred_file)

    import ROOT
    from ROOT import gStyle, gROOT, TColor
    gROOT.Reset()
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)

    validation_offset = int(len(input_) * (1 - valSplit) + 1)
    if validation_offset + numPrint > len(input_):
        validation_offset = max(0, len(input_) - numPrint)
    nPrint = min(numPrint, len(input_))

    def tgt_pix(tp):
        """target/prediction (dx, dy) are in units of 0.01 cm (100 um): convert to pixels"""
        return float(tp[0]) * 0.01 / pitchX, float(tp[1]) * 0.01 / pitchY

    canvasTot, canvasProb, mapTot, graphTargetTot, mapProbPredTot, graphPredTot = [], [], [], [], [], []
    cw, ch = 1000, int(1000 * jetDimY / jetDimX) + 200
    for jet in range(nPrint):
        canvasProb.append([ROOT.TCanvas("canvasProb_%d_%d" % (jet, t), "", cw, ch) for t in range(overlapNum)])
        mapProbPredTot.append([ROOT.TH2F("mapProbPredTot_%d_%d" % (jet, t), "", jetDimX, -halfX, halfX, jetDimY, -halfY, halfY) for t in range(overlapNum)])
        mapTot.append([ROOT.TH2F("mapTot_%d_%d" % (jet, l), "", jetDimX, -halfX, halfX, jetDimY, -halfY, halfY) for l in range(layNum)])
        canvasTot.append([ROOT.TCanvas("canvasTot_%d_%d" % (jet, l), "", cw, ch) for l in range(layNum)])
        graphTargetTot.append([ROOT.TGraph() for l in range(layNum)])
        graphPredTot.append([ROOT.TGraph() for l in range(layNum)])

    for jet in range(nPrint):
        print("=================================== New Event ======================================")
        j_eff = jet + validation_offset
        jeta = float(input_jeta[j_eff])
        for lay in range(layNum):
            for g, col, sty in ((graphPredTot[jet][lay], 7, 28), (graphTargetTot[jet][lay], 6, 2)):
                g.SetMarkerColor(col)
                g.SetMarkerStyle(sty)
                g.SetMarkerSize(2)
            for x, y in zip(*np.nonzero(input_[j_eff][:, :, lay])):
                mapTot[jet][lay].SetBinContent(int(x) + 1, int(y) + 1, float(input_[j_eff][x, y, lay]))
        if not DRAW_ONLY:
            for trk in range(overlapNum):
                for x, y in zip(*np.nonzero(validation_prob[j_eff][:, :, trk] > 1e-4)):
                    mapProbPredTot[jet][trk].SetBinContent(int(x) + 1, int(y) + 1, float(validation_prob[j_eff][x, y, trk]))
        tarPoint = predPoint = 0
        for x in range(jetDimX):
            for y in range(jetDimY):
                for trk in range(overlapNum):
                    if (not ON_DATA) and target_prob[j_eff][x][y][trk] == 1:
                        tp = target_[j_eff][x][y][trk]
                        xx, yy = tgt_pix(tp)
                        x1, y1 = x + xx - halfX, y + yy - halfY
                        graphTargetTot[jet][1].SetPoint(tarPoint, x1, y1)
                        for lay in (0, 2, 3):
                            graphTargetTot[jet][lay].SetPoint(tarPoint, *dc.prop_on_layer(x1, y1, tp[2] * 0.01, tp[3] * 0.01, jeta, lay, args.geometry))
                        tarPoint += 1
                    if (not DRAW_ONLY) and validation_prob[j_eff][x][y][trk] > (prob_thr - 0.14 * trk):
                        vp = validation_par[j_eff][x][y][trk]
                        xx, yy = tgt_pix(vp)
                        x1, y1 = x + xx - halfX, y + yy - halfY
                        graphPredTot[jet][1].SetPoint(predPoint, x1, y1)
                        for lay in (0, 2, 3):
                            graphPredTot[jet][lay].SetPoint(predPoint, *dc.prop_on_layer(x1, y1, vp[2] * 0.01, vp[3] * 0.01, jeta, lay, args.geometry))
                        predPoint += 1
                        print("New Pred, bin (x,y):", x - halfX, y - halfY,
                              "prediction(x,y,eta,phi,pt)=", np.round(vp[:5], 4), "prob=", round(float(validation_prob[j_eff][x][y][trk]), 3))
                        if not ON_DATA:
                            print("    target(x,y,eta,phi,pt)=", np.round(target_[j_eff][x][y][trk][:5], 4), "prob=", target_prob[j_eff][x][y][trk])

    output_file = ROOT.TFile("DeepCore_mapValidation_ev{}.root".format(jetNum), "recreate")
    if RGB:
        NCont = 10
        array_of_palette = []
        for red, green, blue in [([1., 1.], [1., 0.], [1., 0.]), ([1., 0.], [1., 0.], [1., 0.]),
                                 ([1., 0.], [1., 1.], [1., 0.]), ([1., 0.], [1., 0.], [1., 1.])]:
            FI = TColor.CreateGradientColorTable(2, array2("d", [0., 1.]), array2("d", red), array2("d", green), array2("d", blue), NCont)
            array_of_palette.append(array2("i", [FI + i for i in range(NCont)]))

    def cms_label():
        lat = ROOT.TLatex()
        lat.SetNDC()
        lat.SetTextSize(0.045)
        lat.DrawLatex(0.10, 0.92, "#bf{CMS} #scale[0.7]{#it{Simulation Preliminary}}")
        lat.DrawLatex(0.78, 0.92, "#bf{14 TeV}")
        return lat

    def style(h, title, ztitle):
        h.SetTitle(title)
        h.GetXaxis().SetTitle("x [pixel, %g #mum]" % (pitchX * 1e4))
        h.GetYaxis().SetTitle("y [pixel, %g #mum]" % (pitchY * 1e4))
        h.GetZaxis().SetTitle(ztitle)
        for a in (h.GetXaxis(), h.GetYaxis()):
            a.SetTitleSize(0.05)

    keep = []
    for jet in range(nPrint):
        for lay in range(layNum):
            canvasTot[jet][lay].cd()
            canvasTot[jet][lay].SetRightMargin(0.15)
            h = mapTot[jet][lay]
            style(h, "Pixel Window, layer %d" % (lay + 1), "charge [/chargeNorm]")
            if RGB:
                gStyle.SetPalette(NCont, array_of_palette[lay % len(array_of_palette)])
            h.Draw("colz")
            keep.append(cms_label())
            if jet == outEvent and RGB:
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_input_layer%d_event%d.pdf" % (lay, jet))
            if not ON_DATA:
                graphTargetTot[jet][lay].Draw("SAME P")
            graphPredTot[jet][lay].Draw("SAME P")
            leg = ROOT.TLegend(0.12, 0.80, 0.30, 0.88)
            if not ON_DATA:
                leg.AddEntry(graphTargetTot[jet][lay], "Target", "p")
            leg.AddEntry(graphPredTot[jet][lay], "Prediction", "p")
            leg.Draw("SAME")
            keep.append(leg)
            canvasTot[jet][lay].Write()
            if jet == outEvent and RGB:
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_crosses_layer%d_event%d.pdf" % (lay, jet))
        for trk in range(overlapNum):
            canvasProb[jet][trk].cd()
            canvasProb[jet][trk].SetRightMargin(0.15)
            gStyle.SetPalette(57)
            h = mapProbPredTot[jet][trk]
            style(h, "TCP Prediction Map, overlap %d" % trk, "Probability")
            h.GetZaxis().SetRangeUser(0, 1)
            h.Draw("colz")
            if not ON_DATA:
                graphTargetTot[jet][1].Draw("SAME P")
            graphPredTot[jet][1].Draw("SAME P")
            keep.append(cms_label())
            canvasProb[jet][trk].Write()
        if jet == outEvent and RGB:
            canvasProb[jet][0].SaveAs("Probability_crosses_event%d.pdf" % jet)
    output_file.Close()
    print("validation ROOT file written")

    #------------------ parameter distributions and residuals (vectorized) ------------------#
    if not ON_DATA:
        print("parameter file: start")
        pdf_par = PdfPages("parameter_file_ev{}.pdf".format(jetNum))
        if DRAW_ONLY:
            sel = (target_prob == 1)
        else:  # DeepCore 2.2 selection, only "associated" TCPs
            l2off = ~(input_[..., 1] > 0.0001)
            l134off = ~np.any(input_[..., [0, 2, 3]] > 0.0001, axis=3) if layNum >= 4 else np.zeros_like(l2off)
            thr = (prob_thr + np.asarray(dth)[None, None, None, :] + 0.3 * l2off[..., None]
                   + np.asarray(dth1)[None, None, None, :] * l134off[..., None])
            sel = (validation_prob > thr) & (target_prob == 1)
        tgt_sel = target_[sel]
        pred_sel = None if DRAW_ONLY else validation_par[sel]
        print("selected TCPs:", len(tgt_sel))

        # ranges in um follow the pitch: x +-8 pixels, y +-3 pixels
        rx, ry = 8 * pitchX * 1e4, 3 * pitchY * 1e4
        titles = [r"$\Delta x$", r"$\Delta y$", r"$\Delta \eta$", r"$\Delta \phi$", r"$p_T$"]
        for par in range(parNum):
            tg = tgt_sel[:, par]
            pr = None if DRAW_ONLY else pred_sel[:, par]
            if par in (0, 1):      # units of 0.01 cm -> um
                bt, bp, br = tg * 100, (None if DRAW_ONLY else pr * 100), (None if DRAW_ONLY else (pr - tg) * 100)
                rng = (-rx, rx) if par == 0 else (-ry, ry)
            elif par in (2, 3):    # units of 0.01
                bt, bp, br = tg * 0.01, (None if DRAW_ONLY else pr * 0.01), (None if DRAW_ONLY else (pr - tg) * 0.01)
                rng = (-0.03, 0.03)
            else:                  # pt, relative residual
                bt, bp = tg, pr
                if not DRAW_ONLY:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        br = (pr - tg) / tg
                    br = br[np.isfinite(br)]
                rng = (-3, 12)
            if not DRAW_ONLY:
                n_ok = int(np.sum(pr * tg > 0))
                print("Parameter {}: correct sign {}/{} = {:.3f}, residual mean {:.4g}, sd {:.4g}".format(
                    par, n_ok, len(tg), n_ok / max(len(tg), 1), br.mean() if len(br) else 0, br.std() if len(br) else 0))
                plt.figure()
                plt.hist(br, 100, range=rng, facecolor="darkorange", alpha=0.75)
                plt.title("Residual Distribution - " + titles[par], fontsize=20)
                plt.xlabel(r"Prediction-Target [$\mu$m]" if par < 2 else ("Prediction-Target" if par < 4 else "(Prediction-Target)/Target"), fontsize=16)
                plt.ylabel("TCP", fontsize=16)
                if len(br):
                    plt.text(0.62, 0.85, "mean = %.3g\nsd = %.3g" % (br.mean(), br.std()), transform=plt.gca().transAxes)
                plt.grid(True)
                if RGB:
                    plt.savefig("residual_{}_{}.pdf".format(jetNum, par), bbox_inches="tight")
                pdf_par.savefig(bbox_inches="tight")
                plt.close()
            trng = (-rx, rx) if par == 0 else (-ry, ry) if par == 1 else (-0.05, 0.05) if par < 4 else (0, 500)
            plt.figure()
            plt.hist(bt, 100, range=trng, facecolor="royalblue", alpha=0.75)
            plt.title("Target Distribution - " + titles[par], fontsize=20)
            plt.xlabel(r"Target [$\mu$m]" if par < 2 else ("Target" if par < 4 else "Target [GeV]"), fontsize=16)
            plt.ylabel("TCP", fontsize=16)
            plt.grid(True)
            pdf_par.savefig(bbox_inches="tight")
            plt.close()
            if DRAW_ONLY:
                continue
            plt.figure()
            plt.hist(bp, 100, range=trng, facecolor="red", alpha=0.75)
            plt.title("Prediction Distribution - " + titles[par], fontsize=20)
            plt.grid(True)
            pdf_par.savefig(bbox_inches="tight")
            plt.close()
            plt.figure()
            plt.hist2d(bt, bp, bins=150, range=[trng, trng], cmap=plt.cm.viridis)
            plt.plot(trng, trng, color="black")
            plt.xlabel("Target " + titles[par], fontsize=16)
            plt.ylabel("Predicted " + titles[par], fontsize=16)
            plt.colorbar()
            if RGB:
                plt.savefig("predVStarget_{}_{}.pdf".format(jetNum, par), bbox_inches="tight")
            pdf_par.savefig(bbox_inches="tight")
            plt.close()
        pdf_par.close()
        print("parameter file: completed")
