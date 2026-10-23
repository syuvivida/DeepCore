#!/usr/bin/env python3
#################################################################################################
##
##  DeepCore: PREDICTION and VALIDATION only (training code removed)
##
##  Ported to the lxplus el9 stack (e.g. LCG_110a): Python 3.13, Keras 3 / TensorFlow 2.x,
##  uproot 5, ROOT 6.
##
##  Environment:
##    source /cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh
##    (or the CPU view /cvmfs/sft.cern.ch/lcg/views/LCG_110a/x86_64-el9-gcc13-opt/setup.sh)
##
##  Workflow:
##   1) python DeepCore.py --predict --input INPUTNAME.root [--weights FILE]
##        -> runs the NN and writes DeepCore_prediction_ev<N>.npz
##   2) python DeepCore.py --output  --input INPUTNAME.root
##        -> reads the .npz and produces the validation ROOT file + PDF plots (needs ROOT)
##   or in one go:
##      python DeepCore.py --predict --output --input INPUTNAME.root
##
##   Extra options: --onData, --drawOnly, --averageValueTest, --testSampleBuild, --rgb
##
#################################################################################################

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['GOTO_NUM_THREADS'] = '40'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['openmp'] = 'True'

import math
import argparse
from array import array as array2

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pylab = plt  # the old 'pylab' calls are all available in pyplot

import uproot


parser = argparse.ArgumentParser(description="DeepCore NN prediction and validation script (no training)")
parser.add_argument('--predict',          dest='Predict',          action='store_true', help='run the NN prediction on --input using --weights')
parser.add_argument('--output',           dest='Output',           action='store_true', help='produce the validation ROOT file and PDF plots (needs ROOT)')
parser.add_argument('--input',            dest='Input',            action='store', required=True, type=str, help='input ntuple (ROOT file)')
parser.add_argument('--weights',          dest='Weights',          action='store', type=str,
                    default='/eos/user/s/syu/DeepCore/training/DeepCore_train_ev34721_ep10.weights.h5',
                    help='trained weights to load for the prediction')
parser.add_argument('--batchSize',        dest='batchSize',        action='store', type=int, default=64, help='batch size used for the prediction')
parser.add_argument('--onData',           dest='onData',           action='store_true', help='running on data (no target available)')
parser.add_argument('--drawOnly',         dest='drawOnly',         action='store_true', help='draw input/target info only, without prediction')
parser.add_argument('--averageValueTest', dest='averageValueTest', action='store_true', help='print average ADC count / occupancy and fraction of 1-values in the target')
parser.add_argument('--testSampleBuild',  dest='testSampleBuild',  action='store_true', help='save a small local test sample (.npz)')
parser.add_argument('--rgb',              dest='rgb',              action='store_true', help='RGB color scheme plots (used in all the presented results)')
args = parser.parse_args()

OUTPUT = args.Output
PREDICT = args.Predict
input_name = args.Input
ON_DATA = args.onData
DRAW_ONLY = args.drawOnly
AVERAGE_VALUES_TEST = args.averageValueTest
TEST_SAMPLE_BUILD = args.testSampleBuild
RGB = args.rgb

#------------------------------------------------------------------------------------------#
#----------------------------- INTERNAL CONFIGURATION PARAMETERS --------------------------#
#------------------------------------------------------------------------------------------#

jetDim = 30        # window size on the pixel layer (fixed by the sample)
overlapNum = 3     # number of overlaps (fixed by the sample)
layNum = 4         # 4 for barrel; 7 for barrel+endcap (fixed by the sample)
parNum = 5         # number of track parameters (fixed by the sample)
inputModuleName = "DeepCoreNtuplizerTest"
inputTreeName = "DeepCoreNtuplizerTree"
# 2017 ntuples:
# inputModuleName = "demo"
# inputTreeName = "NNClustSeedInputSimHitTree"

valSplit = 0.2     # fraction of the input treated as "validation" (used for the event offset in the maps)

# DeepCore 2.2 thresholds
prob_thr = 0.7
dth = [0.0, 0.15, 0.3]
dth1 = [0.1, 0.05, 0]

# plotting configuration
numPrint = 5       # number of events saved in the ROOT file
outEvent = 30      # complete RGB plots for this event only
jetNum_test = 50   # events in the test sample (--testSampleBuild)


#--------------------------------------------------------------------------#
#------------------------- FUNCTIONS --------------------------------------#
#--------------------------------------------------------------------------#

def read_branch(tree, name):
    """uproot5 replacement for the old uproot3 tree.array(name). Returns a regular numpy array."""
    arr = tree[name].array(library="np")
    if arr.dtype == object:  # jagged/object storage -> stack into a regular array
        arr = np.stack([np.asarray(a) for a in arr])
    return arr


def prop_on_layer(x1, y1, eta, phi, eta_jet, lay):
    """linear propagation to the 4 barrel layers, plotting purpose only"""
    theta_jet = 2*math.atan(math.exp(-eta_jet))
    eta = eta+eta_jet
    theta = 2*math.atan(math.exp(-eta))

    dist = {0: 3-6.8, 1: 6.8-6.8, 2: 10.2-6.8, 3: 16-6.8}[lay]
    distx = dist/0.01
    disty = dist/0.015

    y_out = disty*math.sin(theta-theta_jet)/(math.sin(theta_jet)*math.sin(theta))+y1
    x_out = distx*math.tan(-phi)+x1
    return (x_out, y_out)


def test_sample_creation(input_, input_jeta, input_jpt, target_, target_prob):
    print("testing sample creation: ...")
    start = len(input_)-jetNum_test-5
    sel = slice(start, start+jetNum_test)
    np.savez("DeepCore_ev{ev}_test".format(ev=jetNum_test),
             input_=input_[sel], input_jeta=input_jeta[sel], input_jpt=input_jpt[sel],
             target_=target_[sel], target_prob=target_prob[sel][..., 0])
    print("..completed")


def average_1_eval(target_prob):
    print("evaluation of the number of 1, input len=", len(target_prob))
    wei = target_prob[:, :, :, 0, 1] != 0
    ones = (target_prob[:, :, :, 0, 0] == 1) & wei
    n_wei = wei.sum(axis=(1, 2))
    n_ones = ones.sum(axis=(1, 2))
    frac = np.where(n_wei > 0, n_ones/np.maximum(n_wei, 1), 0.)
    aver1 = frac.mean()
    print("average of the number of 1", aver1)
    print("Multiplicative factor to 1", 1/aver1 if aver1 else float('inf'))


def averageADC(input_):
    print("evaluation of average ADC count")
    nonzero = input_[input_ != 0]
    print("average value of input=", nonzero.mean() if nonzero.size else 0)
    print("average occupancy=", nonzero.size/float(input_.size))

    plt.figure()
    plt.hist(nonzero, 100, facecolor='green')
    plt.title('Non-zero ADC count distribution')
    plt.ylabel('entries')
    plt.xlabel('ADC count')
    plt.grid(True)
    plt.savefig("ADC_count.pdf")


def build_model():
    """DeepCore 2.2 architecture (must match the weights that are loaded)"""
    from keras.layers import Input, Conv2D, Reshape, UpSampling2D, concatenate
    from keras.models import Model

    NNinputs_jeta = Input(shape=(1,))
    NNinputs_jpt = Input(shape=(1,))
    NNinputsJet = concatenate([NNinputs_jeta, NNinputs_jpt])
    jetReshaped = Reshape((1, 1, 2))(NNinputsJet)
    jetUps = UpSampling2D(size=(jetDim, jetDim), data_format="channels_last")(jetReshaped)
    NNinputs = Input(shape=(jetDim, jetDim, layNum))
    ComplInput = concatenate([NNinputs, jetUps], axis=3)

    conv30_9 = Conv2D(50, 7, data_format="channels_last", activation='relu', padding="same")(ComplInput)
    conv30_7 = Conv2D(40, 5, data_format="channels_last", activation='relu', padding="same")(conv30_9)
    conv30_5 = Conv2D(40, 5, data_format="channels_last", activation='relu', padding="same")(conv30_7)
    conv20_5 = Conv2D(30, 5, data_format="channels_last", activation='relu', padding="same")(conv30_5)
    conv15_5 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv20_5)

    conv15_3_1 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_5)
    conv15_3_2 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_3_1)
    conv15_3_3 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_3_2)
    conv15_3 = Conv2D(18, 3, data_format="channels_last", padding="same")(conv15_3_3)
    reshaped = Reshape((jetDim, jetDim, overlapNum, parNum+1))(conv15_3)

    conv12_3_1 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv15_5)
    conv1_3_2 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv12_3_1)
    conv1_3_3 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv1_3_2)
    conv1_3_1 = Conv2D(6, 3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    reshaped_prob = Reshape((jetDim, jetDim, overlapNum, 2))(conv1_3_1)

    return Model([NNinputs, NNinputs_jeta, NNinputs_jpt], [reshaped, reshaped_prob])


#-----------------------------------------------------------------------------#
#----------------------------------INPUT -------------------------------------#
#-----------------------------------------------------------------------------#

print("loading data: start")
with uproot.open(input_name) as tfile:
    tree = tfile[inputModuleName][inputTreeName]
    input_ = read_branch(tree, "cluster_measured")[:, :, :, 0:layNum]  # read only layNum layers
    input_jeta = read_branch(tree, "jet_eta")
    input_jpt = read_branch(tree, "jet_pt")
    if not ON_DATA:
        target_ = read_branch(tree, "trackPar")
        target_prob = read_branch(tree, "trackProb")
        wei = target_[:, :, :, :, -1:]
        nev = len(target_prob)
        target_prob = np.reshape(target_prob, (nev, jetDim, jetDim, overlapNum, 1))
        target_prob = np.concatenate([target_prob, wei], axis=4)
print("loading data: completed")

jetNum = int(len(input_jeta)*(1-valSplit))  # used only to name the output files
print("total number of events =", len(input_jeta), ", jetNum (file naming) =", jetNum)

if TEST_SAMPLE_BUILD and not ON_DATA:
    test_sample_creation(input_, input_jeta, input_jpt, target_, target_prob)
if AVERAGE_VALUES_TEST:
    if not ON_DATA:
        average_1_eval(target_prob)
    averageADC(input_)


#--------------------------------------------------------------------------------------------------------#
#----------------------------------------------- PREDICTION ---------------------------------------------#
#--------------------------------------------------------------------------------------------------------#

if PREDICT:
    model = build_model()
    model.summary()

    print("loading weights:", args.Weights)
    model.load_weights(args.Weights)

    print("prediction: start")
    validation_par, validation_prob = model.predict(
        [input_.astype(np.float32),
         input_jeta.reshape(-1, 1).astype(np.float32),
         input_jpt.reshape(-1, 1).astype(np.float32)],
        batch_size=args.batchSize, verbose=2)
    validation_par = np.float64(validation_par)
    np.savez("DeepCore_prediction_ev{ev}".format(ev=jetNum), validation_par=validation_par, validation_prob=validation_prob)
    print("prediction: completed")


#------------------------------------------------------------------------------------------#
#------------------------------------- OUTPUT ROOT FILE -----------------------------------#
#------------------------------------------------------------------------------------------#

if OUTPUT:
    if (not PREDICT) and (not DRAW_ONLY):
        print("prediction loading: start")
        loadpred = np.load("DeepCore_prediction_ev{ev}.npz".format(ev=jetNum))
        validation_par = loadpred['validation_par']
        validation_prob = loadpred['validation_prob']
        print("prediction loading: completed")

    # drop the weight column -> shape (nev, jetDim, jetDim, overlapNum)
    if not ON_DATA:
        target_prob = target_prob[:, :, :, :, 0]
    if not DRAW_ONLY:
        validation_prob = validation_prob[:, :, :, :, 0]

    import ROOT
    from ROOT import gStyle, gROOT, TColor
    gROOT.Reset()
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)
    validation_offset = int(len(input_)*(1-valSplit)+1)
    if validation_offset+numPrint > len(input_):
        validation_offset = max(0, len(input_)-numPrint)

    half = jetDim/2
    canvasTot, canvasProb, mapTot, graphTargetTot, mapProbPredTot, graphPredTot = [], [], [], [], [], []

    for jet in range(numPrint):
        canvasProb.append([ROOT.TCanvas("canvasProb_%d_%d" % (jet, trk), "canvasProb_%d_%d" % (jet, trk), 500, 800) for trk in range(overlapNum)])
        mapProbPredTot.append([ROOT.TH2F("mapProbPredTot_%d_%d" % (jet, trk), "mapProbPredTot_%d_%d" % (jet, trk), jetDim, -half, half, jetDim, -half, half) for trk in range(overlapNum)])
        mapTot.append([ROOT.TH2F("mapTot_%d_%d" % (jet, lay), "mapTot_%d_%d" % (jet, lay), jetDim, -half, half, jetDim, -half, half) for lay in range(layNum)])
        canvasTot.append([ROOT.TCanvas("canvasTot_%d_%d" % (jet, lay), "canvasTot_%d_%d" % (jet, lay), 500, 800) for lay in range(layNum)])
        graphTargetTot.append([ROOT.TGraph() for lay in range(layNum)])
        graphPredTot.append([ROOT.TGraph() for lay in range(layNum)])

    # fill the histos
    for jet in range(numPrint):
        print("=================================== New Event ======================================")
        j_eff = jet+validation_offset
        jeta = float(input_jeta[j_eff])

        for lay in range(layNum):
            tarPoint = 0
            predPoint = 0
            graphPredTot[jet][lay].SetMarkerColor(7)
            graphPredTot[jet][lay].SetMarkerStyle(28)
            graphPredTot[jet][lay].SetMarkerSize(3)
            graphTargetTot[jet][lay].SetMarkerColor(6)
            graphTargetTot[jet][lay].SetMarkerStyle(2)
            graphTargetTot[jet][lay].SetMarkerSize(3)
            for x in range(jetDim):
                for y in range(jetDim):
                    pix = float(input_[j_eff][x][y][lay])
                    mapTot[jet][lay].SetBinContent(x+1, y+1, pix)
                    if pix > 0:
                        print("input pixel:", "(x,y)=", x, y, ", layer=", lay, ", value=", pix)
                    for trk in range(overlapNum):
                        if not DRAW_ONLY:
                            mapProbPredTot[jet][trk].SetBinContent(x+1, y+1, float(validation_prob[j_eff][x][y][trk]))
                        if (not ON_DATA) and target_prob[j_eff][x][y][trk] == 1 and lay == 1:
                            tp = target_[j_eff][x][y][trk]
                            xx = float(tp[0])/0.01*0.01  # normaliz. factor
                            yy = float(tp[1])/0.015*0.01
                            x1, y1 = x+xx-half, y+yy-half
                            graphTargetTot[jet][lay].SetPoint(tarPoint, x1, y1)
                            x0, y0 = prop_on_layer(x1, y1, tp[2]*0.01, tp[3]*0.01, jeta, 0)
                            x2, y2 = prop_on_layer(x1, y1, tp[2]*0.01, tp[3]*0.01, jeta, 2)
                            x3, y3 = prop_on_layer(x1, y1, tp[2]*0.01, tp[3]*0.01, jeta, 3)
                            graphTargetTot[jet][0].SetPoint(tarPoint, x0, y0)
                            graphTargetTot[jet][2].SetPoint(tarPoint, x2, y2)
                            graphTargetTot[jet][3].SetPoint(tarPoint, x3, y3)
                            tarPoint += 1
                        # T1017 threshold
                        if (not DRAW_ONLY) and validation_prob[j_eff][x][y][trk] > (prob_thr-0.14*trk) and lay == 1:
                            vp = validation_par[j_eff][x][y][trk]
                            xx_pr = float(vp[0])/0.01*0.01
                            yy_pr = float(vp[1])/0.015*0.01
                            x1, y1 = x+xx_pr-half, y+yy_pr-half
                            graphPredTot[jet][lay].SetPoint(predPoint, x1, y1)
                            x0, y0 = prop_on_layer(x1, y1, vp[2]*0.01, vp[3]*0.01, jeta, 0)
                            x2, y2 = prop_on_layer(x1, y1, vp[2]*0.01, vp[3]*0.01, jeta, 2)
                            x3, y3 = prop_on_layer(x1, y1, vp[2]*0.01, vp[3]*0.01, jeta, 3)
                            graphPredTot[jet][0].SetPoint(predPoint, x0, y0)
                            graphPredTot[jet][2].SetPoint(predPoint, x2, y2)
                            graphPredTot[jet][3].SetPoint(predPoint, x3, y3)
                            predPoint += 1

                            print("________________________________________")
                            print("New Pred, bin (x,y):", x-half, y-half)
                            if not ON_DATA:
                                tp = target_[j_eff][x][y][trk]
                                print("target(x,y,eta,phi,pt)=", tp[0], tp[1], tp[2], tp[3], tp[4], "Probability target=", target_prob[j_eff][x][y][trk])
                            print("prediction(x,y,eta,phi,pt)=", vp[0], vp[1], vp[2], vp[3], vp[4], "Probability pred=", validation_prob[j_eff][x][y][trk])
                            print(" x0,y0=", x0, y0, " x2,y2=", x2, y2, " x3,y3=", x3, y3)

    output_file = ROOT.TFile("DeepCore_mapValidation_ev{ev}.root".format(ev=jetNum), "recreate")

    if RGB:  # set the color scheme (one palette per layer)
        NCont = 10
        array_of_palette = []
        for red, green, blue in [([1., 1.], [1., 0.], [1., 0.]),
                                 ([1., 0.], [1., 0.], [1., 0.]),
                                 ([1., 0.], [1., 1.], [1., 0.]),
                                 ([1., 0.], [1., 0.], [1., 1.])]:
            FI = TColor.CreateGradientColorTable(2, array2('d', [0., 1.]), array2('d', red), array2('d', green), array2('d', blue), NCont)
            array_of_palette.append(array2('i', [FI+i for i in range(NCont)]))

    def draw_cms_label():
        latexCMS = ROOT.TLatex()
        latexCMS.SetTextSize(0.05)
        latexCMS.DrawLatex(-16.2, 16.2, "#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}")
        latexCMS.DrawLatex(12, 16.2, "#bf{13 TeV}")
        return latexCMS

    # build the canvases
    for jet in range(numPrint):
        for lay in range(layNum):
            canvasTot[jet][lay].cd()
            h = mapTot[jet][lay]
            h.GetXaxis().SetRangeUser(-jetDim, jetDim)
            h.GetYaxis().SetRangeUser(-jetDim, jetDim)
            h.SetTitle("Pixel Window, layer %d" % (lay+1))
            h.GetXaxis().SetTitle("x [pixel]")
            h.GetYaxis().SetTitle("y [pixel]")
            h.GetZaxis().SetTitle("ADC count [/14k]")
            h.GetZaxis().SetTitleOffset(-1.05)
            h.GetXaxis().SetTitleSize(0.06)
            h.GetYaxis().SetTitleSize(0.06)
            h.GetZaxis().SetTitleSize(0.04)
            h.GetXaxis().SetTitleOffset(0.7)
            h.GetYaxis().SetTitleOffset(0.6)

            if RGB:
                gStyle.SetPalette(NCont, array_of_palette[lay])
            h.Draw("colz")
            lat1 = draw_cms_label()

            if jet == outEvent and RGB:
                h.GetZaxis().SetRangeUser(0, 2.7)
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_input_layer%d_event%d.pdf" % (lay, jet))
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_input_layer%d_event%d.png" % (lay, jet))

            if not ON_DATA:
                graphTargetTot[jet][lay].Draw("SAME P")
            graphPredTot[jet][lay].Draw("SAME P")
            for g in (graphTargetTot[jet][lay], graphPredTot[jet][lay]):
                g.SetLineColor(0)
                g.SetFillColor(0)

            legTot = ROOT.TLegend(0.1, 0.9, 0.3, 0.8)
            if not ON_DATA:
                legTot.AddEntry(graphTargetTot[jet][lay], "Target")
            legTot.AddEntry(graphPredTot[jet][lay], "Prediction")
            legTot.SetTextSize(0.03)
            legTot.Draw("SAME")

            canvasTot[jet][lay].Write()

            if jet == outEvent and RGB:
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_crosses_layer%d_event%d.pdf" % (lay, jet))
                canvasTot[jet][lay].SaveAs("RGB_PixelMap_crosses_layer%d_event%d.png" % (lay, jet))

        for trk in range(overlapNum):
            canvasProb[jet][trk].cd()
            h = mapProbPredTot[jet][trk]
            h.GetXaxis().SetRangeUser(-jetDim, jetDim)
            h.GetYaxis().SetRangeUser(-jetDim, jetDim)
            gStyle.SetPalette(57)
            h.Draw("colz")
            h.GetXaxis().SetTitle("x [pixel]")
            h.GetYaxis().SetTitle("y [pixel]")
            h.SetTitle("TCP Prediction Map, overlap %d" % (trk))
            h.GetZaxis().SetTitle("Probability")
            h.GetZaxis().SetTitleOffset(-1.05)
            h.GetXaxis().SetTitleSize(0.06)
            h.GetYaxis().SetTitleSize(0.06)
            h.GetXaxis().SetTitleOffset(0.7)
            h.GetYaxis().SetTitleOffset(0.6)
            if not ON_DATA:
                graphTargetTot[jet][1].Draw("SAME P")
            graphPredTot[jet][1].Draw("SAME P")
            lat2 = draw_cms_label()

            legProb = ROOT.TLegend(0.1, 0.9, 0.3, 0.8)
            if not ON_DATA:
                legProb.AddEntry(graphTargetTot[jet][1], "Target")
            legProb.AddEntry(graphPredTot[jet][1], "Prediction")
            legProb.SetTextSize(0.03)
            legProb.Draw("SAME")

            mapProbPredTot[jet][0].GetZaxis().SetRangeUser(0, 1)
            canvasProb[jet][trk].Write()

        if jet == outEvent and RGB:
            canvasProb[jet][0].SaveAs("Probabiltiy_crosses_event%d.pdf" % (jet))
            canvasProb[jet][0].SaveAs("Probabiltiy_crosses_event%d.png" % (jet))

    output_file.Close()

    #------------------ parameter distributions and residuals (vectorized) ------------------#
    if not ON_DATA:
        print("parameter file: start")
        pdf_par = PdfPages("parameter_file_ev{ev}.pdf".format(ev=jetNum))

        if DRAW_ONLY:
            sel = (target_prob == 1)
        else:
            # DeepCore 2.2 selection: only "associated" TCPs
            l2off = ~(input_[:, :, :, 1] > 0.0001)                               # (nev, x, y)
            l134off = ~np.any(input_[:, :, :, [0, 2, 3]] > 0.0001, axis=3)       # (nev, x, y)
            thr = (prob_thr + np.asarray(dth)[None, None, None, :]
                   + 0.3*l2off[..., None]
                   + np.asarray(dth1)[None, None, None, :]*l134off[..., None])   # (nev, x, y, trk)
            sel = (validation_prob > thr) & (target_prob == 1)

        tgt_sel = target_[sel]                       # (N, parNum+1)
        pred_sel = None if DRAW_ONLY else validation_par[sel]

        for par in range(parNum):
            tg = tgt_sel[:, par]
            pr = None if DRAW_ONLY else pred_sel[:, par]
            if par in (0, 1):
                bins_target = tg*100
                if not DRAW_ONLY:
                    bins = (pr-tg)*100
                    bins_pred = pr*100
            elif par in (2, 3):
                bins_target = tg*0.01
                if not DRAW_ONLY:
                    bins = (pr-tg)*0.01
                    bins_pred = pr*0.01
            else:
                bins_target = tg
                if not DRAW_ONLY:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        bins = (pr-tg)/tg  # relative
                    bins = bins[np.isfinite(bins)]
                    bins_pred = pr

            nbin = len(tg)
            n_sig_ok = 0 if DRAW_ONLY else int(np.sum(pr*tg > 0))
            fracsig = n_sig_ok/float(nbin) if nbin > 0 else 0
            print("Parameter {PAR}, number of correct sign={n}, fraction={f}".format(PAR=par, n=n_sig_ok, f=fracsig))

            mean = sigma = 0.
            if (not DRAW_ONLY) and len(bins) > 0:
                mean = bins.mean()
                sigma = bins.std()

            # residuals
            plt.figure()
            if not DRAW_ONLY:
                if par == 0:
                    plt.hist(bins, 80, facecolor='darkorange', alpha=0.75, range=(-200, 200))
                if par == 1:
                    plt.hist(bins, 120, facecolor='darkorange', alpha=0.75, range=(-300, 300))
                if par in (2, 3):
                    plt.hist(bins, 100, facecolor='darkorange', alpha=0.75, range=(-0.03, 0.02))
                if par == 4:
                    plt.hist(bins, 150, facecolor='darkorange', alpha=0.75, range=(-3, 12))
            if par == 0:
                plt.title(r'Residual Distribution - $\Delta x$', fontsize=22)
                if not DRAW_ONLY:
                    plt.text(80, 35000, r"Mean =%.2f $\mu$m" % (mean), size=14)
                    plt.text(80, 40000, r"$\sigma_{\Delta x}$ = %.1f $\mu$m" % (sigma), size=14)
            if par == 1:
                plt.title(r'Residual Distribution - $\Delta y$', fontsize=22)
                if not DRAW_ONLY:
                    plt.text(100, 40000, r"Mean =%.1f $\mu$m" % (mean), size=14)
                    plt.text(100, 50000, r"$\sigma_{\Delta y}$ = %.1f $\mu$m" % (sigma), size=14)
            if par == 2:
                plt.title(r'Residual Distribution - $\Delta \eta$', fontsize=22)
                if not DRAW_ONLY:
                    plt.text(0.01, 60000, "Mean = 0", size=14)
                    plt.text(0.01, 80000, r"$\sigma_{\Delta \eta}$ = %.3f" % (sigma), size=14)
            if par == 3:
                plt.title(r'Residual Distribution - $\Delta \phi$', fontsize=22)
                if not DRAW_ONLY:
                    plt.text(0.01, 50000, "Mean = 0", size=14)
                    plt.text(0.01, 60000, r"$\sigma_{\Delta \phi}$ = %.3f" % (sigma), size=14)
            if par == 4:
                plt.title(r'Residual Distribution - $p_T$', fontsize=22)
                if not DRAW_ONLY:
                    plt.text(4, 28000, "Mean =%.3f" % (mean), size=14)
                    plt.text(4, 25000, r"$\sigma_{\Delta(p_T)rel}$ = %.3f" % (sigma), size=14)
            plt.ylabel('TCP', fontsize=18)
            if par in (0, 1):
                plt.xlabel(r'Prediction-Target [$\mu$m]', fontsize=22)
            elif par in (2, 3):
                plt.xlabel('Prediction-Target', fontsize=22)
            else:
                plt.xlabel('(Prediction-Target)/Target', fontsize=22)
            plt.grid(True)
            if RGB:
                plt.savefig("residual_{jj}_{PAR}.pdf".format(PAR=par, jj=jetNum), bbox_inches='tight')
                plt.savefig("residual_{jj}_{PAR}.png".format(PAR=par, jj=jetNum), bbox_inches='tight')
            pdf_par.savefig(bbox_inches='tight')
            plt.close()

            # target distribution
            plt.figure()
            if par == 0:
                plt.hist(bins_target, 100, facecolor='royalblue', alpha=0.75, range=(-100, 100))
            if par == 1:
                plt.hist(bins_target, 150, facecolor='royalblue', alpha=0.75, range=(-150, 150))
            if par in (2, 3):
                plt.hist(bins_target, 90, facecolor='royalblue', alpha=0.75, range=(-0.05, 0.04))
            if par == 4:
                plt.hist(bins_target, 100, facecolor='royalblue', alpha=0.75, range=(0, 500))
            titles = [r'$\Delta x$', r'$\Delta y$', r'$\Delta \eta$', r'$\Delta \phi$', r'$p_T$']
            plt.title('Target Distribution - ' + titles[par], fontsize=22)
            if par in (0, 1):
                plt.xlabel(r'Target [$\mu$m]', fontsize=18)
            elif par in (2, 3):
                plt.xlabel('Target', fontsize=18)
            else:
                plt.xlabel('Target [GeV]', fontsize=18)
            plt.ylabel('TCP', fontsize=18)
            plt.grid(True)
            if RGB:
                plt.savefig("target_{jj}_{PAR}.pdf".format(PAR=par, jj=jetNum), bbox_inches='tight')
                plt.savefig("target_{jj}_{PAR}.png".format(PAR=par, jj=jetNum), bbox_inches='tight')
            pdf_par.savefig(bbox_inches='tight')
            plt.close()

            if DRAW_ONLY:
                continue

            # prediction distribution
            plt.figure()
            if par in (0, 1):
                plt.hist(bins_pred, 150, facecolor='red', alpha=0.75, range=(-150, 150))
            if par in (2, 3):
                plt.hist(bins_pred, 70, facecolor='red', alpha=0.75, range=(-0.04, 0.03))
            if par == 4:
                plt.hist(bins_pred, 100, facecolor='red', alpha=0.75, range=(0, 500))
            plt.title('Prediction Distribution - ' + titles[par], fontsize=22)
            if par in (0, 1):
                plt.xlabel(r'Prediction [$\mu$m]', fontsize=18)
            elif par in (2, 3):
                plt.xlabel('Prediction', fontsize=18)
            else:
                plt.xlabel('Prediction [GeV]', fontsize=18)
            plt.ylabel('TCP', fontsize=18)
            plt.grid(True)
            if RGB:
                plt.savefig("prediction_{jj}_{PAR}.pdf".format(PAR=par, jj=jetNum), bbox_inches='tight')
                plt.savefig("prediction_{jj}_{PAR}.png".format(PAR=par, jj=jetNum), bbox_inches='tight')
            pdf_par.savefig(bbox_inches='tight')
            plt.close()

            # scatter plot target vs prediction
            plt.figure()
            if par in (0, 1):
                lim = [-150, 150]
                plt.hist2d(bins_target, bins_pred, bins=300, range=[lim, lim], cmap=plt.cm.viridis)
                plt.plot(lim, lim, color='black')
                v = 'x' if par == 0 else 'y'
                plt.ylabel(r'Predicted $\Delta %s$ [$\mu$m]' % v, fontsize=18)
                plt.xlabel(r'Target $\Delta %s$ [$\mu$m]' % v, fontsize=18)
            elif par in (2, 3):
                lim = [-0.04, 0.04]
                plt.hist2d(bins_target, bins_pred, bins=160, range=[lim, lim], cmap=plt.cm.viridis)
                plt.plot(lim, lim, color='black')
                v = r'\eta' if par == 2 else r'\phi'
                plt.ylabel(r'Predicted $\Delta %s$' % v, fontsize=18)
                plt.xlabel(r'Target $\Delta %s$' % v, fontsize=18)
            else:
                lim = [-500, 500]
                plt.hist2d(bins_target, bins_pred, bins=250, range=[lim, lim])
                plt.plot(lim, lim, color='black')
                plt.plot([0, 0], lim, color='black', linestyle='dashed')
                plt.plot(lim, [0, 0], color='black', linestyle='dashed')
                plt.ylabel('Predicted $p_T$ [GeV]', fontsize=18)
                plt.xlabel('Target $p_T$ [GeV]', fontsize=18)
            plt.colorbar()
            if RGB:
                plt.savefig("predVStarget_{jj}_{PAR}.pdf".format(PAR=par, jj=jetNum), bbox_inches='tight')
                plt.savefig("predVStarget_{jj}_{PAR}.png".format(PAR=par, jj=jetNum), bbox_inches='tight')
            pdf_par.savefig(bbox_inches='tight')
            plt.close()

        pdf_par.close()
        print("parameter file: completed")
