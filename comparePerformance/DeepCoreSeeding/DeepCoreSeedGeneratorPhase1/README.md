# DeepCore Phase 1 (Run 3) with a PyTorch model — CMSSW_14_2_2

`plugins/DeepCoreSeedGeneratorONNX.cc` is the CMSSW_14_2_2 `DeepCoreSeedGenerator` with only the TensorFlow
inference replaced by ONNX Runtime. Same windows, image (ADC/14000), thresholds (probThr 0.7 + DeepCore 2.2.1
offsets), seed parameters and errors, and the same module parameters.

## Setup
```
cmsrel CMSSW_14_2_2 && cd CMSSW_14_2_2/src && cmsenv
mkdir -p DeepCoreSeeding && cp -r <unzipped>/DeepCoreSeeding/DeepCoreSeedGeneratorPhase1 DeepCoreSeeding/
#   (only this folder: the Phase-2 package does not compile in 14_2_2)
python DeepCore_training_pytorch.py --weights <phase-1 checkpoint ep052>.pt --exportONNX DeepCore_phase1.onnx
python3 DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/test/check_onnx_phase1.py DeepCore_phase1.onnx   # PASS
cp DeepCore_phase1.onnx DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/data/
scram b -j 8
```

## Three step3 configurations
```
mkdir ~/val_run3 && cd ~/val_run3
bash $CMSSW_BASE/src/DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/test/validation/make_configs_phase1.sh 8
```
| config | what |
|---|---|
| `step3_jetcore.py` | default Run-3 jet-core step (pixel-pair seeds) |
| `step3_deepcore_tf.py` | `--procModifiers seedingDeepCore`: DeepCore as in CMSSW (TensorFlow, `DeepCoreSeedGenerator_TrainedModel_barrel_2023.pb`) |
| `step3_deepcore_onnx.py` | the same, with `jetCoreRegionalStepSeedsBarrel` replaced by `DeepCoreSeedGeneratorONNX` and your ONNX model |

All three: `RAW2DIGI,L1Reco,RECO,RECOSIM,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM`, DQM output,
era `Run3_2024`, `auto:phase1_2024_realistic` (set `ERA`/`COND` to match your step2 sample, e.g. `Run3_2025`,
`auto:phase1_2025_realistic`), and the `Tracking/JetCore` folder (CMSSW adds it only with seedingDeepCore;
`customisePhase1JetCoreValidation` adds it to the default one too). `ONNX=...` selects another model file.

Input: a Run-3 step2 (GEN-SIM-DIGI-RAW) sample with high-pT jets, e.g.
`dasgoclient -query="dataset=/RelValQCD_Pt_1800_2400_14/CMSSW_14_*/GEN-SIM-DIGI-RAW"` (era/conditions must match).

## Run, harvest, plot
```
bash $V/submit_validation.sh run3_jetcore       step3_jetcore.py       files.txt /eos/user/s/syu/deepcore_run3
bash $V/submit_validation.sh run3_deepcore_tf   step3_deepcore_tf.py   files.txt /eos/user/s/syu/deepcore_run3
bash $V/submit_validation.sh run3_deepcore_onnx step3_deepcore_onnx.py files.txt /eos/user/s/syu/deepcore_run3
bash $V/harvest_all.sh run3 /eos/user/s/syu/deepcore_run3          # -> DQM_run3_<tag>.root (JetCore folder harvested)
makeTrackValidationPlots.py DQM_run3_jetcore.root DQM_run3_deepcore_tf.root DQM_run3_deepcore_onnx.root -o plots_run3
```
(`V=$CMSSW_BASE/src/DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/test/validation`.) For a quick interactive check
run `cmsRun step3_deepcore_onnx.py` with `process.source.fileNames`/`maxEvents` set.

Sanity check first: an ONNX export of a model equivalent to the CMSSW TensorFlow one should reproduce
`deepcore_tf` (same seeds and efficiencies); then compare your retrained model.

Not compiled or run yet.
