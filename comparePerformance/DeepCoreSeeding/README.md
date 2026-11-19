# DeepCore seed generator for Phase 2 (ONNX)

`DeepCoreSeedGeneratorPhase2` runs the PyTorch DeepCore model (checkpoint ep088) inside CMSSW through ONNX
Runtime. It is a new module and does not replace `RecoTracker/TkSeedGenerator/DeepCoreSeedGenerator`, the
Phase-1 TensorFlow version, so the two can be compared.

## Setup (CMSSW_20_1_0_pre3)

```bash
cd $CMSSW_BASE/src
cp -r <this repo>/DeepCoreSeeding .
# export the chosen checkpoint (in training/pytorch)
python DeepCore_training_pytorch.py --weights DeepCore_ckpt_ep088.pt --exportONNX DeepCore_phase2_ep088.onnx
python check_onnx.py --weights DeepCore_ckpt_ep088.pt --onnx DeepCore_phase2_ep088.onnx --input VAL.root   # must PASS
mkdir -p DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/data
cp DeepCore_phase2_ep088.onnx DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/data/
scram b -j 8
```

## Run

```bash
cd DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test
cmsRun run_DeepCoreSeedGeneratorPhase2_cfg.py inputFiles=file:/eos/.../Ntuplizer_step1_0_v2.root maxEvents=20 dumpDir=dump
```
The input is the same kind of file the ntuplizer reads: it needs `siPixelClustersPreSplitting`,
`offlinePrimaryVertices` and `ak4CaloJets`. The output, `deepcore_seeds.root`, contains the seeds plus
a `reco::TrackCollection` with one entry per seed (its position on layer 2 and the predicted momentum),
for quick plots.

## Validate before using the seeds in tracking

1. **Images:** run `phase2_DeepCoreNtuplizer.py` on the same input file, then
   `python3 compare_images.py --dump dump --ntuple DeepCoreTrainingSample.root`.
   Most windows should be **identical**. Small differences are expected for one reason only: the
   ntuplizer chooses the module on each layer among modules with SimHits, while the generator, which
   has no simulation information, chooses among modules with clusters.
2. **Network:** `python3 compare_images.py --dump dump --weights DeepCore_ckpt_ep088.pt --pytorchDir <repo>/training/pytorch`
   checks that ONNX Runtime in CMSSW gives the PyTorch outputs on the same images.

## How the module reproduces the training

| | Phase-1 generator (TF) | Phase-2 generator (this one) = `DeepCoreNtuplizerPhase2` |
|---|---|---|
| window | 30×30, 100×150 µm | **120×45, 25×100 µm**; modules with another pitch are skipped |
| inference | TensorFlow `.pb`, one window at a time | **ONNX Runtime**, the windows of a jet in batches (`maxBatch`) |
| directions | merged clusters, without the merged test | merged-cluster test as in `JetCoreClusterSplitter`, with Phase-2 parameters; layer 1, or layer 2 if none, + the jet |
| module per layer | min \|x\| with \|y\| < 3.35 | min (x² + y²) of the crossing, forward-only intersection |
| image charge | adc / 14000 | adc / `chargeNorm` (14000) |
| seed position | pixel edge + offset | **pixel centre** (`localX(i+0.5)`) + offset, in the mirrored frame on flipped modules |
| jets | pT > 100 GeV | pT > 500 GeV, \|η\| < 1.4 (the barrel training sample); configurable |
| output arrays | `double[30][30][3][5]` on the stack | flat ONNX output vectors (no stack overflow at 120×45) |

## Main parameters

- `probThr` (0.85): the probability threshold. `compare_checkpoints.py` showed the knee of the
  efficiency-vs-fake-rate curve at about 0.8, so try 0.8 as well.
- `probThrDeltaOverlap`, `probThrDeltaL134off`, `probThrDeltaL2off`: extra threshold offsets from
  DeepCore 2.2.1. They are 0 here, so the module uses one flat threshold, the same one that
  `compare_checkpoints.py` evaluated. The Phase-1 values are given in the parameter comments.
- `seedErrors`: the diagonal of the seed covariance. These are the Phase-1 values; the x and y errors
  could be tightened to the Phase-2 residual widths (about 20–30 µm).
- `ptMin`, `absEtaMax`, `cores`: the jet selection. Keep it the same as the training (barrel, pT > 500 GeV)
  until the network has been tested outside that region.

## Build tracks and validate (Phase-2 jet-core iteration)

`python/customise_DeepCorePhase2.py` puts the seeds into the Phase-2 `JetCoreRegionalStep`
(procModifier `jetCoreInPhase2`), in the same layout as the Phase-1 `seedingDeepCore` modifier:

| region | seeds | building | output |
|---|---|---|---|
| barrel, \|η\| < 1.4 | `jetCoreRegionalStepSeedsBarrel` = **DeepCoreSeedGeneratorPhase2** | barrel builder, seeding-region rebuilding, DeepCore trajectory cleaner (45 % shared hits) | `jetCoreRegionalStepBarrelTracks` |
| endcap, \|η\| < 2.5 | standard pixel-pair seeds (`jetCoreRegionalStepSeedsEndcap`) | standard jet-core builder | `jetCoreRegionalStepEndcapTracks` |
| merge | | `TrackCollectionMerger` | **`jetCoreRegionalStepTracks`** (same label as before) |

Because the merged collection keeps the label `jetCoreRegionalStepTracks`, `jetCoreRegionalStepSelector`,
`earlyGeneralTracks` → `generalTracks` and the MTV selectors (`cutsRecoTracksJetCoreRegionalStep*`) work unchanged.
Use `customiseBarrelOnly` to drop the endcap part.

The seed generator reads `siPixelClustersPreSplitting`, `firstStepPrimaryVertices` and `jetsForCoreTrackingBarrel`
(ak4CaloJetsForTrk, pT > 100 GeV, \|η\| < 1.4; the module then applies its own pT > 500 GeV cut). These
exist in step3 when the `splitClustersInPhase2Pixel` procModifier is on.

### Simplest alternative: add the DeepCore seeds to the jet-core seeds (`customiseCombined`)

Like adding MC-truth seeds: `jetCoreRegionalStepSeeds` becomes a `SeedCombiner` of the original pixel-pair
seeds (`jetCoreRegionalStepSeedsPixelPair`) and the DeepCore seeds (`jetCoreRegionalStepSeedsDeepCore`).
Candidates, fit, selector, `generalTracks`, DQM and MTV are the standard Phase-2 jet-core step, unchanged.
Use `customiseCombined` in place of `customise`. To get the DeepCore trajectory cleaner (45 % shared hits) as well:
`--customise_commands "process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner='jetCoreRegionalStepDeepCoreTrajectoryCleaner'"`.

### Commands

```bash
cd $CMSSW_BASE/src && scram b -j 8
bash DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test/run_jetcore_validation.sh step2.root 100 8
```
The script runs, on the same step2 file:
1. **step3 baseline**: the standard Phase-2 jet-core step (pixel-pair seeds), with `customiseJetCoreValidation`;
2. **step3 DeepCore**: `customise` + `customiseJetCoreValidation`;
3. **step4** harvesting for both (`customiseHarvesting` adds `Tracking/JetCore/*` to the post-processors);
4. `makeTrackValidationPlots.py` on both DQM files (web pages, `plots_jetcore/`);
5. `compare_mtv_jetcore.py`: efficiency and fake rate vs ΔR(track, jet), pT, η and number of hits,
   recomputed from the MTV numerators, and a summary table (`mtv_jetcore.pdf`, `mtv_track.pdf`).

With your own cmsDriver command, add for the DeepCore configuration
```
--customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customise,DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customiseJetCoreValidation
```
and for the baseline only the second function. To change a parameter:
`--customise_commands "process.jetCoreRegionalStepSeedsBarrel.probThr = 0.8"`.

### What to look at

- `Tracking/JetCore/` (trackValidatorJetCore: chi2 associator, log pT, in-time TPs, which Phase-2 normally
  switches off): `jetCoreRegionalStepTracks`, `cutsRecoTracksJetCoreRegionalStepByOriginalAlgo[Hp]`, `generalTracks`.
  The key plots are **efficiency and fake rate vs ΔR(track, jet)** at small ΔR, and vs pT.
- `Tracking/Track/` (standard MTV, hit associator): the same labels, plus the effect on `generalTracks`
  (the gain in the core must not cost fakes or duplicates elsewhere; check `duplicatesRate_drj`).
- `Tracking/JetCore/TrackSeeding/`: seed-level efficiency of `seedTracksjetCoreRegionalStepSeedsBarrel`.
  DeepCore seeds have no hits, so they are matched with the chi2 associator only and are left out of the
  hit-associator seed MTV in `Tracking/TrackSeeding/`.
- The ΔR plots use `highPtJetsForTrk` (ak4CaloJetsForTrk with pT > 1 TeV), so use a high-pT sample: the
  Phase-2 QCD pT 1800–2400 RelVal (or Z′ → qq / b-enriched high-pT QCD), noPU first, then PU200.
- Also compare the timing of `jetCoreRegionalStepSeedsBarrel` (`--customise_commands "process.Timing=cms.Service('Timing',summaryOnly=cms.untracked.bool(True))"`).

## Where do the lost high-pT tracks go? (LostTrackAnalyzer)

`plugins/LostTrackAnalyzer.cc` writes one TTree entry per simulated track (signal, charged, pT > 100 GeV,
|η| < 2.5, MTV vertex cuts) with its match in every collection on the way to `generalTracks`:
the jet-core seeds (as `seedTracks*`), the jet-core track candidates, the raw tracks of every iteration
(the inputs of `earlyGeneralTracks`; in the two-chain configs also the Barrel/Endcap tracks before the
classifier), `earlyGeneralTracks`, `preDuplicateMergingGeneralTracks` and `generalTracks`.
Matching is by hits (`quickTrackAssociatorByHits`, as the standard MTV efficiency). The DeepCore seeds carry
no hits, so for every collection the nearest object in ΔR is also stored (`_dR`, `_dPtRel`, `_dDz`, `_nNear`).
Per track it also stores the number of its pixel clusters that are shared with another simulated track
(merged clusters that were not split), the ΔR to the nearest other simulated track and to the nearest jet.

Seeds and candidates are not in the step3 output, so the analyzer runs inside step3
(`customiseLostTracks`, apply it last). Interactively, on a few hundred events, for the configs of make_configs.sh:
```
cd <dir with the step3_*.py configs>
bash $CMSSW_BASE/src/DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test/validation/run_lost_tracks.sh \
     /store/.../step2.root 500 deepcore base_rebuild replace nojetcore
```
`test/lost_track_summary.py lostTracks_<name>.root` sorts each lost track into the first stage that failed
(merge, selection, fit, building with a pixel-pair or only a DeepCore seed, seeding) and plots these
fractions vs pT, ΔR to the nearest simulated track and η. `--dc-dr` sets the DeepCore seed match
(default ΔR < 0.02; the script prints the ΔR quantiles for the tracks the jet-core step did build, to choose it).

### DeepCore seed duplicate removal

The network gives many seeds per track (about 15 within ΔR < 0.02 in the noPU QCD sample). `duplicateMaxDist`
(cm, default 0 = off), `duplicateMaxDEta`, `duplicateMaxDPhi` (default 0.002) switch on an event-wide
non-maximum suppression: seeds are taken in decreasing probability, and a seed is dropped if a kept seed is
closer than `duplicateMaxDist` on the layer-2 module and within `duplicateMaxDEta`, `duplicateMaxDPhi` in direction.
Two real tracks with ΔR = 0.001 are only ~70 µm apart on layer 2, so too wide a setting removes the second
track of close pairs: check the efficiency for tracks with a neighbour within ΔR < 0.001 (lost_track_summary.py prints it).
make_configs.sh has `replace_dup1` (0.005 cm, 0.001), `replace_dup2` (0.01 cm, 0.002) and `deepcore_dup2`.

### DeepCore seed options (seedCharge, seedQoverPRelError, attachHit)

New parameters of `DeepCoreSeedGeneratorPhase2` (all defaults keep the previous behaviour):
- `seedCharge`: +1 (default; the network does not predict the charge, Phase-1 also used +1), -1, or 0 = one
  seed with each charge per prediction. The duplicate removal only compares seeds of the same charge.
- `seedQoverPRelError`: if > 0, the q/p uncertainty of each seed is this fraction of its predicted |q/p|
  (instead of the fixed `seedErrors[0]` = 0.15^2). The predicted pT is biased low by ~2x, so use >= 1.
- `attachHit`: attach the nearest cluster of `attachHitClusters` (default `siPixelClusters`, the split clusters
  used by tracking) on the layer-2 module, within `attachMaxDX`/`attachMaxDY` (0.01/0.03 cm) of the prediction,
  and start the seed from its position. `attachHitRequired` drops predictions without a cluster.
  A seed with a hit is subject to the candidate maker's `RedundantSeedCleaner`; the make_configs entries switch
  it off for the DeepCore chain (`jetCoreRegionalStepBarrelTrackCandidates.RedundantSeedCleaner = 'none'`).

make_configs.sh entries: `deepcore_q0`, `deepcore_q0_qerr`, `deepcore_hit`, `deepcore_hit_q0`; the control
`deepcore_noDC` (two-chain layout, no DeepCore seeds: `ptMin = 1e9`); the combinations `deepcore_c45`,
`deepcore_c45_lay7`, `deepcore_noDC_c45_lay7`; and `base_cleaner45[_lay7|_lay9]`.

## Not done yet
- **seedCharge / seedQoverPRelError / attachHit have not been compiled.**
- **The seed duplicate removal has not been compiled.** The selection logic was checked against a brute-force
  version in a standalone test (also across φ = ±π).
- **LostTrackAnalyzer has not been compiled.** The summary script was tested on a synthetic tree only.

- **The module has not been compiled or run in CMSSW_20_1_0_pre3.** The image and the decoding follow
  the ntuplizer line by line, and the Python checks were tested. The C++ was written against the
  CMSSW interfaces of the ntuplizer and the Phase-1 generator.
- **The tracking customisation was tested on a mock process only** (task replacement, labels,
  validators), not in a full cmsRun. The barrel/endcap builder settings are the Phase-1 DeepCore ones and
  have not been tuned for the Phase-2 tracker.
- **Barrel only:** the image stores TBPX layers 1–4, as in the training.
