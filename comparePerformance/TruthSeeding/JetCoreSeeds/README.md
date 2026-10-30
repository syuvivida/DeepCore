# TruthSeeding/JetCoreSeeds — MC-truth seeds for tracking in high-pT jet cores

`SimHitJetCoreSeedGenerator` is a CMSSW `edm::global::EDProducer` that builds "perfect"
`TrajectorySeed`s for charged particles inside the core of high-pT jets, directly from Geant4
`PSimHit`s. It is a drop-in replacement for the jetCore-iteration seeds (DeepCore or hit-pair), so
you can measure the tracking performance in dense environments with an ideal seeder.

Starting points in CMSSW: `RecoTracker/TkSeedGenerator/plugins/DeepCoreSeedGenerator.cc` and
`Validation/RecoTrack/plugins/JetCoreMCtruthSeedGenerator.cc`.

## Algorithm

For every event:

1. Select jets from `cores` with `pt > jetPtMin`, `|eta| < jetAbsEtaMax` (CaloJets, PFJets or GenJets).
2. Collect the PSimHits of every SimTrack, matched by `(EncodedEventId, trackId)`. A hit is kept only
   if `PSimHit::particleType() == SimTrack::type()` — this rejects hits from secondaries that were
   not stored and therefore carry their parent's trackId.
3. SimTrack selection: charged, `pt > simTrackPtMin`, production radius `< maxProductionRadius`,
   signal only (optional), at least `minPixelLayers` distinct pixel layers/disks crossed.
4. Seed hit: the first hit in time (`seedLayer = "innermost"`) or the first hit on a chosen layer,
   e.g. `"BPix2"` to mirror DeepCore (falls back to innermost if `fallbackToInnermost`).
5. Core requirement: `ΔR(jet, track momentum at the seed hit) < deltaR`, and optionally
   `ΔR(jet, seed-hit position seen from the simulated PV) < hitDeltaR`.
6. Seed state, on the module that was actually crossed, at its mid-plane:
   position = `PSimHit::localPosition()`, momentum = `PSimHit::momentumAtEntry()`, true charge.
   Errors are configurable (default: DeepCore NN residual widths). Optional reproducible Gaussian
   smearing (`smear = True`) uses the same sigmas, so the seed covariance stays consistent.
7. One seed per SimTrack (a track inside two jet cones is seeded once).

### Differences from `JetCoreMCtruthSeedGenerator`

| | JetCoreMCtruthSeedGenerator | SimHitJetCoreSeedGenerator |
|---|---|---|
| Seed state | SimTrack vertex momentum, straight-line extrapolation | SimHit local position and momentum (includes curvature, energy loss) |
| Seed module | one module per jet, chosen along the jet axis on BPix2 | the module the particle really crossed |
| Charge | always +1 | true charge |
| Secondaries | hits of un-saved secondaries can be attributed to the parent | rejected by PDG-id match |
| Duplicates | possible across jets | one seed per SimTrack |
| Needs reco clusters/vertices | yes | no |
| Threading | `one::EDProducer` | `global::EDProducer` |

## Outputs

| Product | Description |
|---|---|
| `TrajectorySeedCollection` | hit-less seeds, `alongMomentum` (same convention as DeepCore) |
| `reco::TrackCollection` | one track per seed (seed point and momentum), for quick plots |
| `std::vector<int>`, instance `simTrackIndex` | index in the input SimTrack collection, parallel to the seeds |

## Installation

```bash
cmsrel CMSSW_15_1_0          # or your release
cd CMSSW_15_1_0/src && cmsenv
cp -r /path/to/TruthSeeding .
scram b -j 8
```

`scram` generates `TruthSeeding/JetCoreSeeds/simHitJetCoreSeedGenerator_cfi.py` from `fillDescriptions`.

## Usage

Standalone, with GenJets as cores (any GEN-SIM file):

```bash
cmsRun TruthSeeding/JetCoreSeeds/test/runTruthSeeds_cfg.py inputFiles=file:step1.root maxEvents=50 seedLayer=BPix2
```

Inside reconstruction, replacing the jetCore-iteration seeds:

```bash
cmsDriver.py step3 ... --procModifiers seedingDeepCore \
  --customise TruthSeeding/JetCoreSeeds/customiseJetCoreTruthSeeds.customiseJetCoreTruthSeeds
# DeepCore-like seeding on BPix2:
#   --customise TruthSeeding/JetCoreSeeds/customiseJetCoreTruthSeeds.customiseJetCoreTruthSeedsBPix2
```

The customisation works with or without `seedingDeepCore`. Without it, it replaces
`jetCoreRegionalStepSeeds` and sets the trajectory builder to the DeepCore settings for hit-less seeds
(`keepOriginalIfRebuildFails=True`, `requireSeedHitsInRebuild=False`, `lockHits=False`).
With it, `replaceEndcap=True` also replaces the endcap seeds.

**Input requirement:** step3 must see `g4SimHits` SimTracks, SimVertices and PSimHits. Run step2 with
`--eventcontent FEVTDEBUGHLT` (RAWSIM drops PSimHits), or run digi+reco in one job from GEN-SIM.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `cores` | `jetsForCoreTracking` | jet collection |
| `simTracks`, `simVertices` | `g4SimHits` | |
| `simHits` | pixel barrel/endcap low+high tof | add `TrackerHitsTIBLowTof`, ... to allow strip seeds |
| `jetPtMin`, `jetAbsEtaMax` | 100, 2.5 | |
| `deltaR` | 0.1 | ΔR(jet, track momentum at seed hit) |
| `hitDeltaR` | -1 (off) | ΔR(jet, seed-hit position from sim PV) |
| `simTrackPtMin` | 0.5 GeV | |
| `maxProductionRadius` | 10 cm | -1 disables |
| `signalOnly` | True | |
| `minPixelLayers` | 1 | distinct pixel layers/disks crossed |
| `seedLayer` | `innermost` | or `BPix1..4`, `FPix1..3`, `TIB1`, ... |
| `fallbackToInnermost` | True | |
| `minHitMomentumFraction` | 0 | drop hits after a large momentum loss |
| `errorQoverP` | 0.15 /GeV | absolute σ(q/p) |
| `errorQoverPRelative` | -1 (off) | relative σ(q/p), added in quadrature |
| `errorDxDz`, `errorDyDz` | √5e-6 | |
| `errorX`, `errorY` | √2e-5 cm | |
| `smear`, `randomSeed` | False, 12345 | reproducible per-event Gaussian smearing |

## Limitations

* Pile-up SimHits live in `CrossingFrame`s, not in `g4SimHits`; with the default inputs only the
  signal event is seeded (which is what is normally wanted for jet-core studies).
* Seeds carry no RecHits. That matches DeepCore and works with the jetCore builder settings above;
  builders that require seed hits need a SimHit→RecHit association, which is not done here.
* The SimTrack→TrackingParticle link is not stored; use `simTrackIndex` plus the SimTrack
  collection, or associate the CKF tracks with the standard track associators.

## JetCoreMCtruthSeedGeneratorPhase2

Phase-2 version of `Validation/RecoTrack/plugins/JetCoreMCtruthSeedGenerator.cc` (same idea: seeds for the
SimTracks in a deltaR cone around each jet core, on one target layer, DeepCore-like errors). Pixel pitch and
module bounds come from the geometry; forward jets use an endcap disk; true charge; one seed per SimTrack;
Phase-2 seed-id packing. Default mode places each seed at the SimTrack's own PSimHit on the target layer
(`seedOnTrueModule = True`); `False` reproduces the Phase-1 straight-line extrapolation onto the jet's module.

Replace the Phase-2 jet-core seeds:

```python
from TruthSeeding.JetCoreSeeds.JetCoreMCtruthSeedGeneratorPhase2_cfi import JetCoreMCtruthSeedGeneratorPhase2 as _p2
process.jetCoreRegionalStepSeeds = _p2.clone(cores='jetsForCoreTracking', ptMin=100)
process.jetCoreRegionalStepTrajectoryBuilder.keepOriginalIfRebuildFails = True
process.jetCoreRegionalStepTrajectoryBuilder.requireSeedHitsInRebuild = False
process.jetCoreRegionalStepTrajectoryBuilder.lockHits = False
```
