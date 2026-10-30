"""
Replace the seeds of the jetCoreRegionalStep iteration with MC-truth seeds built from PSimHits.

Usage (cmsDriver):
    --customise TruthSeeding/JetCoreSeeds/customiseJetCoreTruthSeeds.customiseJetCoreTruthSeeds

Works with and without the `seedingDeepCore` process modifier:
  * with    seedingDeepCore -> replaces jetCoreRegionalStepSeedsBarrel (DeepCore seeds, |eta| < 1.4);
                               the endcap is replaced too if replaceEndcap=True
  * without seedingDeepCore -> replaces jetCoreRegionalStepSeeds (hit-pair seeds) and switches the
                               trajectory builder to the settings DeepCore uses for hit-less seeds

The input must contain g4SimHits SimTracks/SimVertices/PSimHits (e.g. GEN-SIM, or a DIGI-RAW
written with FEVTDEBUG/FEVTDEBUGHLT event content).
"""
import FWCore.ParameterSet.Config as cms
from TruthSeeding.JetCoreSeeds.simHitJetCoreSeedGenerator_cfi import simHitJetCoreSeedGenerator


def _hitlessSeedBuilder(builder):
    # same settings used in JetCoreRegionalStep_cff.py for the (hit-less) DeepCore seeds
    builder.keepOriginalIfRebuildFails = True
    builder.requireSeedHitsInRebuild = False
    builder.lockHits = False


def customiseJetCoreTruthSeeds(process, replaceEndcap=False, **overrides):
    if hasattr(process, 'jetCoreRegionalStepSeedsBarrel'):          # seedingDeepCore active
        process.jetCoreRegionalStepSeedsBarrel = simHitJetCoreSeedGenerator.clone(
            cores='jetsForCoreTrackingBarrel', jetAbsEtaMax=1.4, **overrides)
        if replaceEndcap and hasattr(process, 'jetCoreRegionalStepSeedsEndcap'):
            process.jetCoreRegionalStepSeedsEndcap = simHitJetCoreSeedGenerator.clone(
                cores='jetsForCoreTrackingEndcap', jetAbsEtaMax=2.5, **overrides)
            _hitlessSeedBuilder(process.jetCoreRegionalStepEndcapTrajectoryBuilder)
            process.jetCoreRegionalStepEndcapTrajectoryFilter.seedPairPenalty = 0
    elif hasattr(process, 'jetCoreRegionalStepSeeds'):               # standard jetCore step
        process.jetCoreRegionalStepSeeds = simHitJetCoreSeedGenerator.clone(
            cores='jetsForCoreTracking', **overrides)
        _hitlessSeedBuilder(process.jetCoreRegionalStepTrajectoryBuilder)
    else:
        raise RuntimeError('customiseJetCoreTruthSeeds: no jetCoreRegionalStep seeds found in the process')
    return process


def customiseJetCoreTruthSeedsBPix2(process):
    """DeepCore-like variant: seeds on BPix layer 2 (fallback to the innermost hit)."""
    return customiseJetCoreTruthSeeds(process, seedLayer='BPix2')
