"""
Customise functions to run the Phase-2 DeepCore seed generator inside the
Phase-2 JetCoreRegionalStep (procModifier jetCoreInPhase2) and to validate the
resulting tracks with the MultiTrackValidator.

Reconstruction (DeepCore configuration only):
    --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customise

Validation (use for BOTH the baseline and the DeepCore configuration, so the
two DQM files contain the same folders):
    --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customiseJetCoreValidation

Harvesting (step4):
    --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customiseHarvesting

What `customise` does (mirrors the Phase-1 `seedingDeepCore` procModifier):

  barrel  |eta|<1.4 : DeepCoreSeedGeneratorPhase2 (ONNX, ep088) seeds
                      -> barrel trajectory builder (seeding-region rebuilding,
                         DeepCore trajectory cleaner)  -> jetCoreRegionalStepBarrelTracks
  endcap  |eta|<2.5 : the standard pixel-pair seeds of the jet-core step
                      -> jetCoreRegionalStepEndcapTracks
  merge             : TrackCollectionMerger -> jetCoreRegionalStepTracks
                      (same label as before, so jetCoreRegionalStepSelector,
                       earlyGeneralTracks and the validation are unchanged)

Set `barrelOnly=True` (see customiseBarrelOnly) to drop the endcap pixel-pair
part and keep only the DeepCore tracks in the jet-core iteration.
"""
import FWCore.ParameterSet.Config as cms

_JC = "RecoTracker.IterativeTracking.JetCoreRegionalStep_cff"


def _get(process, name):
    """Return process.<name>; if it is not in the process yet, take it from the
    jet-core cff (already modified by the process eras/procModifiers)."""
    if hasattr(process, name):
        return getattr(process, name)
    import importlib
    mod = importlib.import_module(_JC)
    obj = getattr(mod, name).clone() if hasattr(getattr(mod, name), "clone") else getattr(mod, name)
    setattr(process, name, obj)
    return getattr(process, name)


def _deepCoreSeeds(process, **kw):
    from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.deepCoreSeedGeneratorPhase2_cfi import deepCoreSeedGeneratorPhase2
    pars = dict(
        vertices="firstStepPrimaryVertices",
        pixelClusters="siPixelClustersPreSplitting",
        cores="jetsForCoreTrackingBarrel",
    )
    pars.update(kw)
    return deepCoreSeedGeneratorPhase2.clone(**pars)


def customise(process, barrelOnly=False, **seedPars):
    if not hasattr(process, "JetCoreRegionalStepTask"):
        raise RuntimeError("JetCoreRegionalStepTask not found: run step3 with --procModifiers jetCoreInPhase2")

    # make sure everything we need from the jet-core cff is in the process
    for n in ["jetsForCoreTrackingBarrel", "jetsForCoreTrackingEndcap", "firstStepGoodPrimaryVertices",
              "jetCoreRegionalStepSeedLayers",
              "jetCoreRegionalStepEndcapTrackingRegions", "jetCoreRegionalStepEndcapHitDoublets",
              "jetCoreRegionalStepSeedsEndcap",
              "jetCoreRegionalStepBarrelTrajectoryFilter", "jetCoreRegionalStepEndcapTrajectoryFilter",
              "jetCoreRegionalStepBarrelTrajectoryBuilder", "jetCoreRegionalStepEndcapTrajectoryBuilder",
              "jetCoreRegionalStepChi2Est", "jetCoreRegionalStepDeepCoreTrajectoryCleaner",
              "jetCoreRegionalStepBarrelTrackCandidates", "jetCoreRegionalStepEndcapTrackCandidates",
              "jetCoreRegionalStepBarrelTracks", "jetCoreRegionalStepEndcapTracks",
              "jetCoreRegionalStepBarrel", "jetCoreRegionalStepEndcap"]:
        _get(process, n)

    # 1) DeepCore Phase-2 seeds replace the Phase-1 (TensorFlow) barrel seeder
    process.jetCoreRegionalStepSeedsBarrel = _deepCoreSeeds(process, **seedPars)

    # Phase-2 jet-core track fits keep the trajectories in the event
    process.jetCoreRegionalStepBarrelTracks.TrajectoryInEvent = True
    process.jetCoreRegionalStepEndcapTracks.TrajectoryInEvent = True

    # 2) merge barrel (DeepCore) and endcap (pixel pairs) into jetCoreRegionalStepTracks
    from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import TrackCollectionMerger
    if barrelOnly:
        producers, classifiers = ["jetCoreRegionalStepBarrelTracks"], ["jetCoreRegionalStepBarrel"]
    else:
        producers = ["jetCoreRegionalStepBarrelTracks", "jetCoreRegionalStepEndcapTracks"]
        classifiers = ["jetCoreRegionalStepBarrel", "jetCoreRegionalStepEndcap"]
    process.jetCoreRegionalStepTracks = TrackCollectionMerger.clone(
        trackProducers=producers,
        inputClassifiers=classifiers,
        foundHitBonus=100.0,
        lostHitPenalty=1.0,
        # Phase 2 (timing layer) keeps trajectories down to generalTracks: earlyGeneralTracks copies
        # them from each input, and the DuplicateListMerger (generalTracks) assumes one trajectory per
        # track. Without this, the jet-core tracks have no trajectories, the two lists get out of step
        # and generalTracks aborts in TrackCollectionCloner.cc:49.
        copyExtras=cms.untracked.bool(True),
        copyTrajectories=cms.untracked.bool(True),
    )

    # 3) new task; jetCoreRegionalStepSelector (Phase-2 multiTrackSelector) still
    #    reads jetCoreRegionalStepTracks and feeds earlyGeneralTracks
    barrel = cms.Task(process.jetsForCoreTrackingBarrel,
                      process.firstStepGoodPrimaryVertices,
                      process.jetCoreRegionalStepSeedsBarrel,
                      process.jetCoreRegionalStepBarrelTrackCandidates,
                      process.jetCoreRegionalStepBarrelTracks,
                      process.jetCoreRegionalStepBarrel)
    endcap = cms.Task(process.jetsForCoreTrackingEndcap,
                      process.jetCoreRegionalStepSeedLayers,
                      process.jetCoreRegionalStepEndcapTrackingRegions,
                      process.jetCoreRegionalStepEndcapHitDoublets,
                      process.jetCoreRegionalStepSeedsEndcap,
                      process.jetCoreRegionalStepEndcapTrackCandidates,
                      process.jetCoreRegionalStepEndcapTracks,
                      process.jetCoreRegionalStepEndcap)
    tasks = [barrel] if barrelOnly else [barrel, endcap]
    # change the existing Task object in place, so iterTrackingTask etc. see it
    t = process.JetCoreRegionalStepTask
    for item in list(t._collection):
        t.remove(item)
    t.add(*tasks)
    t.add(process.jetCoreRegionalStepTracks, process.jetCoreRegionalStepSelector)

    # 4) DQM seed monitor of the jet-core step: its seeds/candidates labels no longer exist
    _fixSeedDQM(process, barrelOnly)

    # 5) the seed MTV: swap the (now unused) pixel-pair seed label for the new ones
    _fixSeedValidation(process, ["jetCoreRegionalStepSeedsBarrel"] + ([] if barrelOnly else ["jetCoreRegionalStepSeedsEndcap"]))

    return process


def _fixSeedDQM(process, barrelOnly):
    """TrackSeedMonjetCoreRegionalStep reads jetCoreRegionalStepSeeds/TrackCandidates, which are gone.
    Point it to the endcap (pixel-pair) seeds; with barrelOnly remove it (as seedingDeepCore does in Phase 1)."""
    name = "TrackSeedMonjetCoreRegionalStep"
    if not hasattr(process, name):
        return
    mon = getattr(process, name)
    if not barrelOnly:
        mon.SeedProducer = "jetCoreRegionalStepSeedsEndcap"
        mon.TCProducer = "jetCoreRegionalStepEndcapTrackCandidates"
        if hasattr(mon, "RegionProducer"):
            mon.RegionProducer = "jetCoreRegionalStepEndcapTrackingRegions"
        if hasattr(mon, "RegionCandidates"):
            mon.RegionCandidates = "jetsForCoreTrackingEndcap"
        return
    for seqs in (process.sequences, process.paths, process.endpaths):
        for s in seqs.values():
            s.remove(mon)


def customiseCheckTrajectories(process):
    """Diagnostic only: check trajectory <-> track consistency of every collection on the way to
    generalTracks, and keep the job alive so that the check runs (do not use for physics)."""
    process.ttCheck = cms.EDAnalyzer("TrajTrackConsistencyChecker",
        collections=cms.VInputTag(
            "jetCoreRegionalStepBarrelTracks", "jetCoreRegionalStepEndcapTracks", "jetCoreRegionalStepTracks",
            "initialStepTracks", "highPtTripletStepTracks", "earlyGeneralTracks",
            "muonSeededTracksInOut", "muonSeededTracksOutIn",
            "preDuplicateMergingGeneralTracks", "mergedDuplicateTracks"),
        maxPrint=cms.uint32(10))
    process.ttCheckPath = cms.EndPath(process.ttCheck)
    if process.schedule is not None:
        process.schedule.append(process.ttCheckPath)
    # avoid the assert in generalTracks so the event reaches the check
    process.generalTracks.copyTrajectories = cms.untracked.bool(False)
    process.options.TryToContinue = cms.untracked.vstring("ProductNotFound")
    return process


def customiseCombined(process, deepCoreCleaner=False, **seedPars):
    """Simplest setup (like adding MC-truth seeds): DeepCore seeds are ADDED to the standard
    jet-core pixel-pair seeds and everything downstream is the standard Phase-2 jet-core step.

      jetCoreRegionalStepSeedsPixelPair : the original pixel-pair seeds (renamed)
      jetCoreRegionalStepSeedsDeepCore  : DeepCoreSeedGeneratorPhase2 (barrel, pT>500 GeV jets)
      jetCoreRegionalStepSeeds          : SeedCombiner of the two (same label as before, so the
                                          candidates, tracks, selector, DQM and MTV are untouched)

    The candidate maker already handles hitless seeds (doSeedingRegionRebuilding = True by default).
    deepCoreCleaner=True uses the DeepCore trajectory cleaner (45 % shared hits) instead of the default."""
    if not hasattr(process, "jetCoreRegionalStepSeeds"):
        raise RuntimeError("jetCoreRegionalStepSeeds not found: run step3 with --procModifiers jetCoreInPhase2")
    for n in ["jetsForCoreTrackingBarrel", "firstStepGoodPrimaryVertices"]:
        _get(process, n)
    process.jetCoreRegionalStepSeedsPixelPair = process.jetCoreRegionalStepSeeds.clone()
    process.jetCoreRegionalStepSeedsDeepCore = _deepCoreSeeds(process, **seedPars)
    process.jetCoreRegionalStepSeeds = cms.EDProducer("SeedCombiner",
        seedCollections=cms.VInputTag("jetCoreRegionalStepSeedsPixelPair", "jetCoreRegionalStepSeedsDeepCore"))
    process.JetCoreRegionalStepTask.add(process.jetsForCoreTrackingBarrel,
                                        process.jetCoreRegionalStepSeedsPixelPair,
                                        process.jetCoreRegionalStepSeedsDeepCore)
    if deepCoreCleaner:
        _get(process, "jetCoreRegionalStepDeepCoreTrajectoryCleaner")
        process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner = "jetCoreRegionalStepDeepCoreTrajectoryCleaner"
    return process


def customiseReplaceBarrel(process, **seedPars):
    """DeepCore REPLACES the pixel-pair seeds in barrel jets: same layout as customise(), but the
    pixel-pair ("endcap") part only seeds jets with 1.4 < |eta| < 2.5. Note that Phase-1
    seedingDeepCore does NOT do this: its endcap jets are |eta| < 2.5, so pixel pairs also seed
    barrel jets there (as in customise())."""
    customise(process, **seedPars)
    process.jetsForCoreTrackingEndcap.cut = "pt > 100 && abs(eta) > 1.4 && abs(eta) < 2.5"
    return process


def customiseBarrelOnly(process):
    return customise(process, barrelOnly=True)


# ---------------------------------------------------------------- validation
def _seedTracks(process, seedLabel):
    """TrackFromSeedProducer for a seed collection (same as TrackValidation_cff)."""
    name = "seedTracks" + seedLabel
    if not hasattr(process, name):
        from Validation.RecoTrack.trajectorySeedTracks_cfi import trajectorySeedTracks as _trajectorySeedTracks
        setattr(process, name, _trajectorySeedTracks.clone(src=seedLabel))
    return name


def _fixSeedValidation(process, seedLabels):
    """Seed-level MTV (hit associator) for the generic seeding validator and a
    chi2-associator seeding validator dedicated to the jet-core seeds."""
    if not (hasattr(process, "trackValidatorSeedingTrackingOnly") and hasattr(process, "trackValidatorsTrackingOnly")):
        return
    names = [_seedTracks(process, s) for s in seedLabels]
    t = cms.Task(*[getattr(process, n) for n in names])
    process.jetCoreSeedTracksTask = t
    process.trackValidatorsTrackingOnly.associate(t)

    v = process.trackValidatorSeedingTrackingOnly
    old = "seedTracksjetCoreRegionalStepSeeds"
    # DeepCore seeds carry no hits -> the hit associator cannot match them;
    # they are validated with the chi2 associator (JetCore seeding MTV) only
    withHits = [n for n in names if not n.endswith("SeedsBarrel")]
    v.label = [str(x) for x in v.label if str(x) != old] + withHits
    v.doResolutionPlotsForLabels = withHits
    if hasattr(process, "trackValidatorJetCoreSeedingTrackingOnly"):
        process.trackValidatorJetCoreSeedingTrackingOnly.label = names
        process.trackValidatorJetCoreSeedingTrackingOnly.doResolutionPlotsForLabels = names


def _hasJetCoreStep(process):
    """True if the jet-core step actually feeds generalTracks. Its modules are always DEFINED
    (the cff is loaded in every Phase-2 config), but they only run with --procModifiers jetCoreInPhase2,
    which adds jetCoreRegionalStepTracks to earlyGeneralTracks."""
    egt = getattr(process, "earlyGeneralTracks", None)
    if egt is None:
        return False
    for par in ("TrackProducers", "trackProducers"):
        if hasattr(egt, par):
            return any(str(t).split(":")[0] == "jetCoreRegionalStepTracks" for t in getattr(egt, par))
    return False


def customiseJetCoreValidation(process):
    """Add the jet-core MTVs (Tracking/JetCore, chi2 associator, log-pT) that
    Phase-2 removes by default. Apply to baseline and DeepCore configs."""
    for n in ["trackValidatorsTrackingOnly", "trackValidatorJetCore", "trackValidatorJetCoreSeedingTrackingOnly"]:
        if not hasattr(process, n):
            print("customiseJetCoreValidation: %s not in process, nothing done" % n)
            return process
    seq = process.trackValidatorsTrackingOnly
    if not _hasJetCoreStep(process):
        # no jet-core step (reference without jetCoreInPhase2): same folder, generalTracks only
        process.trackValidatorJetCore.label = ["generalTracks"]
        process.trackValidatorJetCore.doResolutionPlotsForLabels = ["generalTracks"]
        seq += process.trackValidatorJetCore
        return process
    # seeds: DeepCore config has Barrel/Endcap seeds, baseline the pixel-pair seeds
    if hasattr(process, "jetCoreRegionalStepSeedsBarrel") and \
       process.jetCoreRegionalStepSeedsBarrel.type_() == "DeepCoreSeedGeneratorPhase2":
        seeds = ["jetCoreRegionalStepSeedsBarrel"]
        if "jetCoreRegionalStepEndcapTracks" in process.jetCoreRegionalStepTracks.trackProducers.value():
            seeds.append("jetCoreRegionalStepSeedsEndcap")
    elif hasattr(process, "jetCoreRegionalStepSeedsDeepCore"):      # customiseCombined
        seeds = ["jetCoreRegionalStepSeeds", "jetCoreRegionalStepSeedsPixelPair", "jetCoreRegionalStepSeedsDeepCore"]
    else:
        seeds = ["jetCoreRegionalStepSeeds"]
    names = [_seedTracks(process, s) for s in seeds]
    process.jetCoreSeedTracksForValTask = cms.Task(*[getattr(process, n) for n in names])
    seq.associate(process.jetCoreSeedTracksForValTask)
    process.trackValidatorJetCoreSeedingTrackingOnly.label = names
    process.trackValidatorJetCoreSeedingTrackingOnly.doResolutionPlotsForLabels = names
    seq += process.trackValidatorJetCore
    seq += process.trackValidatorJetCoreSeedingTrackingOnly
    return process


def customiseHarvesting(process):
    """Harvest the Tracking/JetCore folders (efficiency, fake rate, ...)."""
    for n in ["postProcessorTrack", "postProcessorTrackSummary", "postProcessorTrack2D"]:
        if hasattr(process, n):
            m = getattr(process, n)
            sub = m.subDirs.value() if hasattr(m.subDirs, "value") else list(m.subDirs)
            if "Tracking/JetCore/*" not in sub:
                m.subDirs.append("Tracking/JetCore/*")
    return process


# ---------------------------------------------------------------- track-by-track study
def _jetCoreSeedLabels(process):
    """Seed collections of the jet-core step in this configuration (same logic as customiseJetCoreValidation)."""
    if hasattr(process, "jetCoreRegionalStepSeedsBarrel") and \
       process.jetCoreRegionalStepSeedsBarrel.type_() == "DeepCoreSeedGeneratorPhase2":
        seeds = ["jetCoreRegionalStepSeedsBarrel"]
        if "jetCoreRegionalStepEndcapTracks" in [str(t) for t in process.jetCoreRegionalStepTracks.trackProducers]:
            seeds.append("jetCoreRegionalStepSeedsEndcap")
        return seeds
    if hasattr(process, "jetCoreRegionalStepSeedsDeepCore"):            # customiseCombined
        return ["jetCoreRegionalStepSeedsPixelPair", "jetCoreRegionalStepSeedsDeepCore"]
    return ["jetCoreRegionalStepSeeds"]


def customiseLostTracks(process, ptMin=100., absEtaMax=2.5, fileName="lostTracks.root"):
    """Add LostTrackAnalyzer: one TTree entry per simulated track above ptMin, with its hit-based match
    in every iteration, the jet-core seeds/candidates/tracks, earlyGeneralTracks and generalTracks,
    and a geometric match to the (hitless) DeepCore seeds. Run it inside step3 (seeds and candidates
    are not stored in the output). Apply after the DeepCore customise functions."""
    tasks = cms.Task()
    # truth helpers (normally already there with the VALIDATION step)
    if not hasattr(process, "tpClusterProducer"):
        process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
    if not hasattr(process, "quickTrackAssociatorByHits"):
        process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
    if not hasattr(process, "trackingParticleNumberOfLayersProducer"):
        process.load("SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff")
    tasks.add(process.tpClusterProducer, process.quickTrackAssociatorByHits,
              process.trackingParticleNumberOfLayersProducer)

    tracks, candidates = [], []
    egt = getattr(process, "earlyGeneralTracks", None)
    if egt is not None:
        for par in ("trackProducers", "TrackProducers"):
            if hasattr(egt, par):
                tracks += [str(t).split(":")[0] for t in getattr(egt, par)]
                break
    if _hasJetCoreStep(process):
        jc = process.jetCoreRegionalStepTracks
        if hasattr(jc, "trackProducers"):                           # two chains, merged by TrackCollectionMerger
            for t in jc.trackProducers:
                t = str(t).split(":")[0]
                tracks.append(t)
                cand = getattr(process, t).src.value() if hasattr(getattr(process, t), "src") else None
                if cand:
                    candidates.append(cand)
        elif hasattr(jc, "src"):
            candidates.append(jc.src.value())
        for s in _jetCoreSeedLabels(process):
            n = _seedTracks(process, s)
            tasks.add(getattr(process, n))
            tracks.append(n)
    for t in ("earlyGeneralTracks", "preDuplicateMergingGeneralTracks", "generalTracks"):
        if hasattr(process, t):
            tracks.append(t)
    seen = set()
    tracks = [t for t in tracks if not (t in seen or seen.add(t))]
    candidates = [c for c in dict.fromkeys(candidates)]
    print("customiseLostTracks: tracks     =", tracks)
    print("customiseLostTracks: candidates =", candidates)

    process.lostTrackAnalyzer = cms.EDAnalyzer("LostTrackAnalyzer",
        trackingParticles=cms.InputTag("mix", "MergedTrackTruth"),
        associator=cms.InputTag("quickTrackAssociatorByHits"),
        beamSpot=cms.InputTag("offlineBeamSpot"),
        jets=cms.InputTag("ak4CaloJetsForTrk"),
        clusterTP=cms.InputTag("tpClusterProducer"),
        tpNLayers=cms.InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"),
        tpNPixelLayers=cms.InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"),
        tracks=cms.VInputTag(*tracks),
        candidates=cms.VInputTag(*candidates),
        ptMin=cms.double(ptMin), absEtaMax=cms.double(absEtaMax),
        tipMax=cms.double(3.5), lipMax=cms.double(30.),       # MTV TrackingParticle selection
        jetPtMin=cms.double(100.), nearDR=cms.double(0.02), otherTPPtMin=cms.double(1.),
        signalOnly=cms.bool(True), chargedOnly=cms.bool(True))
    process.lostTrackTask = tasks
    process.lostTrackPath = cms.EndPath(process.lostTrackAnalyzer, process.lostTrackTask)
    if process.schedule is not None:
        process.schedule.append(process.lostTrackPath)
    if not hasattr(process, "TFileService"):
        process.TFileService = cms.Service("TFileService", fileName=cms.string(fileName))
    return process
