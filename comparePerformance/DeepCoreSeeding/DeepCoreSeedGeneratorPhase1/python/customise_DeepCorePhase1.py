"""Phase-1 DeepCore with a PyTorch model exported to ONNX.

Use with the seedingDeepCore procModifier, e.g. in cmsDriver:
  --procModifiers seedingDeepCore \\
  --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/customise_DeepCorePhase1.customiseDeepCoreONNX
The weight file is DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/data/DeepCore_phase1.onnx unless changed, e.g.
  --customise_commands "process.jetCoreRegionalStepSeedsBarrel.weightFile = 'DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/data/DeepCore_phase1_ep52.onnx'"
"""
import FWCore.ParameterSet.Config as cms

DEFAULT_WEIGHTS = "DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/data/DeepCore_phase1.onnx"


def customiseDeepCoreONNX(process, weightFile=DEFAULT_WEIGHTS):
    """Replace the TensorFlow DeepCoreSeedGenerator by DeepCoreSeedGeneratorONNX with the same parameters
    (jets, thresholds, ...) and the ONNX weight file. The module keeps its name, so all tasks and the
    track candidate maker are unchanged."""
    if not hasattr(process, "jetCoreRegionalStepSeedsBarrel") or \
       process.jetCoreRegionalStepSeedsBarrel.type_() != "DeepCoreSeedGenerator":
        raise RuntimeError("customiseDeepCoreONNX: no DeepCoreSeedGenerator 'jetCoreRegionalStepSeedsBarrel' "
                           "in the process (run with --procModifiers seedingDeepCore)")
    old = process.jetCoreRegionalStepSeedsBarrel
    pars = old.parameters_()
    pars["weightFile"] = cms.FileInPath(weightFile)
    process.jetCoreRegionalStepSeedsBarrel = cms.EDProducer("DeepCoreSeedGeneratorONNX", **pars)
    return process


# ---------------------------------------------------------------- validation (Run 3 / Phase 1)
def _scheduled(process, label):
    return any(label in p.moduleNames() for p in list(process.paths_().values()) + list(process.endpaths_().values()))


def customisePhase1JetCoreValidation(process):
    """Make sure the Tracking/JetCore MultiTrackValidator runs (CMSSW adds it only with seedingDeepCore), so the
    default jet-core configuration has the same folder as the DeepCore ones. Does nothing if it already runs."""
    if not hasattr(process, "trackValidatorJetCore") or _scheduled(process, "trackValidatorJetCore"):
        return process
    task = cms.Task()
    for m in ("trackAssociatorByChi2", "trackingParticlesInTime",
              "cutsRecoTracksJetCoreRegionalStep", "cutsRecoTracksJetCoreRegionalStepHp",
              "cutsRecoTracksJetCoreRegionalStepByOriginalAlgo", "cutsRecoTracksJetCoreRegionalStepByOriginalAlgoHp"):
        if hasattr(process, m):
            task.add(getattr(process, m))
    process.jetCoreValidationTask = task
    process.jetCoreValidationPath = cms.EndPath(process.trackValidatorJetCore, process.jetCoreValidationTask)
    if process.schedule is not None:
        outs = [i for i, p in enumerate(process.schedule) if "output" in p.label_().lower()]
        process.schedule.insert(outs[0] if outs else len(process.schedule), process.jetCoreValidationPath)
    return process


def customisePhase1JetCoreHarvesting(process):
    """Harvesting: compute efficiencies/fake rates also in Tracking/JetCore (CMSSW does it only with seedingDeepCore)."""
    for name in ("postProcessorTrack", "postProcessorTrackSummary", "postProcessorTrack2D"):
        if hasattr(process, name):
            m = getattr(process, name)
            if "Tracking/JetCore/*" not in list(m.subDirs):
                m.subDirs.append("Tracking/JetCore/*")
    return process
