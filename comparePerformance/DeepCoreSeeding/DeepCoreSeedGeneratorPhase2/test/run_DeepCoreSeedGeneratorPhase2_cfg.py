# Run the Phase-2 DeepCore seed generator on RECO-level events (same input as the ntuplizer step 2).
#
#   cmsRun run_DeepCoreSeedGeneratorPhase2_cfg.py inputFiles=file:Ntuplizer_step1_0_v2.root maxEvents=20 \
#          [probThr=0.85] [dumpDir=dump] [threads=4]
#
# Output: deepcore_seeds.root with the seeds (TrajectorySeedCollection) and the seed "tracks"
# (reco::TrackCollection, position on layer 2 + predicted direction/pt) for quick checks.
# With dumpDir set, every window (image + network output) is written there for compare_images.py.
import os
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

opt = VarParsing("analysis")
opt.register("probThr", 0.85, VarParsing.multiplicity.singleton, VarParsing.varType.float, "probability threshold")
opt.register("dumpDir", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "dump windows here")
opt.register("weightFile", "DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/data/DeepCore_phase2_ep088.onnx",
             VarParsing.multiplicity.singleton, VarParsing.varType.string, "ONNX model (FileInPath)")
opt.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "threads")
opt.setDefault("maxEvents", 10)
opt.setDefault("outputFile", "deepcore_seeds.root")
opt.parseArguments()

# same era / geometry / global tag as Ntuplizer/DeepCoreTraining/test/phase2_DeepCoreNtuplizer.py
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
process = cms.Process("DEEPCORE", Phase2C26I13M9)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryExtendedRun4D128Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T35", "")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opt.maxEvents))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(opt.inputFiles))
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(opt.threads),
                                     numberOfStreams=cms.untracked.uint32(0),
                                     wantSummary=cms.untracked.bool(True))

from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.deepCoreSeedGeneratorPhase2_cfi import deepCoreSeedGeneratorPhase2
process.deepCoreSeeds = deepCoreSeedGeneratorPhase2.clone(
    vertices="offlinePrimaryVertices",
    pixelClusters="siPixelClustersPreSplitting",
    cores="ak4CaloJets",
    probThr=opt.probThr,
    weightFile=opt.weightFile,
    dumpDir=opt.dumpDir,
)
if opt.dumpDir:
    os.makedirs(opt.dumpDir, exist_ok=True)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName=cms.untracked.string(opt.outputFile),
                               outputCommands=cms.untracked.vstring("drop *", "keep *_deepCoreSeeds_*_*"))
process.p = cms.Path(process.deepCoreSeeds)
process.e = cms.EndPath(process.out)
