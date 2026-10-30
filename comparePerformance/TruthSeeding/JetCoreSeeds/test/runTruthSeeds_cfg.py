# Standalone test: build truth seeds around generator-level jets and store them.
#   cmsRun runTruthSeeds_cfg.py inputFiles=file:step1_GEN-SIM.root maxEvents=50
# Any file containing g4SimHits products works (GEN-SIM, FEVTDEBUG...).
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

opts = VarParsing('analysis')
opts.register('globalTag', 'auto:phase1_2024_realistic', VarParsing.multiplicity.singleton, VarParsing.varType.string,
              'global tag (must match the input geometry)')
opts.register('seedLayer', 'innermost', VarParsing.multiplicity.singleton, VarParsing.varType.string,
              "'innermost' or e.g. 'BPix2'")
opts.parseArguments()

process = cms.Process('TRUTHSEEDS', Run3)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, opts.globalTag, '')

process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.SimHitJetCoreSeedGenerator = dict()

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.maxEvents))
process.source = cms.Source('PoolSource', fileNames=cms.untracked.vstring(opts.inputFiles))

# truth jets as cores (GEN-SIM already contains ak4GenJetsNoNu)
process.genJetCores = cms.EDFilter('CandPtrSelector',
                                   src=cms.InputTag('ak4GenJetsNoNu'),
                                   cut=cms.string('pt > 100 && abs(eta) < 2.5'),
                                   filter=cms.bool(False))

from TruthSeeding.JetCoreSeeds.simHitJetCoreSeedGenerator_cfi import simHitJetCoreSeedGenerator
process.truthJetCoreSeeds = simHitJetCoreSeedGenerator.clone(cores='genJetCores', seedLayer=opts.seedLayer)

process.out = cms.OutputModule('PoolOutputModule',
                               fileName=cms.untracked.string(opts.outputFile or 'truthSeeds.root'),
                               outputCommands=cms.untracked.vstring('drop *',
                                                                    'keep *_truthJetCoreSeeds_*_*',
                                                                    'keep *_genJetCores_*_*',
                                                                    'keep *_ak4GenJetsNoNu_*_*'))

process.p = cms.Path(process.genJetCores + process.truthJetCoreSeeds)
process.e = cms.EndPath(process.out)
