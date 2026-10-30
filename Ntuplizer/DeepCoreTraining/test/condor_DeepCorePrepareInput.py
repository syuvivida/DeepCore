import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

# 1. Setup VarParsing for command-line arguments                                                                                                              
options = VarParsing.VarParsing('analysis')

options.register(
    'jobNum',
    0, # default to job 0 if run locally                                                                                                                      
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "HTCondor Process ID / Job index"
)

options.register(
    'eventsPerJob',
    10000, # default events per job
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of events to process per job"
)

options.parseArguments()

from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
process = cms.Process('TRAINING', Phase2C26I13M9)


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D128Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

# 2. Set event limits
events_to_skip = options.jobNum * options.eventsPerJob
print(events_to_skip)

jobNum=str(options.jobNum)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.eventsPerJob)
)
# 3. Read ALL input files into fileNames
with open("primaryFileList.txt", "r") as f:
    all_files = ['file:'+line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("secondaryFileList.txt", "r") as f:
    allsecond_files = ['file:'+line.strip() for line in f if line.strip() and not line.startswith("#")]    

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(all_files),
    secondaryFileNames = cms.untracked.vstring(allsecond_files),
    # Skip events across the combined file list
    skipEvents = cms.untracked.uint32(events_to_skip)
)


process.options = cms.untracked.PSet(
   allowUnscheduled = cms.untracked.bool(True),
   numberOfThreads = cms.untracked.uint32(8),
   numberOfStreams = cms.untracked.uint32(8),
   wantSummary = cms.untracked.bool(True)
)


process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:Ntuplizer_step1_'+jobNum+'.root'),
    outputCommands = cms.untracked.vstring(["drop *"]),
    splitLevel = cms.untracked.int32(0)
)

process.RECOSIMoutput.outputCommands.append("keep SimTracks_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep SimVertexs_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep PSimHits_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_ak4CaloJets_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_offlinePrimaryVertices_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_*iPixelCluster*_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_simSiPixelDigis_*_*")


# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase2_realistic_T35','') ## Using 2021 since we are using run 3 mc 

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
#process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.localreco)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.digi2raw_step)
process.schedule.extend([process.raw2digi_step,process.reconstruction_step])
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
