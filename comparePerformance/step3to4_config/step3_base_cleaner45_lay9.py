# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase2_realistic_T35 --era Phase2C26I13M9 --geometry ExtendedRun4D128 --datatier DQMIO --eventcontent DQM -s RAW2DIGI,RECO,RECOSIM,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --procModifiers splitClustersInPhase2Pixel,jetCoreInPhase2 --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customiseJetCoreValidation --customise_commands 'process.jetCoreRegionalStepTrajectoryBuilder.keepOriginalIfRebuildFails = True; process.jetCoreRegionalStepTrajectoryBuilder.requireSeedHitsInRebuild = False; process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner = '"'"'jetCoreRegionalStepDeepCoreTrajectoryCleaner'"'"'; [setattr(ps, '"'"'minNumberLayers'"'"', 9) or setattr(ps, '"'"'minNumber3DLayers'"'"', 4) for ps in process.jetCoreRegionalStepSelector.trackSelectors]' --filein file:dummy.root --fileout file:step3_base_cleaner45_lay9.root --nStreams 1 --nThreads 8 -n -1 --python_filename step3_base_cleaner45_lay9.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
from Configuration.ProcessModifiers.splitClustersInPhase2Pixel_cff import splitClustersInPhase2Pixel
from Configuration.ProcessModifiers.jetCoreInPhase2_cff import jetCoreInPhase2

process = cms.Process('RECO',Phase2C26I13M9,splitClustersInPhase2Pixel,jetCoreInPhase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D128Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:dummy.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.obsolete.untracked.bool,
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_base_cleaner45_lay9.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.prevalidation_step = cms.Path(process.globalPrevalidationTrackingOnly)
process.validation_step = cms.EndPath(process.globalValidationTrackingOnly)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineTracking)
process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOffline)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,process.prevalidation_step,process.validation_step,process.dqmoffline_step,process.dqmofflineOnPAT_step,process.DQMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 1

# customisation of the process.

# Automatic addition of the customisation function from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.customise_DeepCorePhase2
from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.customise_DeepCorePhase2 import customiseJetCoreValidation 

#call to customisation function customiseJetCoreValidation imported from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.customise_DeepCorePhase2
process = customiseJetCoreValidation(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions


# Customisation from command line

process.jetCoreRegionalStepTrajectoryBuilder.keepOriginalIfRebuildFails = True; process.jetCoreRegionalStepTrajectoryBuilder.requireSeedHitsInRebuild = False; process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner'; [setattr(ps, 'minNumberLayers', 9) or setattr(ps, 'minNumber3DLayers', 4) for ps in process.jetCoreRegionalStepSelector.trackSelectors]
#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
process.options.TryToContinue = cms.untracked.vstring('BadAlloc')
