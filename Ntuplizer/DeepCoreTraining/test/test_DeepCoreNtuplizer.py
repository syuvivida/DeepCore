import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
process = cms.Process('NTUPLE', Phase2C26I13M9)          

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D128Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

## process.GlobalTag.globaltag="94X_mc2017_realistic_v10"
#process.GlobalTag.globaltag= "120X_mcRun3_2021_realistic_v18"
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase2_realistic_T35','') 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) ) #-1 = tutti (numero edi eventi)

process.source = cms.Source("PoolSource",
    # replace 'myfile.root',' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:/eos/cms/store/group/phys_tracking/syu/DeepCore/DeepCoreNtuplizerInput/phase2/RelValQCD_Pt_1800_2400_14/Ntuplizer_step1_0_v2.root'
    ),
)

process.options = cms.untracked.PSet(
   allowUnscheduled = cms.untracked.bool(True),
   numberOfThreads = cms.untracked.uint32(8),
   numberOfStreams = cms.untracked.uint32(8),
   wantSummary = cms.untracked.bool(True)
)

process.DeepCoreNtuplizerTest = cms.EDProducer('DeepCoreNtuplizer' ,
 ptMin = cms.double(500), #500 used for barrel training, 1000 used for endcap training
 pMin=cms.double(0),
 deltaR = cms.double(0.1),
 barrelTrain =cms.bool(True),
 endcapTrain =cms.bool(False),
  
 vertices = cms.InputTag("offlinePrimaryVertices"),
 pixelClusters=cms.InputTag("siPixelClustersPreSplitting"),
 cores = cms.InputTag("ak4CaloJets"),
 centralMIPCharge = cms.double(18000.0),
 chargeFractionMin = cms.double(2),
 simTracks= cms.InputTag("g4SimHits"),
 simHit= cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
 simHitEC= cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
 pixelCPE = cms.string( "PixelCPEGeneric" )
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
 ignoreTotal = cms.untracked.int32(1),
 oncePerEventMode = cms.untracked.bool(True)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("DeepCoreTrainingSample_Eiko.root"),
    closeFileFast = cms.untracked.bool(True)
  )
  

#process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')                                             
#process.MessageLogger.cerr.DeepCoreNtuplizer = cms.untracked.PSet(                                              
#    limit = cms.untracked.int32(-1) # -1 means print every instance without limit                               
#)                                                                                                               




process.p = cms.Path(process.DeepCoreNtuplizerTest

) 
# 500 is the goodone (for barel training) #1000 is the tested one, with p cut insted of pt
