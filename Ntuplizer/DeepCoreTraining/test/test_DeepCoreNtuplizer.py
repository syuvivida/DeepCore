import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
##process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_esprefer_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

## process.GlobalTag.globaltag="94X_mc2017_realistic_v10"
#process.GlobalTag.globaltag= "120X_mcRun3_2021_realistic_v18"
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase1_2026_realistic','')
#121X_mcRun3_2021_realistic_v18 ## Updating global tag since we are using run 3 2021 mc
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) ) #-1 = tutti (numero edi eventi)

process.source = cms.Source("PoolSource",
    # replace 'myfile.root',' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:Ntuplizer_step1.root'
     #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_17_0_0_pre1/RelValQCD_Pt_1800_2400_14/GEN-SIM-RECO/160X_mcRun3_2026_realistic_v6_STD_RegeneratedGS_2026_noPU_20260519_170947-v1/2590000/252ce9ce-810e-4300-abc8-7a75943353d3.root'
     ## 'file:/eos/uscms/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/220127_205703/0000/Ntuplizer_output1_1.root'
     ## 'file:/eos/uscms/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/output/Ntuplizer_output1.root'
      ##  'file:/uscms_data/d3/hichemb/princeton/project2/CMSSW_12_0_0_pre4/src/RecoTracker/DeepCoreTraining/test/step3.root'
       ## 'root://cms-xrd-global.cern.ch//store/user/arizzi/TrainJetCore/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/TrainJetCoreAll/181026_130638/0005/step3_5435.root' #barrel example
        # 'root://cms-xrd-global.cern.ch//store/user/vbertacc/DeepCoreTrainingSampleEC_signelCore_2k/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/DeepCoreTrainingSampleEC_all/200509_143853/0000/step3_10.root'#endcap example
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
 fullTrain =cms.bool(False),
  
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
  
process.MessageLogger.cerr.threshold = "Info"
process.MessageLogger.debugModules = ["DeepCoreNtuplizerTest"]


process.p = cms.Path(process.DeepCoreNtuplizerTest

) 
# 500 is the goodone (for barel training) #1000 is the tested one, with p cut insted of pt
