import CRABClient
from WMCore.Configuration import Configuration
config = Configuration()
config.section_("General")
config.General.requestName = 'DeepCorePrepareInput'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'test_DeepCorePrepareInput.py'
config.JobType.numCores=8
config.JobType.maxMemoryMB=20000

config.section_("Data")
#Barrel input
config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_17_0_0_pre2-160X_mcRun3_2026_realistic_v6_STD_2026_RecycledGS_noPU-v2/GEN-SIM-RECO'
config.Data.secondaryInputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_17_0_0_pre1-160X_mcRun3_2026_realistic_v6_STD_RegeneratedGS_2026_noPU_20260519_170947-v2/GEN-SIM'
#config.Data.inputDataset= '/RelValQCD_Pt_1800_2400_14/CMSSW_12_0_0_pre3-PU_120X_mcRun3_2021_realistic_v1_aodsim-v1/AODSIM'
#config.Data.secondaryInputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_12_0_0_pre3-120X_mcRun3_2021_realistic_v1-v1/GEN-SIM'
#config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_12_0_0_pre3-PU_120X_mcRun3_2021_realistic_v1_aodsim-v1/AODSIM'
#config.Data.secondaryInputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_12_0_0_pre3-PU_120X_mcRun3_2021_realistic_v1-v1/GEN-SIM-DIGI-RAW'
#config.Data.inputDataset = '/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17DRPremix-92X_upgrade2017_realistic_v10-v5/AODSIM'
#config.Data.secondaryInputDataset = '/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17GS-92X_upgrade2017_realistic_v10-v1/GEN-SIM'

#Encap input
# config.Data.inputDataset = '/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/AODSIM'
# config.Data.secondaryInputDataset = '/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/GEN-SIM-DIGI-RAW'

config.Data.inputDBS = 'global'
#config.Data.splitting = 'EventAwareLumiBased'


config.Data.splitting = 'Automatic'
#config.Data.unitsPerJob = 1000
#NJOBS = 9
#config.Data.totalUnits = config.Data.unitsPerJob * NJOBS

config.Data.outLFNDirBase = '/store/group/phys_tracking/syu/DeepCore/DeepCoreNtuplizerInput/' 
#config.Data.outLFNDirBase = '/store/user/%s/DeepCoreNtuplizerInput' % (getUsernameFromCRIC())
#config.Data.outLFNDirBase = '/eos/uscms/store/user/hichemb/DeepCoreNtuplizerInput' ## line needs to be removed since outdated
config.Data.publication = True
config.Data.outputDatasetTag = 'DeepCoreNtuplizerInput'


config.section_("Site")
#config.Site.storageSite = 'T2_IT_Pisa'
# T1_RU_JINR_Disk
config.Site.storageSite = 'T2_CH_CERN' 
##config.Site.storageSite = 'T3_US_FNALLPC' 
