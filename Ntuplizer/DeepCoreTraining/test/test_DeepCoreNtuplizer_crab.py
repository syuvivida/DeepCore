import CRABClient
from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'DeepCoreNtuplizer_pt_fix'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'test_DeepCoreNtuplizer.py' ## correct python file
config.JobType.numCores=8
config.JobType.maxMemoryMB=20000

config.section_("Data")
## config.Data.inputDataset = '/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/output'
## config.Data.userInputFiles = ['/eos/uscms/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/Ntuplizer_output1.root']
config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/hboucham-DeepCoreNtuplizerInput-e101c270c65c32f52437c2373244ff6e/USER'
config.Data.inputDBS = 'phys03' ##'phys03' ## might need to change to global?

## If using Lumi based split
#config.Data.splitting = 'EventAwareLumiBased'
#config.Data.unitsPerJob = 5000
#NJOBS = 380
#config.Data.totalUnits = config.Data.unitsPerJob * NJOBS

## If using File based split
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1

config.Data.publication = False
config.Data.outputDatasetTag = 'DeepCoreTrainingSample_pt_fix'

config.section_("Site")
config.Site.storageSite = "T2_CH_CERN"
