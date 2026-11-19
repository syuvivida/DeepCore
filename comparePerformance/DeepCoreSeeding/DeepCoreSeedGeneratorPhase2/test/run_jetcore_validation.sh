#!/bin/bash
# Build Phase-2 jet-core tracks with (a) the default pixel-pair seeds and
# (b) DeepCore Phase-2 seeds, then harvest and compare with the MTV.
#
# Usage (inside CMSSW_20_1_0_pre3/src after cmsenv and scram b):
#   bash run_jetcore_validation.sh step2.root [nEvents] [nThreads]
set -e
IN=${1:?give step2.root}
N=${2:-10}
NT=${3:-8}
CUST=DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2

COMMON="--conditions auto:phase2_realistic_T35 --era Phase2C26I13M9 --geometry ExtendedRun4D128 \
 --datatier GEN-SIM-RECO,DQMIO --eventcontent FEVTDEBUGHLT,DQM \
 -s RAW2DIGI,RECO,RECOSIM,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
 --procModifiers splitClustersInPhase2Pixel,jetCoreInPhase2 \
 --filein file:${IN} --nStreams 1 --nThreads ${NT} -n ${N} --no_exec"

for cfg in base deepcore; do
  if [ $cfg = base ]; then C="--customise ${CUST}.customiseJetCoreValidation"
  else C="--customise ${CUST}.customise,${CUST}.customiseJetCoreValidation"; fi
  cmsDriver.py step3 ${COMMON} ${C} \
     --fileout file:step3_${cfg}.root --python_filename step3_${cfg}.py
  cmsRun step3_${cfg}.py > step3_${cfg}.log 2>&1
  echo "step3 ${cfg} done (log step3_${cfg}.log)"

  cmsDriver.py step4 -s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM \
     --conditions auto:phase2_realistic_T35 --era Phase2C26I13M9 --geometry ExtendedRun4D128 \
     --scenario pp --filetype DQM --mc -n -1 --filein file:step3_${cfg}_inDQM.root \
     --customise ${CUST}.customiseHarvesting \
     --python_filename step4_${cfg}.py --no_exec
  cmsRun step4_${cfg}.py > step4_${cfg}.log 2>&1
  mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_${cfg}.root
done

# standard MTV web pages (generalTracks, per-iteration, vs dR(track,jet), ...)
makeTrackValidationPlots.py --extended -o plots_jetcore DQM_base.root DQM_deepcore.root

# dedicated jet-core comparison (Tracking/JetCore, chi2 associator) + summary table
python3 $CMSSW_BASE/src/DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test/compare_mtv_jetcore.py \
   --files DQM_base.root DQM_deepcore.root --names "pixel pairs" DeepCore \
   --folder JetCore --out mtv_jetcore.pdf
python3 $CMSSW_BASE/src/DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test/compare_mtv_jetcore.py \
   --files DQM_base.root DQM_deepcore.root --names "pixel pairs" DeepCore \
   --folder Track --labels generalTracks cutsRecoTracksJetCoreRegionalStepByOriginalAlgo cutsRecoTracksJetCoreRegionalStepByOriginalAlgoHp \
   --out mtv_track.pdf
