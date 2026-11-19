#!/bin/bash
# usage: bash harvest.sh <tag> <eos_outdir>   -> DQM_<tag>.root   (log: harvest_<tag>.log)
# Run 3 (Phase 1) version; ERA/COND as in make_configs_phase1.sh. The JetCore folder is harvested for all tags.
TAG=${1:?tag}; OUT=${2:?eos outdir}
ERA=${ERA:-Run3_2024}; COND=${COND:-auto:phase1_2024_realistic}
TOP=$PWD; WD=$TOP/harvest_$TAG; LOG=$TOP/harvest_$TAG.log
rm -rf $WD && mkdir -p $WD && cd $WD
ls $OUT/$TAG/*.root 2>/dev/null | sed 's#^#file:#' > files.txt
N=$(wc -l < files.txt)
if [ "$N" -eq 0 ]; then echo "[$TAG] no DQM files in $OUT/$TAG (jobs not finished or all failed?)"; exit 1; fi
echo "[$TAG] harvesting $N files ..."
cmsDriver.py step4 -s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM \
  --conditions $COND --era $ERA --geometry DB:Extended \
  --scenario pp --filetype DQM --mc -n -1 --filein filelist:files.txt \
  --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/customise_DeepCorePhase1.customisePhase1JetCoreHarvesting \
  --python_filename step4_$TAG.py --no_exec > $LOG 2>&1 || { echo "[$TAG] cmsDriver failed, see $LOG"; exit 1; }
cmsRun step4_$TAG.py >> $LOG 2>&1 < /dev/null || { echo "[$TAG] cmsRun failed, see $LOG"; tail -5 $LOG; exit 1; }
F=$(ls DQM_V0001_*.root 2>/dev/null | head -1)
[ -n "$F" ] || { echo "[$TAG] no DQM_V0001 output, see $LOG"; exit 1; }
mv $F $TOP/DQM_$TAG.root && echo "[$TAG] -> DQM_$TAG.root"
