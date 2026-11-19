#!/bin/bash
# usage: bash harvest.sh <tag> <eos_outdir>   -> DQM_<tag>.root   (log: harvest_<tag>.log)
# Runs in its own work directory, so several tags never collide; returns non-zero on failure.
TAG=${1:?tag}; OUT=${2:?eos outdir}
TOP=$PWD; WD=$TOP/harvest_$TAG; LOG=$TOP/harvest_$TAG.log
rm -rf $WD && mkdir -p $WD && cd $WD
ls $OUT/$TAG/*.root 2>/dev/null | sed 's#^#file:#' > files.txt
N=$(wc -l < files.txt)
if [ "$N" -eq 0 ]; then echo "[$TAG] no DQM files in $OUT/$TAG (jobs not finished or all failed?)"; exit 1; fi
echo "[$TAG] harvesting $N files ..."
cmsDriver.py step4 -s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM \
  --conditions auto:phase2_realistic_T35 --era Phase2C26I13M9 --geometry ExtendedRun4D128 \
  --scenario pp --filetype DQM --mc -n -1 --filein filelist:files.txt \
  --customise DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2.customiseHarvesting \
  --python_filename step4_$TAG.py --no_exec > $LOG 2>&1 || { echo "[$TAG] cmsDriver failed, see $LOG"; exit 1; }
cmsRun step4_$TAG.py >> $LOG 2>&1 < /dev/null || { echo "[$TAG] cmsRun failed, see $LOG"; tail -5 $LOG; exit 1; }
F=$(ls DQM_V0001_*.root 2>/dev/null | head -1)
[ -n "$F" ] || { echo "[$TAG] no DQM_V0001 output, see $LOG"; exit 1; }
mv $F $TOP/DQM_$TAG.root && echo "[$TAG] -> DQM_$TAG.root"
