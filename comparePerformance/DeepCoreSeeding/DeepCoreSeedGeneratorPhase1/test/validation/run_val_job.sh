#!/bin/bash
# One HTCondor job: run a step3 config on one input file, copy the DQM file to EOS.
# args: CMSSW_BASE  config.py  input(LFN or path)  outdir  tag  jobid  [maxEvents]
set -e
CB=$1; CFG=$2; IN=$3; OUT=$4; TAG=$5; ID=$6; MAXEV=${7:--1}
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CB/src && eval $(scramv1 runtime -sh) && cd - > /dev/null
case $IN in
  /store/*) IN="root://eoscms.cern.ch//eos/cms$IN" ;;
  /eos/*)   IN="root://eoscms.cern.ch/$IN" ;;
esac
cp $CFG job_cfg.py
cat >> job_cfg.py <<PY
process.source.fileNames = ['$IN']
process.maxEvents.input = $MAXEV
for _n, _m in process.outputModules_().items():
    _m.fileName = 'out_' + _n + '.root'
PY
cmsRun job_cfg.py > job.log 2>&1 || { tail -50 job.log; exit 1; }
# copy to EOS with xrdcp (more robust than the FUSE mount on batch nodes)
case $OUT in
  /eos/user/*) XRD=root://eosuser.cern.ch ;;
  /eos/cms/*)  XRD=root://eoscms.cern.ch ;;
  *)           XRD="" ;;
esac
for f in out_*.root; do
  if [ -n "$XRD" ]; then
    xrdfs ${XRD#root://} mkdir -p $OUT/$TAG
    xrdcp -f $f $XRD/$OUT/$TAG/${TAG}_${ID}.root
  else
    mkdir -p $OUT/$TAG && cp $f $OUT/$TAG/${TAG}_${ID}.root
  fi
done
tail -5 job.log
