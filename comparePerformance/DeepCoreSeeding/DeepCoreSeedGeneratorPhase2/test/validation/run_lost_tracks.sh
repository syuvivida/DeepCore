#!/bin/bash
# Track-by-track study of the lost high-pT tracks (LostTrackAnalyzer), run interactively.
# usage: bash run_lost_tracks.sh INPUT NEVENTS name1 [name2 ...]
#   INPUT  : step2 file (LFN /store/..., /eos/... path or file:...)
#   names  : the step3_<name>.py configs made by make_configs.sh (e.g. nojetcore base_rebuild replace deepcore)
# output   : lostTracks_<name>.root (TTree lostTrackAnalyzer/tp) and lost_<name>.txt / lost_<name>.pdf
set -e
IN=$1; NEV=$2; shift 2
case $IN in
  /store/*) IN="root://eoscms.cern.ch//eos/cms$IN" ;;
  /eos/*)   IN="root://eoscms.cern.ch/$IN" ;;
esac
HERE=$(cd $(dirname $0) && pwd)
for N in "$@"; do
  [ -f step3_$N.py ] || { echo "step3_$N.py not found (run make_configs.sh first)"; continue; }
  cp step3_$N.py lost_${N}_cfg.py
  cat >> lost_${N}_cfg.py <<PY

# ---- LostTrackAnalyzer
from DeepCoreSeeding.DeepCoreSeedGeneratorPhase2.customise_DeepCorePhase2 import customiseLostTracks
process = customiseLostTracks(process, ptMin=100.)
process.TFileService.fileName = 'lostTracks_$N.root'
process.source.fileNames = ['$IN']
process.maxEvents.input = $NEV
for _n, _m in process.outputModules_().items():
    _m.fileName = 'lostdqm_${N}_' + _n + '.root'
PY
  echo "== $N: cmsRun lost_${N}_cfg.py ($NEV events)"
  cmsRun lost_${N}_cfg.py > lost_$N.log 2>&1 || { tail -40 lost_$N.log; exit 1; }
  python3 $HERE/../lost_track_summary.py lostTracks_$N.root --name $N --out lost_$N.pdf | tee lost_$N.txt
done
