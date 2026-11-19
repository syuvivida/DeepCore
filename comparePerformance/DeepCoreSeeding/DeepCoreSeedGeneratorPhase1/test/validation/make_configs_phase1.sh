#!/bin/bash
# Run 3 (Phase 1) step3 validation configs for CMSSW_14_2_2: default jet-core step vs DeepCore (TensorFlow, CMSSW
# model) vs DeepCore (PyTorch model through ONNX). DQM output only.
# usage: bash make_configs_phase1.sh [nThreads]      (in a directory of your choice, after cmsenv in CMSSW_14_2_2)
# environment:
#   ERA   (default Run3_2024)  COND (default auto:phase1_2024_realistic)  -- must match the step2 sample
#   PROCMOD extra procModifiers, comma separated, e.g. premix_stage2 for a premixed-pileup step2 sample
#         (its sim links are under "mixData" instead of "simSiPixelDigis"; check with edmDumpEventContent)
#   ONNX  weight file for the PyTorch model, relative to src/ (default .../DeepCoreSeedGeneratorPhase1/data/DeepCore_phase1.onnx)
NT=${1:-8}
ERA=${ERA:-Run3_2024}; COND=${COND:-auto:phase1_2024_realistic}
ONNX=${ONNX:-DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/data/DeepCore_phase1.onnx}
C=DeepCoreSeeding/DeepCoreSeedGeneratorPhase1/customise_DeepCorePhase1
VAL=$C.customisePhase1JetCoreValidation
[ -f "$CMSSW_BASE/src/$ONNX" ] || echo "WARNING: $CMSSW_BASE/src/$ONNX not found (needed by deepcore_onnx at run time)"

make() {  # name, customise list, [customise_commands], [procModifiers]
  local extra=() pm=()
  [ -n "$3" ] && extra=(--customise_commands "$3")
  local mods=$(echo "${PROCMOD:-},$4" | sed 's/^,*//; s/,*$//')
  [ -n "$mods" ] && pm=(--procModifiers "$mods")
  cmsDriver.py step3 --conditions $COND --era $ERA --geometry DB:Extended \
    --datatier DQMIO --eventcontent DQM \
    -s RAW2DIGI,L1Reco,RECO,RECOSIM,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
    "${pm[@]}" --customise "$2" "${extra[@]}" \
    --filein file:dummy.root --fileout file:step3_$1.root \
    --nStreams 1 --nThreads $NT -n -1 --python_filename step3_$1.py --no_exec > make_$1.log 2>&1 \
    && echo "made step3_$1.py" || { echo "FAILED step3_$1.py, see make_$1.log"; tail -5 make_$1.log; }
}
# 1) default Run-3 jet-core step (pixel-pair seeds); the Tracking/JetCore folder is added for comparison
make jetcore        "$VAL"
# 2) DeepCore as in CMSSW (TensorFlow, RecoTracker/TkSeedGenerator/data/DeepCore/..._barrel_2023.pb)
make deepcore_tf    "$VAL" "" seedingDeepCore
# 3) DeepCore with the PyTorch model (ONNX); same module parameters as 2), only the network differs
make deepcore_onnx  "$C.customiseDeepCoreONNX,$VAL" "process.jetCoreRegionalStepSeedsBarrel.weightFile = '$ONNX'" seedingDeepCore
