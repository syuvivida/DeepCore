#!/usr/bin/env python3
#################################################################################################
##
##  Convert a trained DeepCore Keras model (.h5 / .weights.h5 / .keras) into a frozen
##  TensorFlow graph (.pb) that can be loaded in CMSSW (tensorflow::loadGraphDef).
##
##  Works with Keras 3 / TensorFlow 2.x (e.g. LCG_110a on lxplus el9):
##    source /cvmfs/sft.cern.ch/lcg/views/LCG_110a/x86_64-el9-gcc13-opt/setup.sh
##
##  Usage:
##    # full model saved with model.save(...)
##    python h5_to_pb.py --input DeepCore_model_ev34721_ep10.h5 --output DeepCore_ep10.pb
##
##    # weights only (model.save_weights(...)): the DeepCore 2.2 architecture is rebuilt here
##    python h5_to_pb.py --input DeepCore_train_ev34721_ep10.weights.h5 --output DeepCore_ep10.pb
##
##  Graph node names (must match what the CMSSW producer is configured with):
##    inputs : --inputNames  IMAGE ETA PT     (default: input_3 input_1 input_2)
##    outputs: --outputNames PAR PROB         (default: output_node0 output_node1)
##  The defaults reproduce the names of the original Keras-2 DeepCore graphs, where the
##  inputs were created in the order jet eta (input_1), jet pt (input_2), pixel map (input_3).
##  Check them against inputTensorName / outputTensorName in your CMSSW configuration.
##
#################################################################################################

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

parser = argparse.ArgumentParser(description="Convert a DeepCore Keras model (.h5) into a frozen TensorFlow graph (.pb)")
parser.add_argument('--input', required=True, help='.h5 / .keras full model, or .weights.h5 weights-only file')
parser.add_argument('--output', default='', help='output .pb file (default: input name with .pb)')
parser.add_argument('--inputNames', nargs=3, default=['input_3', 'input_1', 'input_2'],
                    metavar=('IMAGE', 'ETA', 'PT'), help='graph names of the pixel-map, jet-eta and jet-pt inputs')
parser.add_argument('--outputNames', nargs=2, default=['output_node0', 'output_node1'],
                    metavar=('PAR', 'PROB'), help='graph names of the track-parameter and probability outputs')
parser.add_argument('--weightsOnly', action='store_true',
                    help='treat --input as weights only and rebuild the DeepCore 2.2 architecture (automatic for *.weights.h5)')
parser.add_argument('--text', action='store_true', help='also write a human-readable .pbtxt')
parser.add_argument('--noCheck', action='store_true', help='skip the Keras vs frozen-graph comparison')
args = parser.parse_args()

# DeepCore configuration (must match the training)
jetDim = 30
layNum = 4
overlapNum = 3
parNum = 5


def build_model():
    """DeepCore 2.2 architecture (identical to DeepCore.py)"""
    from keras.layers import Input, Conv2D, Reshape, UpSampling2D, concatenate
    from keras.models import Model

    NNinputs_jeta = Input(shape=(1,))
    NNinputs_jpt = Input(shape=(1,))
    NNinputsJet = concatenate([NNinputs_jeta, NNinputs_jpt])
    jetReshaped = Reshape((1, 1, 2))(NNinputsJet)
    jetUps = UpSampling2D(size=(jetDim, jetDim), data_format="channels_last")(jetReshaped)
    NNinputs = Input(shape=(jetDim, jetDim, layNum))
    ComplInput = concatenate([NNinputs, jetUps], axis=3)

    conv30_9 = Conv2D(50, 7, data_format="channels_last", activation='relu', padding="same")(ComplInput)
    conv30_7 = Conv2D(40, 5, data_format="channels_last", activation='relu', padding="same")(conv30_9)
    conv30_5 = Conv2D(40, 5, data_format="channels_last", activation='relu', padding="same")(conv30_7)
    conv20_5 = Conv2D(30, 5, data_format="channels_last", activation='relu', padding="same")(conv30_5)
    conv15_5 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv20_5)

    conv15_3_1 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_5)
    conv15_3_2 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_3_1)
    conv15_3_3 = Conv2D(18, 3, data_format="channels_last", activation='relu', padding="same")(conv15_3_2)
    conv15_3 = Conv2D(18, 3, data_format="channels_last", padding="same")(conv15_3_3)
    reshaped = Reshape((jetDim, jetDim, overlapNum, parNum+1))(conv15_3)

    conv12_3_1 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv15_5)
    conv1_3_2 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv12_3_1)
    conv1_3_3 = Conv2D(30, 3, data_format="channels_last", activation='relu', padding="same")(conv1_3_2)
    conv1_3_1 = Conv2D(6, 3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    reshaped_prob = Reshape((jetDim, jetDim, overlapNum, 2))(conv1_3_1)

    return Model([NNinputs, NNinputs_jeta, NNinputs_jpt], [reshaped, reshaped_prob])


#------------------------------ load the Keras model ------------------------------#
in_path = args.input
weights_only = args.weightsOnly or in_path.endswith('.weights.h5')

if weights_only:
    print("Rebuilding the DeepCore 2.2 architecture and loading weights from", in_path)
    model = build_model()
    model.load_weights(in_path)
else:
    print("Loading full model from", in_path)
    # compile=False: the custom training losses are not needed (and not defined here)
    model = keras.models.load_model(in_path, compile=False)

model.summary()

if len(model.inputs) != 3 or len(model.outputs) != 2:
    raise SystemExit("Expected a model with 3 inputs and 2 outputs, got %d inputs and %d outputs"
                     % (len(model.inputs), len(model.outputs)))

# model input order is [pixel map, jet eta, jet pt]
in_shapes = [tuple(t.shape) for t in model.inputs]
print("model input shapes :", in_shapes)
print("model output shapes:", [tuple(t.shape) for t in model.outputs])

#------------------------------ freeze the graph ------------------------------#
img_name, eta_name, pt_name = args.inputNames
par_name, prob_name = args.outputNames

specs = [tf.TensorSpec([None] + list(in_shapes[0][1:]), tf.float32, name=img_name),
         tf.TensorSpec([None] + list(in_shapes[1][1:]), tf.float32, name=eta_name),
         tf.TensorSpec([None] + list(in_shapes[2][1:]), tf.float32, name=pt_name)]


@tf.function(input_signature=specs)
def serve(image, eta, pt):
    par, prob = model([image, eta, pt], training=False)
    return tf.identity(par, name=par_name), tf.identity(prob, name=prob_name)


concrete = serve.get_concrete_function()
frozen = convert_variables_to_constants_v2(concrete)
graph_def = frozen.graph.as_graph_def()

out_path = args.output or (in_path.replace('.weights.h5', '').replace('.keras', '').replace('.h5', '') + '.pb')
out_dir, out_file = os.path.split(os.path.abspath(out_path))
tf.io.write_graph(graph_def, out_dir, out_file, as_text=False)
if args.text:
    tf.io.write_graph(graph_def, out_dir, out_file.replace('.pb', '') + '.pbtxt', as_text=True)

node_names = {n.name for n in graph_def.node}
placeholders = [n.name for n in graph_def.node if n.op == 'Placeholder']
print("\nFrozen graph written to:", os.path.join(out_dir, out_file))
print("number of nodes     :", len(graph_def.node))
print("input  placeholders :", placeholders)
for name in (par_name, prob_name):
    print("output node %-14s: %s" % (name, "found" if name in node_names else "NOT FOUND"))
for name in (img_name, eta_name, pt_name):
    if name not in placeholders:
        print("WARNING: input placeholder '%s' not found in the graph" % name)

#------------------------------ check: Keras vs frozen graph ------------------------------#
if not args.noCheck:
    rng = np.random.default_rng(1)
    n = 4
    x_img = rng.random((n,) + tuple(in_shapes[0][1:]), dtype=np.float32)
    x_eta = rng.uniform(-1.4, 1.4, (n, 1)).astype(np.float32)
    x_pt = rng.uniform(500, 2000, (n, 1)).astype(np.float32)

    k_par, k_prob = model.predict([x_img, x_eta, x_pt], verbose=0)

    def _import():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped = tf.compat.v1.wrap_function(_import, [])
    g = wrapped.graph
    run = wrapped.prune([g.get_tensor_by_name(nm + ':0') for nm in (img_name, eta_name, pt_name)],
                        [g.get_tensor_by_name(nm + ':0') for nm in (par_name, prob_name)])
    f_par, f_prob = run(tf.constant(x_img), tf.constant(x_eta), tf.constant(x_pt))

    d_par = float(np.max(np.abs(k_par - f_par.numpy())))
    d_prob = float(np.max(np.abs(k_prob - f_prob.numpy())))
    print("\nCheck on %d random jets: max |Keras - frozen| par=%.3g prob=%.3g" % (n, d_par, d_prob))
    print("CHECK OK" if max(d_par, d_prob) < 1e-4 else "CHECK FAILED: outputs differ")
