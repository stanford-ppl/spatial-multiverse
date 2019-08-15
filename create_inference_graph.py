#!/usr/bin/python

# ------------------------------------------------------------------------------
# 
# TensorFlow trained models come in two formats:
# - Checkpoint (NAME.meta and NAME.ckpt)
# - SavedModel (saved_model.pb)
#
# This script starts from either of these two and creates an unoptimized inference graph (.pb).
# It does this by calling TensorFlow APIs to freeze variables to constants and remove training nodes.
# It builds on top of:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
# The next script will optimize the output graph for FPGA synthesis using TensorFlow optimization utilities.
# In some cases however, those utilities miss optimization opportunities for graphs with control flow. So
# before freezing the graph in this script we also fold constants through control (Switch/Merge) operations.
# 
# ------------------------------------------------------------------------------

# ========================================================================================================
# Imports
# ========================================================================================================

import tensorflow as tf
import os.path
from tensorflow.python.framework import graph_util
import utils
import sys


# ========================================================================================================
# Parse arguments
# ========================================================================================================

args = utils.get_args(5, '''create_inference_graph.py  input_format  input_model  output_node_name  output_graph_dir  output_graph_name

Example 1: SavedModel

  Consider a directory models/resnet/ which contains a saved_model.pb. The output node of this graph
  (e.g. the SoftMax, or the input to the SoftMax) is called 'softmax_tensor'. Then use the script as follows:
     $ python create_inference_graph.py  saved_model  models/resnet/  softmax_tensor  models/resnet/  res50

Example 2: Checkpoint

  Consider a directory models/resnet/ which contains ResNet.meta and ResNet.ckpt. The output node of this graph
  (e.g. the SoftMax, or the input to the SoftMax) is called 'prob'. Then use the script as follows:
     $ python create_inference_graph.py  checkpoint  models/resnet/ResNet  prob  models/resnet/  res50

You can find the output node name in the code which generated the model, or you can use a dummy name, call this script,
and examine the initial graph summary to find the output node name, then rerun this script using that name.
''')

input_format = args[0]
model = args[1]
output_node_name = args[2]
output_dir = args[3]
output_name = args[4]


# ========================================================================================================
# Other Parameters
# ========================================================================================================

DESIRED_BATCH_SIZE = 1

# For inference can fix the input size.
# fixed_input_size = [DESIRED_BATCH_SIZE, 224, 224, 3]
fixed_input_size = None

# This is usually set False, only set True if graph has control flow (Switch, Merge)
fold_constants_through_control_ops = False
bool_switch_inputs_to_feed_in = [0, 1, 0, 1] # Runtime inputs of which subgraph to select in each Switch


# ========================================================================================================
# Convert Training Graph to Inference Graph
# ========================================================================================================

def get_inference_graph(sess, output_node_name):

  input_graph_def = sess.graph.as_graph_def(from_version=None, add_shapes=True)

  # Fix the shape if needed, e.g. so TensorFlow has static and not dynamic tensor shapes
  if fixed_input_size is not None:
    assert len(fixed_input_size) == 4
    assert fixed_input_size[0] == DESIRED_BATCH_SIZE
    assert fixed_input_size[1] == fixed_input_size[2]
    input_graph_def.node[0].attr['_output_shapes'].list.shape[0].unknown_rank = False
    for idx in range(len(fixed_input_size)):
      input_graph_def.node[0].attr['_output_shapes'].list.shape[0].dim.add()
      input_graph_def.node[0].attr['_output_shapes'].list.shape[0].dim[idx].size = fixed_input_size[idx]
      input_graph_def.node[0].attr['shape'].shape.dim.add()
      input_graph_def.node[0].attr['shape'].shape.dim[idx].size = fixed_input_size[idx]
  
  # The next script will run optimizations on the graph. In some cases however, those optimization utilities 
  # get confused by control operations. Usually this involves a constant boolean which used to be a variable 
  # in the graph indicating when to run a subgraph for training forward pass vs. for inference, e.g. dropout,
  # batch norm, or another path that was used by training but we want out of the inference graph. Some examples 
  # of where manual intervention is needed:
  # - Batch Norm: https://github.com/tensorflow/tensorflow/issues/8404
  # - Dropout: https://github.com/tensorflow/tensorflow/issues/5867
  # Workarounds for those specific graphs were provided but the more general issue is that constant folding does
  # not propagate through Switch/Merge. To get around these cases in a general way, the traversal below accepts 
  # a list of branch inputs and folds them into the graph. It can be used to fix the specific issues mentioned above 
  # and more generally to eliminate any subgraphs not needed by inference which are controlled by branching.
  if fold_constants_through_control_ops:
    merge_count = 0
    for node in input_graph_def.node:
      # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/merge
      # "Merge waits for at least one of the tensors in inputs to become available.
      #  It is usually combined with Switch to implement branching."
      if node.op == "Merge":
        # This Merge is fed by a Switch, but we only want to maintain
        # a particular subgraph of the parent Switch.
        input_to_break = 1  # The other input
        if bool_switch_inputs_to_feed_in[merge_count]:
          input_to_break = 0
        # print 'Deleting ' + node.input[input_to_break]
        del node.input[input_to_break]
        merge_count += 1
        node.op = "Identity"  # Now make this an Identity to pass through the Merge
      # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/switch
      if node.op == "Switch":
        del node.input[1] # Input 1 is the pred, remove since it's been fed in above
        node.op = "Identity"  # Now make this an Identity to pass through the Switch
        
  # Connections to unused subgraphs have now been broken.
  # Now need to constant fold through the replaced control nodes.
  # This is done in the next script, which runs TF folding utilities.
  # First do some final simplifications prior to constant folding:
  
  # Convert variabes -> constants
  freeze1 = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, [output_node_name])
  
  # This eliminates Identity and other nodes, e.g. CheckNumerics
  freeze2 = tf.graph_util.remove_training_nodes(freeze1, [output_node_name])
  
  # Eliminate unused branches
  freeze3 = tf.graph_util.extract_sub_graph(freeze1,[output_node_name])
  
  return freeze3


with tf.Session(graph=tf.Graph()) as sess:

  if input_format == 'saved_model':
    print 'Reading saved model from ' + model
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model)
    # Get the graph
    input_graph_def  = tf.get_default_graph().as_graph_def()
  else:
    assert input_format == 'checkpoint'
    print 'Reading ' + model + '.meta and ' + model + '.ckpt'  
    # Read the graph (in saver metadata) and the weights (in checkpoint)
    new_saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)    
    # Set the graph of the session
    ext = ''
    if os.path.isfile(model + '.ckpt'):
      ext = '.ckpt'
    new_saver.restore(sess, model + ext)
    # Get the graph
    input_graph_def  = sess.graph.as_graph_def(from_version=None, add_shapes=True)

  print '# Nodes before Freeze:' + str(len(input_graph_def.node))
  summary_filename = output_dir + '/' + output_name + '.node_list.initial'
  print 'Writing a summary of the initial graph to ' + summary_filename
  utils.write_summary_file(summary_filename, sess, input_graph_def)

  # Freeze graph
  output_graph_def = get_inference_graph(sess, output_node_name)
  
  # Save graph to file.
  model_file_full = output_dir + '/' + output_name + '.pb'
  if not os.path.isfile(model_file_full):
    print 'Writing frozen graph to ' + model_file_full
    tf.train.write_graph(output_graph_def, output_dir, output_name + '.pb', as_text=False)
    # with tf.gfile.GFile(model_file_full, "wb") as f:
    #   f.write(output_graph_def.SerializeToString())
  else:
    print 'Note: Tried writing frozen graph to ' + model_file_full + ', but file exists. Please move or delete that file'
    sys.exit(0)

  print '# Nodes after Freeze:' + str(len(output_graph_def.node))
  summary_filename = output_dir + '/' + output_name + '.node_list.frozen'
  print 'Writing a summary of the initial graph to ' + summary_filename
  utils.write_summary_file(summary_filename, sess, output_graph_def)


# ========================================================================================================
# Pointer to Next Step
# ========================================================================================================

print 'Finished'
print '==> Next step: optimize this graph using optimize_inference_graph.py'
