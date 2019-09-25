#!/usr/bin/python

# ------------------------------------------------------------------------------
# 
# optimize_inference_graph.py /path/to/model.pb
# 
# This script calls TensorFlow optimization utilities to convert a TensorFlow .pb graph to a more optimized 
# graph for FPGA synthesis. It requires installation of TensorFlow and Bazel.
# It should be run after create_inference_graph.py which creates the inference graph and folds control nodes.
# 
# ------------------------------------------------------------------------------

# ========================================================================================================
# Imports
# ========================================================================================================

import utils
import os
import tensorflow as tf
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph


# ========================================================================================================
# Parse arguments
# ========================================================================================================

args = utils.get_args(4, '''optimize_inference_graph.py  /path/to/model.pb  input_node  output_node  input_dims

Example:
     $ python  optimize_inference_graph.py  models/resnet/res50.pb  input_tensor  softmax_tensor  224,224,3
''')

model = args[0]
if not os.path.isabs(model):
  model = os.getcwd() + '/' + model
model_no_ext = '.'.join(model.split('.')[:-1])

input_node = args[1]
output_node = args[2]
input_dims = args[3]


# ========================================================================================================
# Other Parameters
# ========================================================================================================

DESIRED_BATCH_SIZE = 1


# ========================================================================================================
# Run Optimizations
# ========================================================================================================

# The graph is now frozen, so we can run optimizations like constant folding. This works even for graphs
# with Switch/Merge because control inputs were folded in by the previous script, create_inference_graph.py.

def get_graph_def_from_file(graph_filepath):
  tf.reset_default_graph()
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms, output_names, outname):
  input_names = [input_node]
  graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,  
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name=outname)

model_dir  = '/'.join(model_no_ext.split('/')[:-1])
model_name = model_no_ext.split('/')[-1]

transforms = ['strip_unused_nodes(type=float, shape="' + str(DESIRED_BATCH_SIZE) + ',' + input_dims + '")'
  'remove_nodes(op=Identity, op=CheckNumerics)',
  'fold_constants(use_saved_model=false)',
  'fold_batch_norms',
  'fold_old_batch_norms',
  'fuse_pad_and_conv',
  'fuse_resize_and_conv',
  'fuse_resize_pad_and_conv',
  'merge_duplicate_nodes',
  'remove_control_dependencies',
  'sort_by_execution_order']
optimize_graph(model_dir, model_name + '.pb',
               transforms, [output_node], outname= model_name + '_opt1.pb')
optimize_graph(model_dir, model_name + '_opt1.pb',
               transforms, [output_node], outname= model_name + '_opt2.pb')

print 'The optimized graph has been written to ' + model_no_ext + '_opt2.pb'
print '==> Next step: generate Spatial for this optimized graph by running:'
print '      $ python dnn_to_spatial.py ' + model_no_ext + '_opt2.pb'
