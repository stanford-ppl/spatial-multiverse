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
#
# This optimization script is based on these:
#     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py  (now deprecated by TensorFlow)
#     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md

print 'Generating commands for TensorFlow graph optimization passes.'
print 'The user needs to navigate to installed tensorflow/ directory and run these.'

cmd = '''
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=''' + model_no_ext + '''.pb \
--out_graph=''' + model_no_ext + '''_opt1.pb \
--inputs=\'''' + input_node + '''\' \
--outputs=\'''' + output_node + '''\' \
--transforms='
  strip_unused_nodes(type=float, shape="''' + str(DESIRED_BATCH_SIZE) + ''',''' + input_dims + '''")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(use_saved_model=false)
  fold_batch_norms
  fold_old_batch_norms
  fuse_pad_and_conv
  fuse_resize_and_conv
  fuse_resize_pad_and_conv
  merge_duplicate_nodes
  remove_control_dependencies
  sort_by_execution_order'
'''
# os.system(cmd)
print cmd

# Run the command again in case there are unresolved dependencies

cmd = '''
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=''' + model_no_ext + '''_opt1.pb \
--out_graph=''' + model_no_ext + '''_opt2.pb \
--inputs=\'''' + input_node + '''\' \
--outputs=\'''' + output_node + '''\' \
--transforms='
  strip_unused_nodes(type=float, shape="''' + str(DESIRED_BATCH_SIZE) + ''',''' + input_dims + '''")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(use_saved_model=false)
  fold_batch_norms
  fold_old_batch_norms
  fuse_pad_and_conv
  fuse_resize_and_conv
  fuse_resize_pad_and_conv
  merge_duplicate_nodes
  remove_control_dependencies
  sort_by_execution_order'
'''
# os.system(cmd)
print cmd

print '''

Run the commands above. They need the transform_graph utility to have been built. If this has not yet been
built, you can build it by running:

  bazel build tensorflow/tools/graph_transforms:transform_graph

Then cd to your tensorflow/ root dir and run the two commands above.

'''

print 'This will write the optimized graph to ' + model_no_ext + '_opt2.pb'
print '==> Next step: generate Spatial for this optimized graph by running:'
print '      $ python dnn_to_spatial.py ' + model_no_ext + '_opt2.pb'
