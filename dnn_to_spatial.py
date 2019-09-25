#!/usr/bin/python

# ------------------------------------------------------------------------------
# 
# dnn_to_spatial.py /path/to/model.[pb, graph, or pbtxt]
# 
# Converts from TensorFlow to a Spatial Language program. Support for other 
# frameworks is also planned.
# 
# ------------------------------------------------------------------------------

# ========================================================================================================
# Imports
# ========================================================================================================

import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import graph_pb2
import utils
import os


# ========================================================================================================
# Parse arguments
# ========================================================================================================

args = utils.get_args(1, 'dnn_to_spatial.py /path/to/model.[pb, graph, or pbtxt]')
model = args[0]


# ========================================================================================================
# Other Parameters
# ========================================================================================================

include_imagenet_classification = False # Can set true for CNNs using ImageNet to print classification ranking


# ========================================================================================================
# Read Model
# ========================================================================================================

input_format = model.split('.')[-1]

if input_format == 'pb':
  output_graph_def = graph_pb2.GraphDef()
  with open(model, "rb") as f:
    output_graph_def.ParseFromString(f.read())

elif input_format in ['graph', 'pbtxt']:
  output_graph_def = graph_pb2.GraphDef()
  from google.protobuf import text_format
  with open(model, "r") as f:
    text_format.Merge(f.read(), output_graph_def)

# Make a session for the frozen graph
tf.import_graph_def(output_graph_def)
frz_sess = tf.Session()
print

summary_filename = model + '.node_list.final'
print 'Writing a graph summary to ' + summary_filename
utils.write_summary_file(summary_filename, frz_sess, output_graph_def, imported=True)


# ========================================================================================================
# Initialize Device Parameters
# ========================================================================================================

device = 'vu9p'   # In the future, support other devices and read this as an input to the script

device_params = utils.read_config_file('devices/' + device + '.cfg')


# ========================================================================================================
# Initialize Design Parameters
# ========================================================================================================

reuse = True
reuse_FC = False
reuse_schedule = {}
reuse_args = {}  # Map layer to arg list, and each arg in that arg list to a value list
reuse_weight_dram = {}  # Map layer to weight dram list, and each dram in that list to a file list
reuse_fc_weight_dram = {}  # Map layer to weight dram list, and each dram in that list to a file list
reuse_tmp_dram_dims = {} # Map tmp DRAM to dims
reuse_tmp_dram_ordered = []
reuse_tmp_dram_children = {}
reuse_tmp_dram_parents = {}
reuse_layer_list = []
reuse_layer_to_ops = {}
reuse_layer_to_IP = {}
reuse_layer_to_kxk = {}
reuse_FC_name = ''
include_sigmoid_initialization = False
fc_section = False
max_depthwise_conv_input = None
processed_softmax = False


# ========================================================================================================
# Initialize each code block
# ========================================================================================================
app_name = model.split('/')[-1].split('.')[0].replace('_', '').replace('-', '').lower()
if include_imagenet_classification:
  args_help = '"/path/to/input.csv  /path/to/classes.csv  /path/to/weights/directory/"'
else:
  args_help = '"/path/to/input.csv  /path/to/weights/directory/"'
file_opening = '''package spatial.tests.apps

import spatial.dsl._
import utils.math._

@spatial class ''' + app_name + ''' extends SpatialTest {
  // override def compileArgs: Args = super.compileArgs and "--forceFuseFMA" and "--noBindParallels"
  override def runtimeArgs: Args = ''' + args_help + '''
  
  type T = FixPt[TRUE,_10,_22]
  
'''

var_declarations = ''
accel_global_LUTs = ''
accel_defs = ''
accel_function = ''
data_mem_declarations = ''
data_mem_set = ''
tmp_mem_declarations = ''
tmp_mem_declarations_no_reuse = ''
weight_mem_declarations = ''
weight_mem_declarations_no_reuse = ''
# accel_function_args = []
host_before_accel = '''  def main(args: Array[String]): Unit = {

    val debug:scala.Boolean = false
    
'''

# ========================================================================================================
# Global Data Structures
# ========================================================================================================

# Now that we've read the graph, we can go over the nodes. Each node is a NodeDef object.
# Contents of proto objects (node, op and attr) are here:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto
global_i_number = 0
global_c_number = 0
global_val_number = 0
name_to_tmpvar = {} # Instead of tmp, could use a legalized version of node.name (e.g. '/' -> '_')

# Refactor: name_to_node is only needed when we do not write the output for a node until we have gone through more of
# the graph. Currently this is done only for consts, which either (1) are written to disk after the graph is processed
# or (2) are scalars used by later nodes. (1) already exists in constants_to_write_to_disk, so name_to_node could be renamed 
# "tmpvar to scalars" and store only that case. Also, that map for (2) could be combined with extra_paddings, 
# since that is a special-case of (2)
name_to_node = {} # only used to get constant values
extra_paddings = {}
unpadded_dims = {}

# This is like name to node but tmpvar to node, for consts.
# Could use name to node but this is more specialized and can replace name_to_node
# by going name -> tmpvar then tmpvar -> node
constants_to_write_to_disk = {}
weight_files_to_concat = []

input_ops = ['Placeholder', 'DecodeJpeg']#, 'RandomUniform']

ops_to_skip = ['Softmax', 'Identity', 'NoOp', 'Squeeze', 'Shape']

# If we see these and they connect to a jpeg, skip them in Accel and do the op
# on host instead (e.g. decode, resize + interpolate using a Scala library)
jpeg_processing_ops = ['Cast', 'ExpandDims', 'ResizeBilinear', 'Sub', 'Mul']
nodes_used_for_jpeg_read = set()

# Same as above, but for preprocessing
preprocessing_ops = ['Mul', 'Split', 'Sub', 'Concat', 'ConcatV2', 'Pad']
nodes_used_for_preprocessing = set()

# Information across nodes, e.g. for fusion and whether an output is in SRAM and is 2D
conv_input_output_from_DRAM = True
# output_to_SRAM_or_DRAM = {}   # Can also make this a map
input_dram_already_created = set()

# Fusion
nodes_already_processed_through_fusion = set()

# Skip nodes for inference
# "Add a 50% dropout during training only. Dropout also scales
#  activations such that no rescaling is needed at evaluation time."
superops_to_skip = ['dropout']
current_dropout_input = ''

# Store reshape information per node
tmpvar_to_reshape_string = {}

# Proto to Spatial type translation map
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
proto_type_to_spatial_type = {}
proto_type_to_spatial_type[1] = 'T'
proto_type_to_spatial_type[3] = 'Int32'
proto_type_to_spatial_type[10] = 'Bool'
proto_type_to_spatial_type[12] = 'UInt8'

num_classes = None  # result of final add / softmax
final_node = None
final_node_SRAM = None
  
  
# ========================================================================================================
# Global helper functions
# ========================================================================================================

def closest_pow_2(n):
  closest = 1
  while True:
    if closest*2 > n:
      break
    closest = closest*2
  return int(closest)

# Return dimensions of a Const node
def get_const_dims(node):
  input_sizes = []
  for dim in node.attr['value'].tensor.tensor_shape.dim:
    input_sizes.append(str(dim.size))
  return input_sizes
  
# There are cases where large blocking of weights is not needed to reduce data
# read bandwidth, e.g. when data is in SRAM or inCh is small so data is small. 
# In such cases there is no need to load B*k*k in kxk convolution since B is small 
# and usually W load par is too. In these cases it is better to skip the reshape.
def skip_weight_reshape(dims):
  global conv_input_output_from_DRAM
  if conv_input_output_from_DRAM:
    # If data is small, B is also small so no need to reshape
    return int(dims[2]) <= 3
  else:
    # Since data in SRAM is smaller, use blocking for all but
    # grayscale to reduce overhead of pipeline iterations
    return int(dims[2]) == 1

def reformat_memory(dims):
  if len(dims) == 4:
    # 1x1 convolution weights are 4D but can be represented as 2D
    if dims[0] == '1':
      assert dims[1] == '1'
      return [dims[3], dims[2]]
    # elif skip_weight_reshape(dims):
    # Concat if small data since little compute, i.e. loads must be fast
    # (vs. if large data then we could save cycles by doing 3,49 as 3*49
    # but not bottleneck so that way can save a reshape)
    elif not conv_input_output_from_DRAM:
      return [dims[3], dims[2] + '*' + dims[0] + '*' + dims[1]]
    return [dims[3], dims[2], dims[0] + '*' + dims[1]]
  elif len(dims) == 3:
    return [dims[2], dims[0], dims[1]]
  elif len(dims) == 2:
    return [dims[1], dims[0]]
  else:
    return dims


# ========================================================================================================
# Generate Spatial for each node
# ========================================================================================================

# Initial traversal: Count number of FC layers
num_matmul = 0
for node in output_graph_def.node:
  if node.op == 'MatMul':
    num_matmul += 1
# Alternatively can always re-use (i.e. >1)
if num_matmul > device_params['num_SLR_regions']:
  reuse_FC = True

# Initial traversal: If Relu is used, see if Relu or Relu6
num_relu  = 0
num_relu6 = 0
for node in output_graph_def.node:
  if node.op == 'Relu':
    num_relu += 1
  elif node.op == 'Relu6':
    num_relu6 += 1
# For now assuming only one type is used, but not hard to support both,
# just need to add another mux.
assert num_relu == 0 or num_relu6 == 0
if num_relu6 > 0:
  use_relu6 = True
else:
  use_relu6 = False

# Initial traversal: ops with no inputs
# Make a directory for weights
weight_path = '.'.join(model.split('.')[0:-1]) + '_spatial_weights/'
if not os.path.exists(weight_path):
  os.mkdir(weight_path)
node_idx = 0
while(True):
  
  if node_idx >= len(output_graph_def.node):
    break
  node = output_graph_def.node[node_idx]
  node_idx += 1
  name_to_node[node.name] = node
  
  # ------------------------------------------------------------------------------------------------------
  # Skip certain nodes
  # ------------------------------------------------------------------------------------------------------
  # Do superop check first since it might contain other nodes to skip
  if node.name.split('/')[0] in superops_to_skip:
    continue

  # ------------------------------------------------------------------------------------------------------
  # Input or constant
  # ------------------------------------------------------------------------------------------------------
  # Check if input op or constant
  if node.op in input_ops:
  
    # See if we want to skip this
    if node.attr["shape"].shape.unknown_rank:
     continue
    
    tmpvar = 'i' + str(global_i_number)
    name_to_tmpvar[node.name] = tmpvar
    if node.op == 'Placeholder':
      nodes_used_for_preprocessing.add(node.name)
      input_sizes = []
      # Can use get_tensor_dims_from_input_name for this
      for dim in node.attr["shape"].shape.dim[1:]:  # Recall Tensorflow stores in NHWC
        input_sizes.append(str(dim.size))
      assert len(node.input) == 0
      host_before_accel += '    val ' + tmpvar + ' = loadCSV1D[T](args(' + str(global_i_number) + '), "\\n")'
      if len(input_sizes) > 1:
        host_before_accel += '.reshape(' + ','.join(reformat_memory(input_sizes)) + ')'
      # Omit reshape string for this because we just reshaped it above if it needed reshaping.
      # If there is an explicit Reshape called on it, it will be added back here later.
      # tmpvar_to_reshape_string[tmpvar] = ','.join(reformat_memory(input_sizes))
      
      # Check: if this input is small, keep in SRAM
      total_input_size = 1
      for dim in node.attr["shape"].shape.dim[1:]:
        total_input_size = total_input_size * dim.size
      if total_input_size < device_params['image_buffer_size']:
        conv_input_output_from_DRAM = False
        reuse = False # Can also check number of layers directly. Smaller input tends to = fewer layers.
        # reuse_FC = False # Can also check size of 1st FC, but if conv is in SRAM then FC data will also be small
      
    elif node.op == 'DecodeJpeg':
      assert len(node.input) == 1
      nodes_used_for_jpeg_read.add(node.name)
      host_before_accel += '    val ' + tmpvar + ' = DecodeJPEG("input' + str(global_i_number) + '.jpg")'
    host_before_accel += "\n"
    global_i_number += 1
    # accel_function_args.append(tmpvar)
    continue
    
  # Constants
  elif node.op == 'Const':
    assert len(node.input) == 0
    dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
    
    # Check if we should store these weights as a file or hard-code them
    # E.g. sometimes there is a tensor content but it is only 1 word
    store_weights_as_file = False
    scalar_weight_is_in_tensor = False
    pads_are_in_tensor = False
    if node.attr['value'].tensor.tensor_content:
      store_weights_as_file = True
      # dtype 1 means tf DT_FLOAT
      if len(node.attr['value'].tensor.tensor_content) == 4 and node.attr['value'].tensor.dtype in [1,3]:
        store_weights_as_file = False
        scalar_weight_is_in_tensor = True
      elif len(node.attr['value'].tensor.tensor_content) == 4*8 and node.attr['value'].tensor.dtype in [3]: # Paddings
        assert 'paddings' in node.name
        store_weights_as_file = False
        pads_are_in_tensor = True
    
    if store_weights_as_file:
      tmpvar = 'c' + str(global_c_number)
      name_to_tmpvar[node.name] = tmpvar
      # Store this node and later save it to file
      if not '/shape' in node.name and not '/reduction_indices' in node.name:
        constants_to_write_to_disk[tmpvar] = node
        input_sizes = get_const_dims(node)
        tmpvar_to_reshape_string[tmpvar] = ','.join(reformat_memory(input_sizes))
        global_c_number += 1
        # accel_function_args.append(tmpvar)
    elif pads_are_in_tensor:
      tmp_file = open('TMP_FILE_CONST', 'wb')
      tmp_file.write(node.attr['value'].tensor.tensor_content)
      tmp_file.close()
      import numpy as np
      tmp_in = np.fromfile('TMP_FILE_CONST', dtype=np.int32, count=-1)
      assert tmp_in.size == 8
      # print 'extra_paddings[' + node.name + '] = ' + str(tmp_in.tolist())
      extra_paddings[node.name] = tmp_in.tolist()
      name_to_tmpvar[node.name] = None
    else:
      val = ''
      if scalar_weight_is_in_tensor:
        tmp_file = open('TMP_FILE_CONST', 'wb')
        tmp_file.write(node.attr['value'].tensor.tensor_content)
        tmp_file.close()
        import numpy as np
        tmp_in = np.fromfile('TMP_FILE_CONST', dtype=np.float32, count=-1)
        assert tmp_in.size == 1
        val = str(tmp_in[0])
      elif node.attr['value'].tensor.float_val:
        val = str(node.attr['value'].tensor.float_val[0])
      elif node.attr['value'].tensor.int_val:
        val = str(node.attr['value'].tensor.int_val[0])
      #elif node.attr['value'].tensor.bool_val:
      #  val = str(node.attr['value'].tensor.bool_val[0])
      assert val
      name_to_tmpvar[node.name] = val + '.to[' + dtype + ']'
      # print node.name + ' = ' + str(val)
    continue

# Next traversal: remaining ops with inputs
node_idx = 0
while(True):
  
  # Define helpers used by various ops
  
  def burst_align(dim):
    import math
    burst_size = 16
    return int(math.ceil(float(dim)/float(burst_size))*int(burst_size))

  def get_tensor_dims_from_input_name(tensor_name, frz_sess):
    if ':' in tensor_name:
      tensor_name = 'import/' + tensor_name
    else:
      tensor_name = 'import/' + tensor_name + ':0'
    tensor = frz_sess.graph.get_tensor_by_name(tensor_name)
    
    """
    print tf.shape(tensor)
    with tf.Session() as sess:
        print sess.run(tf.shape(tensor))
    """
    
    assert tensor.get_shape()
    return tensor.get_shape()
    
  def get_dims_str(tensor_name, frz_sess, ignore_initial=0):
    dims_tensor = get_tensor_dims_from_input_name(tensor_name, frz_sess)[ignore_initial:]
    if not dims_tensor.dims:
      # Sometimes an input to a node has no dims, but we
      # can infer them from the sizes of later ops
      return None
    dims = dims_tensor.as_list()
    dims_str = []
    for dim in dims:
      if dim:
        dims_str.append(str(dim))
      else:
        dims_str.append(None)
    # print dims_str
    return dims_str
  
  if node_idx >= len(output_graph_def.node):
    # Done processing nodes, so make output the last node's output
    assert num_classes
    data_mem_declarations += '    val ' + final_node + '_DRAM = DRAM[T](' + str(burst_align(num_classes)) + ')' + "\n"
    accel_function += '        ' + final_node + '_DRAM(0::' + str(burst_align(num_classes)) + ') store ' + final_node_SRAM + '_SRAM' + "\n"
    break
  node = output_graph_def.node[node_idx]
  node_idx += 1
  name_to_node[node.name] = node

  # ------------------------------------------------------------------------------------------------------
  # Skip certain nodes
  # ------------------------------------------------------------------------------------------------------
  # Do superop check first since it might contain other nodes to skip
  if node.name.split('/')[0] in superops_to_skip:
    superop = node.name.split('/')[0]
    if superop == 'dropout':
      if node.name in ['dropout/truediv', 'dropout/div', 'dropout/dropout/div']:
        assert len(node.input) == 2
        current_dropout_input = node.input[0]
      elif node.name in ['dropout/mul', 'dropout/dropout/mul']:
        assert current_dropout_input
        name_to_tmpvar[node.name] = name_to_tmpvar[current_dropout_input]
    continue
  
  if node.name in nodes_already_processed_through_fusion:
    continue
  
  if node.op in jpeg_processing_ops:
    if node.input[0] in nodes_used_for_jpeg_read:
      nodes_used_for_jpeg_read.add(node.name)
      name_to_tmpvar[node.name] = name_to_tmpvar[node.input[0]]
      host_before_accel += '    // Apply ' + node.name + ' to ' + name_to_tmpvar[node.name]
      if len(node.input) > 1:
        host_before_accel += '  (with value = ' + str( tensor_util.MakeNdarray(name_to_node[node.input[1]].attr['value'].tensor) ) + ')'
      host_before_accel += "\n" 
      
      continue
  
  """
  Preprocessing node pattern example:
  
    images
      op = Placeholder
    mul/y
      op = Const
    mul
      op = Mul
      in0 = images
      in1 = mul/y
    split/split_dim
      op = Const
    split
      op = Split
      output size = (1, 224, 224, 1)
      in0 = split/split_dim
      in1 = mul
    concat/concat_dim
      op = Const
    concat
      op = Concat
      output size = (1, 224, 224, 3)
      in0 = concat/concat_dim
      in1 = split:2
      in2 = split:1
      in3 = split
    sub/y
      op = Const
      output size = (3,)
    sub
      op = Sub
      output size = (1, 224, 224, 3)
      in0 = concat
      in1 = sub/y
  """
  if node.op in preprocessing_ops:
    # We have found a node type associated with preprocessing. Now check that this node is part of the initial
    # preprocessing of the input, and check for all inputs (any could be a preprocessing node)
    preprocessing_input_node = None
    for input_name in node.input:
      # For split
      if ':' in input_name:
        input_name = ''.join(input_name.split(':')[:-1])
      if input_name in nodes_used_for_preprocessing:
        preprocessing_input_node = input_name
        break
    
    # If the parent is also a preprocessing node, apply this preprocessing operation.
    # Can also fuse these like this:
    """
      val i0_preprocessed = (0::3, 0::224, 0::224){(i,j,k) =>
        if      (i == 0) { i0(2,j,k)*255.0.to[T] - 103.06262207.to[T] }
        else if (i == 1) { i0(1,j,k)*255.0.to[T] - 115.90288544.to[T] }
        else             { i0(0,j,k)*255.0.to[T] - 123.15163422.to[T] }
      };
    """
    # When there is no preprocessing in the graph, it can be done as an extra input from the user when converting image -> csv, or if input to
    # Spatial is later a jpg it can be done in fused way above based on user input.
    # Can also support more cases, e.g. in the case of split -> sub -> concat, there are 3 subs, not 1 (above example is split -> concat -> sub).
    if preprocessing_input_node:
      # Get the remaining inputs
      other_inputs = []
      for input_name in node.input:
        if ':' in input_name:
          input_name = ''.join(input_name.split(':')[:-1])
        if input_name != preprocessing_input_node:
          other_inputs.append(input_name)
      assert len(other_inputs) == 1
      # Add the current node to the preprocessing nodes
      nodes_used_for_preprocessing.add(node.name)
      input_dims = reformat_memory(get_dims_str(preprocessing_input_node, frz_sess, 1))
      assert len(input_dims) in [2,3]
      num_input_channels = 1
      if len(input_dims) == 3:
        num_input_channels = int(input_dims[0])
      output_dims = reformat_memory(get_dims_str(node.name, frz_sess, 1))
      assert len(output_dims) in [2,3]
      num_output_channels = 1
      if len(output_dims) == 3:
        num_output_channels = int(output_dims[0])
      tmpvar_input = name_to_tmpvar[preprocessing_input_node]
      # Perform this node's operation on current input
      if node.op == 'Mul':
        tmpvar = tmpvar_input + '_scale'
        name_to_tmpvar[node.name] = tmpvar
        assert num_input_channels in [1,3]
        assert num_input_channels == num_output_channels
        host_before_accel += '    val ' + tmpvar + ' = (0::' + ', 0::'.join(input_dims) + ')'
        if num_input_channels == 1:
          host_before_accel += '{i => ' + tmpvar_input + '(i)'
        else:
          host_before_accel += '{(i,j,k) => ' + tmpvar_input + '(i,j,k)'
        host_before_accel += '*' + str( tensor_util.MakeNdarray(name_to_node[other_inputs[0]].attr['value'].tensor) ) + '.to[T]};' + "\n"
      elif node.op == 'Sub':
        tmpvar = tmpvar_input + '_meansub'
        name_to_tmpvar[node.name] = tmpvar
        assert num_input_channels in [1,3]
        assert num_input_channels == num_output_channels
        host_before_accel += '    val ' + tmpvar + ' = (0::' + ', 0::'.join(input_dims) + ')'
        subtraction_values = tensor_util.MakeNdarray(name_to_node[other_inputs[0]].attr['value'].tensor)
        if num_input_channels == 1:
          assert subtraction_values.size == 1
          host_before_accel += '{i => ' + tmpvar_input + '(i) - ' + str(subtraction_values) + '.to[T]};'
        else:
          assert subtraction_values.size == 3
          host_before_accel += '''{(i,j,k) =>
      if      (i == 0) { ''' + tmpvar_input + '''(0,j,k) - ''' + str(subtraction_values[0]) + '''.to[T] }
      else if (i == 1) { ''' + tmpvar_input + '''(1,j,k) - ''' + str(subtraction_values[1]) + '''.to[T] }
      else             { ''' + tmpvar_input + '''(2,j,k) - ''' + str(subtraction_values[2]) + '''.to[T] }
    };'''
        host_before_accel += "\n"
      elif node.op == 'Split':
        tmpvar = tmpvar_input + '_split'
        name_to_tmpvar[node.name] = tmpvar
        split_axis = tensor_util.MakeNdarray(name_to_node[other_inputs[0]].attr['value'].tensor)
        assert split_axis.size == 1
        assert split_axis == 3
        assert num_input_channels == 3
        assert num_output_channels == 1
        host_before_accel += '    val ' + tmpvar + '_0 = (0::' + ', 0::'.join(input_dims[1:]) + ')'
        host_before_accel += '{(j,k) => ' + tmpvar_input + '(0,j,k)};' + "\n"
        host_before_accel += '    val ' + tmpvar + '_1 = (0::' + ', 0::'.join(input_dims[1:]) + ')'
        host_before_accel += '{(j,k) => ' + tmpvar_input + '(1,j,k)};' + "\n"
        host_before_accel += '    val ' + tmpvar + '_2 = (0::' + ', 0::'.join(input_dims[1:]) + ')'
        host_before_accel += '{(j,k) => ' + tmpvar_input + '(2,j,k)};' + "\n"
      elif node.op == 'Concat':
        tmpvar = tmpvar_input + '_concat'
        name_to_tmpvar[node.name] = tmpvar
        concat_axis = tensor_util.MakeNdarray(name_to_node[other_inputs[0]].attr['value'].tensor)
        assert concat_axis.size == 1
        assert concat_axis == 3
        assert num_input_channels == 1
        assert num_output_channels == 3
        # Get the concat order
        concat_order = []
        for input_name in node.input:
          if input_name == other_inputs[0]:
            continue
          elif ':' not in input_name:
            concat_order.append('0')
          else:
            concat_order.append(input_name.split(':')[-1])
        assert len(concat_order) == 3
        # Concat
        host_before_accel += '    val ' + tmpvar + ' = (0::' + ', 0::'.join(output_dims) + ')'
        host_before_accel += '''{(i,j,k) =>
      if      (i == 0) { ''' + tmpvar_input + '_' + concat_order[0] + '''(j,k) }
      else if (i == 1) { ''' + tmpvar_input + '_' + concat_order[1] + '''(j,k) }
      else             { ''' + tmpvar_input + '_' + concat_order[2] + '''(j,k) }
    };'''
        host_before_accel += "\n"
      elif node.op == 'Pad':
        tmpvar = tmpvar_input + '_pad'
        name_to_tmpvar[node.name] = tmpvar
        assert num_input_channels in [1,3]
        host_before_accel += '    val ' + tmpvar + ' = (0::' + ', 0::'.join(output_dims) + ')'
        assert node.input[1] in extra_paddings.keys()
        padding = int(extra_paddings[node.input[1]][2])
        start = str(padding)
        end   = str(int(output_dims[1]) - padding)
        if num_input_channels == 1:
          host_before_accel += '{(j,k)   => if (j>=' + start + ' && j<' + end + ' && k>=' + start + ' && k<' + end + ') ' + tmpvar_input + '(j-' + start + ',k-' + start + ')     else 0.to[T]'
        else:
          host_before_accel += '{(i,j,k) => if (j>=' + start + ' && j<' + end + ' && k>=' + start + ' && k<' + end + ') ' + tmpvar_input + '(i,j-' + start + ',k-' + start + ') else 0.to[T]'
        host_before_accel += '};' + "\n"
      else:
        assert False
      continue
  
  if node.op in ops_to_skip:
    # Skip this op by short-circuiting names
    # Reshape also short-circuits below when 1D to 1D
    if node.input[0] in name_to_tmpvar.keys():
      name_to_tmpvar[node.name] = name_to_tmpvar[node.input[0]]
    if node.op == 'Softmax':
      processed_softmax = True
    continue
  # Alternatively, could require the user to specify the output as the Softmax or something before
  if processed_softmax:
    continue

  # ------------------------------------------------------------------------------------------------------
  # Input or constant (already processed by iteration 1)
  # ------------------------------------------------------------------------------------------------------
  # Check if input op or constant
  if node.op in input_ops:
    continue
  elif node.op == 'Const':
    continue
  
  # ------------------------------------------------------------------------------------------------------
  # Computation node
  # ------------------------------------------------------------------------------------------------------
  # Otherwise we have an op  
  
  # Define helpers specific to compute nodes
  
  def get_inputs(node, name_to_tmpvar):
    node_tmpvar_inputs = []
    for input in node.input:
      if ':' in input:
        input = ''.join(input.split(':')[:-1])
      if input in name_to_tmpvar.keys():
        node_tmpvar_inputs.append(name_to_tmpvar[input])
      else:
        # node_tmpvar_inputs.append(input)
        print 'ERROR: Could not get input tmpvar for node ' + node.name + ', input ' + input
        assert False
    return node_tmpvar_inputs
  
  def get_data_dims_str(node, frz_sess):
    return get_dims_str(node.input[0], frz_sess, 1)
  
  def get_kernel_dims_str(node, frz_sess):
    return get_dims_str(node.input[1], frz_sess, 0)
  
  # Return the window size for a pool node
  # For a conv this isn't needed since the kernel is inputs[1]
  def get_pool_kernel_dims(node):
    kernel_dims = []
    for k in node.attr['ksize'].list.i:
      kernel_dims.append(str(int(k)))
    assert len(kernel_dims) == 4
    assert kernel_dims[0] == kernel_dims[3]
    assert kernel_dims[1] == kernel_dims[2]
    return kernel_dims
    
  # Return output dims and stride/padding for a conv or pool node
  # Can also see node_list.final, this information is within the tensor already.
  # E.g. call get_dims_str() on the node to get the output shape.
  # https://www.tensorflow.org/api_docs/python/tf/nn/convolution
  def get_output_dim(node, kernel_dims, input_dims, out_channels):
    padding = node.attr['padding'].s
    strides = []
    for s in node.attr['strides'].list.i:
      strides.append(str(int(s)))
    assert len(strides) == 4
    assert strides[0] == strides[3]
    assert strides[1] == strides[2] # Eventually not needed
    # For SAME, padding can also be forced by a fused tf.pad
    if padding == 'SAME':
      import math
      out_height = int(math.ceil(float(int(input_dims[0])) / float(strides[1])))
      out_width  = int(math.ceil(float(int(input_dims[1])) / float(strides[2])))
      out_size = str(out_height) + ',' + str(out_width) + ',' + str(out_channels)
    else:
      assert padding == 'VALID'
      import math
      out_height = int(math.ceil(float(int(input_dims[0]) - int(kernel_dims[0]) + 1) / float(strides[1])))
      out_width  = int(math.ceil(float(int(input_dims[1]) - int(kernel_dims[1]) + 1) / float(strides[2])))
      out_size = str(out_height) + ',' + str(out_width) + ',' + str(out_channels)
    return out_height, out_width, out_size, strides, padding
    
  def get_reshape_string(name):
    if name in tmpvar_to_reshape_string.keys():
      reshape_string = '.reshape(' + tmpvar_to_reshape_string[name] + ')'
      if len(reshape_string.split(',')) > 1:
        return reshape_string
    return ''

  # Return the nodes in the pattern if they exist
  # Can also make this depend on graph not topological sort
  def fusion_optimization_match(fusion_pattern, node_list, curr_idx):
    # Now go over the fusion pattern and check if there is a mismatch
    curr_fusion_pattern_op = 0
    fusion_nodes = []
    while True:
      # If exhausted fusion ops, pattern found
      if curr_fusion_pattern_op >= len(fusion_pattern):
        break
      # If not exhausted fusion ops but exhausted nodes, pattern not found
      if curr_idx >= len(node_list):
        return False
      node = node_list[curr_idx]
      # If constant or input, skip
      if node.op in input_ops or node.op == 'Const':
        curr_idx += 1
        continue
      # If match, add to list
      # Can refactor this
      if node.op == fusion_pattern[curr_fusion_pattern_op]:
        fusion_nodes.append(node)
        curr_fusion_pattern_op += 1
        curr_idx += 1
      elif node.op in ['Add', 'BiasAdd'] and fusion_pattern[curr_fusion_pattern_op] in ['Add', 'BiasAdd']:
        fusion_nodes.append(node)
        curr_fusion_pattern_op += 1
        curr_idx += 1
      elif node.op in ['Relu', 'Relu6'] and fusion_pattern[curr_fusion_pattern_op] in ['Relu', 'Relu6']:
        fusion_nodes.append(node)
        curr_fusion_pattern_op += 1
        curr_idx += 1
      # If mismatch, pattern not found
      else:
        return False
    return fusion_nodes
  
  # Register a new layer to reuse
  def register_new_reused_processor(op_name, def_arg_values, hw_block, \
      dse_string='', has_weights=False, bias_name=None, weight_name=None):
  
    global reuse_layer_list
    global accel_function
    global reuse_schedule
    global reuse_args
    global reuse_weight_dram
    global weight_files_to_concat
    global file_opening
    global accel_defs
  
    # If this is the first reuse layer, initiate this code block
    if not reuse_layer_list:
      accel_function += '''
        {{{INSERT_REUSE_LOOP_HERE}}}
'''
    # If this is the first time for this op name, create a new entry
    if op_name not in reuse_layer_list:
      # Could also make check a single LUT with values for layer type (e.g. 1 2 3)
      reuse_schedule[op_name] = []
      # Add args map for this function as well
      new_map = {}
      for arg in def_arg_values.keys():
        new_map[arg] = []
      reuse_args[op_name] = new_map
      
      if has_weights:
        new_dram_map = {}
        new_dram_map['bias'] = []
        new_dram_map['weights'] = []
        reuse_weight_dram[op_name] = new_dram_map
      
      # Print the def header
      file_opening += dse_string
      accel_defs += '      def ' + op_name + '(' + ', '.join(sorted(def_arg_values.keys())) \
        + ', L : Int) : Unit = {' + "\n" + hw_block + '      }' + "\n\n"
      
    # Now register this occurence of the layer in the entry
        
    # LUTs can also be local to the def
    for arg in def_arg_values.keys():
      # if def_arg_values[arg]:
      reuse_args[op_name][arg].append(def_arg_values[arg])
      
    if has_weights:
      reuse_weight_dram[op_name]['bias'].append(bias_name)
      reuse_weight_dram[op_name]['weights'].append(weight_name)
      weight_files_to_concat.append(bias_name)
      weight_files_to_concat.append(weight_name)
    
    reuse_schedule[op_name].append(str(len(reuse_layer_list)))
    # Append this layer to the list
    reuse_layer_list.append(op_name)
  
  # Register a new layer for static model
  def register_layer_ops(op_name, total_ops, ip, kxk):
  
    global reuse_layer_to_ops
    global reuse_layer_to_IP
    global reuse_layer_to_kxk
  
    if op_name not in reuse_layer_to_ops.keys():
      reuse_layer_to_ops[op_name] = 0
    reuse_layer_to_ops[op_name] += total_ops
    reuse_layer_to_IP[op_name] = ip
    reuse_layer_to_kxk[op_name] = kxk
  
  # From https://www.tensorflow.org/versions/r1.12/api_guides/python/nn#Convolution
  # Can also first check special-case of forced padding from a fused tf.pad. If not, then return this calculation.
  def get_same_padding(in_height, in_width, strides, filter_height, filter_width):
    if in_height % int(strides[1]) == 0:
      pad_along_height = max(filter_height - int(strides[1]), 0)
    else:
      pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if in_width % int(strides[2]) == 0:
      pad_along_width = max(filter_width - int(strides[2]), 0)
    else:
      pad_along_width = max(filter_width - (in_width % int(strides[2])), 0)
    pad_top  = int(pad_along_height) / int(2)
    pad_left = int(pad_along_width)  / int(2)
    pad_bottom = pad_along_height - pad_top
    pad_right  = pad_along_width - pad_left
    return pad_top, pad_left, pad_bottom, pad_right

  # For 1x1 convolution the weights are small so we can load a block of weights for multiple in channels
  # Instead of a LUT this can also be calculated dynamically using e.g. MAX__in2D_aligned / (nr*nc),
  # or statically in a LUT using MAX__in2D_aligned at the end
  def block_input_channels(kernel_dims_str):
    return kernel_dims_str[0] == '1' and kernel_dims_str[1] == '1'

  # Given the max buffer size for an image, check if the entire image can fit inside a buffer
  # or if e.g. a Line Buffer is needed.
  # The image buffer size may be e.g. 4096 (in words) for the VU9P which has URAMs (4096 words deep).
  # The SRAM may be banked so fitting inputs in 1 Block RAM may not be the outcome, but this buffer
  # size also applies to the output partial sums.
  def get_max_img_side_size_for_single_buffer():
    import math
    return int(math.sqrt(device_params['image_buffer_size']))
  
  def use_line_buffer(data_dims_str):
    return int(data_dims_str[0]) > get_max_img_side_size_for_single_buffer()
  
  # Helper to conv_before_fusion
  # Note: Some of these optimizations, e.g. for VALID padding, make the generated code simpler but would not impact performance
  # because Spatial would e.g. optimize constant checks anyway. So it may not be worth keeping all the optimizations below,
  # since they make the function more complicated in order to make the generated code simpler, but performance is not impacted.
  def generate_sliding_window(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, half_kernel_size, B_par, \
    load_data_from_dram=True, weight_blocking=True, use_linebuf=False):
      
      conv_loop = ''
      
      # Find proper indentation      
      if not load_data_from_dram:
        indent = '    '
      elif not use_linebuf:
        indent = '      '   # One more loop since can't fuse in_channels loop
      else:
        indent = '        ' # One more loop for loading rows
      
      # Check if weights are loaded in blocks
      if weight_blocking:
        b_idx_end = ', b'
      else:
        b_idx_end = ''
      
      # Data is in DRAM so we could not fuse the in_channels loop with the rows/columns loop
      # Print the extra loop here
      if load_data_from_dram:
        conv_loop += '''
      ''' + indent
        if use_linebuf:
          conv_loop += '''
      ''' + indent + '''Pipe.II(1).Foreach(0 until ''' + out_compute_cols + ''', 0 until B par ''' + B_par + ''') { (c,b) =>'''
        else:
          conv_loop += '''
      ''' + indent + '''Pipe.II(1).Foreach(0 until ''' + out_compute_rows + ''', 0 until ''' + out_compute_cols + ''', 0 until B par ''' + B_par + ''') { (r,c,b) =>'''
      conv_loop += '''
      ''' + indent
      
      # Can use mux here instead of *s
      
      # For VALID padding, row_start, row_end, col_start, col_end, can be simplified
      # But this might not affect performance or util, only make the generated code simpler, so maybe not
      # worth making this function more complicated. If this is always set to True, correctness should not
      # change (it makes this function simpler but generated code will be longer than necessary for VALID padding)
      check_bounds = True
      if padding == 'VALID':
        check_bounds = False
      # Initialize data loading bounds
      if check_bounds:
        if use_linebuf:
          conv_loop += '''  
      ''' + indent + '''  val row_start = min((kr-1).to[Int], max(0.to[Int], (kr-1-r.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val row_end   = min((kr  ).to[Int], max(1.to[Int], (kr+nr-1-r.to[Int]*s.to[Int]).to[Int]) )'''
        else:
          conv_loop += '''
      ''' + indent + '''  val row_start = min((kr-1).to[Int], max(0.to[Int], (kr_ignore-r.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val row_end   = min((kr  ).to[Int], max(1.to[Int], (nr+kr_ignore-r.to[Int]*s.to[Int]).to[Int]) )'''
        conv_loop += '''
      ''' + indent + '''  val col_start = min((kc-1).to[Int], max(0.to[Int], (kc_ignore-c.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val col_end   = min((kc  ).to[Int], max(1.to[Int], (nc+kc_ignore-c.to[Int]*s.to[Int]).to[Int]) )'''
      # If line buffers are used, row_start check is still needed
      elif use_linebuf:
        conv_loop += '''  
      ''' + indent + '''  val row_start = max(0.to[Int], (kr-1-r.to[Int]*s.to[Int]   ).to[Int])'''

      # Print weight load
      conv_loop += '''
      ''' + indent + '''  
      ''' + indent + '''  val kernel: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''    ''' + weight_read_str + '''
      ''' + indent + '''  }}.flatten
      ''' + indent + '''  '''
      
      # Get data read string
      column_skip_str = ''
      if check_bounds:
        column_skip_str = '-kc_ignore'
      if not load_data_from_dram:
        data_idx_str = in_name + '_SRAM(inCh_i, i.to[Int]-kr_ignore+r.to[Int], j.to[Int]' + column_skip_str + '+c.to[Int])'
      elif use_linebuf:
        data_idx_str = in_name + '(kr-i,j.to[Int]' + column_skip_str + '+c.to[Int]*s.to[Int])'  # Can use mux here
      else:
        data_idx_str = in_name + '((i.to[Int]-kr_ignore+mux(s==1,r,r*2))*nc + j.to[Int]' + column_skip_str + '+mux(s==1,c,c*2))'
        
      # Specialize data read for different paddings
      if check_bounds:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''  
      ''' + indent + '''    if (i < ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start || j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i > ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end || j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    
      ''' + indent + '''    else if (i < ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''    }
      ''' + indent + '''    else if (i > ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    
      ''' + indent + '''    else if (i < ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start || j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else {// if (i > ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end || j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''  }}.flatten'''
      
      elif use_linebuf:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''    if (i < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else {
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''    }
      ''' + indent + '''  }}.flatten'''
      
      else:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''  }}.flatten'''
      
      # Print accumulation
      conv_loop += '''
      ''' + indent + '''  
      ''' + indent + '''  val partial_sum = ReduceTree(data.zip(kernel).map{case (data_i, kernel_i) => data_i * kernel_i} :_*){_+_}'''
      if use_linebuf:
        conv_loop += '''
      ''' + indent + '''  if (r >= kr_ignore) {
      ''' + indent + '''    ''' + out_name + '''_SRAM_conv(r.to[Int]-kr_ignore, c, b) = partial_sum + mux(inCh_i == 0, 0.to[T], ''' + out_name + '''_SRAM_conv(r.to[Int]-kr_ignore, c, b))
      ''' + indent + '''  }'''
      else:
        conv_loop += '''
      ''' + indent + '''  ''' + out_name + '''_SRAM_conv(r, c''' + b_idx_end + ''') = partial_sum + mux(inCh_i==0, 0.to[T], ''' + out_name + '''_SRAM_conv(r, c''' + b_idx_end + '''))'''
      
      # Close indents
      conv_loop += '''
      ''' + indent + '''}'''
      if load_data_from_dram:
        if use_linebuf:
          conv_loop += '''
            }'''
        conv_loop += '''
          }
          '''
      return conv_loop
  
  # Can merge w/ generate_sliding_window
  def generate_sliding_window_depthwise(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, half_kernel_size, c_par, \
    load_data_from_dram=True, weight_blocking=True, use_linebuf=False):
      
      # Note: using c_par instead of inB_par because inB can = 1
      
      conv_loop = ''
      
      # Find proper indentation      
      if not load_data_from_dram:
        indent = '  '
      elif not use_linebuf:
        indent = '    '   # One more loop since can't fuse in_channels loop
      else:
        indent = '      ' # One more loop for loading rows
      
      # Check if weights are loaded in blocks
      if weight_blocking:
        b_idx_end = ', ib'
      else:
        b_idx_end = ''
      
      # Data is in DRAM so we could not fuse the in_channels loop with the rows/columns loop
      # Print the extra loop here
      if load_data_from_dram:
        conv_loop += '''
      ''' + indent
        if use_linebuf:
          conv_loop += '''
      ''' + indent + '''Foreach(0 until ''' + out_compute_cols + ''' par ''' + c_par + ''', 0 until inB) { (c,ib) =>'''
        else:
          conv_loop += '''
      ''' + indent + '''Foreach(0 until ''' + out_compute_rows + ''', 0 until ''' + out_compute_cols + ''' par ''' + c_par + ''', 0 until inB) { (r,c,ib) =>'''
      conv_loop += '''
      ''' + indent
      
      # Can use mux here instead of *s
      
      # For VALID padding, row_start, row_end, col_start, col_end, can be simplified
      # But this might not affect performance or util, only make the generated code simpler, so maybe not
      # worth making this function more complicated. If this is always set to True, correctness should not
      # change (it makes this function simpler but generated code will be longer than necessary for VALID padding)
      check_bounds = True
      if padding == 'VALID':
        check_bounds = False
      # Initialize data loading bounds
      if check_bounds:
        if use_linebuf:
          conv_loop += '''  
      ''' + indent + '''  val row_start = min((kr-1).to[Int], max(0.to[Int], (kr-1-r.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val row_end   = min((kr  ).to[Int], max(1.to[Int], (kr+nr-1-r.to[Int]*s.to[Int]).to[Int]) )'''
        else:
          conv_loop += '''
      ''' + indent + '''  val row_start = min((kr-1).to[Int], max(0.to[Int], (kr_ignore-r.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val row_end   = min((kr  ).to[Int], max(1.to[Int], (nr+kr_ignore-r.to[Int]*s.to[Int]).to[Int]) )'''
        conv_loop += '''
      ''' + indent + '''  val col_start = min((kc-1).to[Int], max(0.to[Int], (kc_ignore-c.to[Int]*s.to[Int]   ).to[Int]) )
      ''' + indent + '''  val col_end   = min((kc  ).to[Int], max(1.to[Int], (nc+kc_ignore-c.to[Int]*s.to[Int]).to[Int]) )'''
      # If line buffers are used, row_start check is still needed
      elif use_linebuf:
        conv_loop += '''  
      ''' + indent + '''  val row_start = max(0.to[Int], (kr-1-r.to[Int]*s.to[Int]   ).to[Int])'''

      # Print weight load
      conv_loop += '''
      ''' + indent + '''  
      ''' + indent + '''  val kernel: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''    ''' + weight_read_str + '''
      ''' + indent + '''  }}.flatten
      ''' + indent + '''  '''
      
      # Get data read string
      column_skip_str = ''
      if check_bounds:
        column_skip_str = '-kc_ignore'
      if not load_data_from_dram:
        data_idx_str = in_name + '_SRAM(inCh_i, i.to[Int]-kr_ignore+r.to[Int], j.to[Int]' + column_skip_str + '+c.to[Int])'
      elif use_linebuf:
        data_idx_str = in_name + '(kr-i,j.to[Int]' + column_skip_str + '+c.to[Int]*s.to[Int])'  # Can use mux here
      else:
        data_idx_str = in_name + '((i.to[Int]-kr_ignore+mux(s==1,r,r*2))*nc + j.to[Int]' + column_skip_str + '+mux(s==1,c,c*2))'
        
      # Specialize data read for different paddings
      if check_bounds:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''  
      ''' + indent + '''    if (i < ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start || j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i > ''' + str(half_kernel_size) + ''' && j < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end || j < col_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    
      ''' + indent + '''    else if (i < ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''    }
      ''' + indent + '''    else if (i > ''' + str(half_kernel_size) + ''' && j == ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    
      ''' + indent + '''    else if (i < ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start || j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else if (i == ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else {// if (i > ''' + str(half_kernel_size) + ''' && j > ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i+1 > row_end || j+1 > col_end),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''  }}.flatten'''
      
      elif use_linebuf:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''    if (i < ''' + str(half_kernel_size) + ''') {
      ''' + indent + '''      mux((i < row_start),
      ''' + indent + '''        0.to[T],
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''      )
      ''' + indent + '''    }
      ''' + indent + '''    else {
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''    }
      ''' + indent + '''  }}.flatten'''
      
      else:
        conv_loop += '''
      ''' + indent + '''  val data: List[T] = List.tabulate(kr){i => List.tabulate(kc){j => 
      ''' + indent + '''        ''' + data_idx_str + '''
      ''' + indent + '''  }}.flatten'''
      
      # Print accumulation
      conv_loop += '''
      ''' + indent + '''  
      ''' + indent + '''  val window_sum = ReduceTree(data.zip(kernel).map{case (data_i, kernel_i) => data_i * kernel_i} :_*){_+_}'''
      if use_linebuf:
        conv_loop += '''
      ''' + indent + '''  if (r >= kr_ignore) {
      ''' + indent + '''    ''' + out_name + '''_SRAM_conv(r.to[Int]-kr_ignore, c, ib) = window_sum
      ''' + indent + '''  }'''
      else:
        conv_loop += '''
      ''' + indent + '''  ''' + out_name + '''_SRAM_conv(r, c''' + b_idx_end + ''') = window_sum'''
      
      # Close indents
      conv_loop += '''
          }
      ''' + indent
      if load_data_from_dram:
        if use_linebuf:
          conv_loop += '''
            }'''
      return conv_loop
  
  # Convolution node prior to other fused operations
  def conv_before_fusion(is_input, tmpvar, tmpvar_inputs, bias_tmpvar_inputs, bias_kernel_dims_str, \
    data_dims_str, kernel_dims_str, strides, padding, out_dims_str, final_out_size, make_reuse_def):
  
    # Initializations
    output_string = ''
    dse_string = ''
    layer_B  = 32
    layer_IP = 32
    layer_OP = 1
    layer_WLP = 1
    layer_DLP = 1
    layer_SP = 1
    
    # Blocking output channels (loading a block of kernels at once) is usually important to:
    # - reduce the number of times data is loaded from DRAM
    # - improve pipelining (add iterations to inner loop) / add a new banking dimension to parallelize
    #   (so no need to parallelize r/c, which may complicate banking)
    # But in cases e.g. where data stays in SRAM and improved pipelining / additional parallelism
    # is not needed (e.g. because rows/columns are large enough), then not weight_blocking can save SRAM.
    # In general though it may be best to keep this default True.
    weight_blocking = True
    # For cases of data small enough to fit in SRAM, we may not need to block output channels
    if not conv_input_output_from_DRAM:
      ops = int(kernel_dims_str[3])*int(out_dims_str[0])*int(out_dims_str[1])*int(kernel_dims_str[2])
      if ops > 256*1024:
        layer_B = 8
      elif ops > 32*1024:
        layer_B = 4
      else:
        # Additional parallelism not needed so B (additional banking dimension) can be ignored.
        # Blocking may still be useful to reduce pipeline overheads though, so can add one more
        # check here to see if rows/cols are small
        weight_blocking = False
    
    assert strides[1] == strides[2]
    stride = strides[1]
    
    if make_reuse_def:
      out_channels = 'out_channels'
      out_name = 'out'
      img2D_size = 'MAX__in2D_aligned'
      line_in_cols_sram = 'MAX__nc' #'MAX__nc_aligned'
      weight_name = 'weight'
      weight_dram_name = '{{{REUSE_NAME}}}_weights_concat'
      weight_load_dim = 'weights_start_idx + outCh_i :: weights_start_idx + outCh_i + B, 0::in_channels'
      memreduce_iter = 'in_channels'
      out_sram_cols = 'MAX__oc'
      out_sram_rows = 'MAX__or'
      out_compute_cols = 'oc'
      out_compute_rows = 'or'
      weight_sram_depth = 'MAX__in_channels'
      stride_str = 'stride'
    else:
      out_channels = kernel_dims_str[3]
      out_name = tmpvar
      img2D_size = 'in2D'
      line_in_cols_sram = 'nc' #str(burst_align(data_dims_str[1])) #'nc_aligned'    # May want to align
      weight_name = tmpvar_inputs[1]
      weight_dram_name = weight_name
      weight_load_dim = 'outCh_i :: outCh_i + B, 0::inCh'
      memreduce_iter = 'inCh'
      out_sram_cols = 'oc'
      out_sram_rows = 'or'
      out_compute_cols = 'oc'
      out_compute_rows = 'or'
      weight_sram_depth = 'inCh'
      stride_str = stride
    
    # Check if we are writing to DRAM or SRAM. If SRAM, print that here.
    if conv_input_output_from_DRAM:
      output_string += "\n" + '        val B = B_L' + str(global_val_number)
    else:
      if is_input:
        output_string += '''        val ''' + tmpvar_inputs[0] + '''_SRAM = SRAM[T](''' + ','.join(reformat_memory(data_dims_str)) + ''')
        ''' + tmpvar_inputs[0] + '''_SRAM load ''' + tmpvar_inputs[0] + '''_DRAM(0::''' + ', 0::'.join(reformat_memory(data_dims_str)) + ')' + "\n"
      output_string += "\n" + '        val ' + tmpvar + '_SRAM = SRAM[T](' + ','.join(reformat_memory(final_out_size.split(','))) + ')'
    
    # --------------------------
    # k = 1
    # --------------------------
    # If k == 1, then there is never any padding, so we can move that special case out here.
    # Also, this does not need to unroll any 2D convolution windows and the weight load is different since 2D and not 4D.
    special_case_k1 = int(kernel_dims_str[0]) == 1 and int(kernel_dims_str[1]) == 1
    if special_case_k1:
      # 1x1 is usually only used for larger DNNs so assume DRAM for now
      if not conv_input_output_from_DRAM:
        utils.error_exit('1x1 Convolution currently assumes input / output in DRAM')
              
      # # Line Buffer is not needed for 1x1, so instead here re-order loops to load a block of rows
      # if use_line_buffer(data_dims_str):
      #   # There are 2 ways of doing this, either storing the outputs in blocks of rows x cols x B,
      #   # or storing the outputs in blocks of rows x cols_block x B. Currently doing the first option 
      #   # below with smaller B. Can implement first option here as well.
      #   utils.error_exit('Larger buffer sizes for 1x1 Convolution not yet implemented')      
      
      # Currently, inB below is calculated statically and put into a LUT
      # (see top of Convolution operation processing below)
      output_string += '''
        val s = ''' + stride_str + '''
        val ofmap_block_store_size = mux(inB<B,inB,B.to[Int])
        Foreach(''' + out_channels + ''' by B) { outCh_i => // out channels
        '''
      
      layer_OP = '{{{REUSE_NAME}}}_OP'
      if use_line_buffer(data_dims_str):
        layer_DLP = 4
        layer_WLP = 2 # Can make this 1, and then instead of layer_WLP/2 below, do max(1, layer_WLP/2)
        layer_SP = 4
      else:
        layer_DLP = 16
        layer_WLP = 8
        layer_SP = 8
      
      if bias_tmpvar_inputs and bias_kernel_dims_str:
        if make_reuse_def:
          bias_name = 'bias'
          bias_dram_name = '{{{REUSE_NAME}}}_bias_concat'
          # bias_sram_dim = 'max_out_channels'
          bias_sram_dim = 'B'
          bias_load_dim = 'bias_start_idx + outCh_i.to[Int] :: bias_start_idx + outCh_i.to[Int] + B'
        else:
          bias_name = bias_tmpvar_inputs[1]
          bias_dram_name = bias_name
          # bias_sram_dim = ','.join(bias_kernel_dims_str)
          bias_sram_dim = 'B'
          bias_load_dim = '0::' + ','.join(bias_kernel_dims_str)
          # No in2D_aligned or load_offset needed
        output_string += '''
          val ''' + bias_name + '''_SRAM = SRAM[T](''' + bias_sram_dim + ''')
          ''' + bias_name + '''_SRAM load ''' + bias_dram_name + '''_DRAM(''' + bias_load_dim + ''')'''

      # Calculate size of SRAM and loads
      # This is based on image_buffer_size. Because of banking, making that larger should not
      # affect the number of RAMs used for in_buffer. However making it large may cause a lot
      # of RAMs to be used to store the partial sum results.
      #
      # Note: If there is a case where the largest 2D size is larger than image_buffer_size,
      # the case above (line buffer check) would have run instead.
      # Alternatively, can "increase" the buffer limit here for 1x1, i.e. skip above check and
      # instead change this load and SRAM size to:
      #   max( image_buffer_size, largest size )
      # The largest value is only available after traversing all nodes, so can add a traversal
      # above to find the largest size for 1x1 conv (easy to add).
      # However, it also exists as a constant in the code (in this script it is in the variable
      # img2D_size, which is MAX__in2D_aligned).
      in2D_size = int(data_dims_str[0])*int(data_dims_str[1])
      if use_line_buffer(data_dims_str):
        inChannel_block_size = 1 # can make this closest_pow_2(max_in2D_size / in2D_size)
        layer_B  = 4
        layer_IP = 4
      else:
        inChannel_block_size = closest_pow_2(device_params['image_buffer_size'] / in2D_size)
      load_size = inChannel_block_size*in2D_size
      
      if not make_reuse_def:
        output_string += '''
          val nr = ''' + data_dims_str[0] + '''
          val nc = ''' + data_dims_str[1] + '''
          val in2D = ''' + str(in2D_size) + '''
          val inB = ''' + str(inChannel_block_size) + '''
          val inCh = ''' + kernel_dims_str[2]
      output_string += '''
      
          val ''' + weight_name + '''_SRAM = SRAM[T](B, ''' + weight_sram_depth + ''').flat.noduplicate
          ''' + weight_name + '''_SRAM load ''' + weight_dram_name + '''_DRAM(''' + weight_load_dim + ''' par WLP_L''' + str(global_val_number) + ''')

          val ''' + weight_name + '''_SRAM_reshape = SRAM[T](B, MAX__in_channels).flat.noduplicate
          Foreach(in_channels by 1, B by 1 par WLP_L''' + str(global_val_number) + '''/2) { (inCh_i, b) =>
            ''' + weight_name + '''_SRAM_reshape(b, inCh_i) = ''' + weight_name + '''_SRAM(b, inCh_i)
          }

          val ''' + out_name + '''_SRAM_conv = SRAM[T](''' + out_sram_rows + ', ' + out_sram_cols + ''', B).buffer
          Foreach(''' + memreduce_iter + ''' by inB) { inCh_i => // in channels
            val in_buffer = SRAM[T](''' + str(load_size) + ''')
            val start_idx = inCh_i*in2D'''      
      if make_reuse_def:
        output_string += '''
            in_buffer load tmp_DRAM(load_idx_0, start_idx :: start_idx + ''' + str(load_size) + ''' par DLP_L''' + str(global_val_number) + ''')'''
      else:
        output_string += '''
            in_buffer load ''' + tmpvar_inputs[0] + '''_DRAM(start_idx :: start_idx + ''' + str(load_size) + ''' par DLP_L''' + str(global_val_number) + ''')'''
      output_string += '''
            
            Pipe.II(1).Foreach(0::inB par 1, 0 until ''' + out_compute_cols + ''' par OP_L''' + str(global_val_number) + ''', 0 until ''' + out_compute_rows + ''', 0 until B par IP_L''' + str(global_val_number) + ''') { case List(ib,c,r,b) =>
              val loaded_val = in_buffer(ib*in2D + mux(s==1,r,r*2)*nc + mux(s==1,c,c*2))
              val partial_sum = loaded_val * ''' + weight_name + '''_SRAM_reshape(b, inCh_i.to[Int] + ib.to[Int])              
              val start_from_0 = (inCh_i == 0) && (ib == 0)
              ''' + out_name + '''_SRAM_conv(r,c,b) = partial_sum + mux(start_from_0, 0.to[T], ''' + out_name + '''_SRAM_conv(r,c,b))
            }
          }
          '''
            
    # End of special-cases unrelated to padding
    
    # There are 3 types of paddings:
    # - valid
    # - same
    # - custom
    # Currently, I have 2 code paths, one for valid one for same/custom, because valid allows simplifications.
    # But I could merge all code paths into 1. Valid simplifies the code over Same/Custom but const folding in 
    # compiler can do that simplification so can merge all three into one code path.
    
    # --------------------------
    # VALID padding
    # --------------------------
    # Note: this is similar to SAME with some simplifications, so can refactor
    elif padding == 'VALID':
      
      assert not make_reuse_def # Can later handle reuse
      
      # Could also move B to final dim
      # Now we have it in different places for different SRAMs arbitrarily so we need more variables here
      # Also can get rid of B_str vs. B_str_global, and do like SAME which does not need these
      b_idx = ''
      b_idx_end = ''
      B_sram_dim = ''
      B_sram_dim_end = ''
      B_str_global = '1'
      B_str = '1'
      if weight_blocking:
        b_idx = 'b, '
        b_idx_end = ', b'
        B_str_global = 'B_L' + str(global_val_number)
        B_str = 'B'
        B_sram_dim = 'B, '
        B_sram_dim_end = ', B'
      
      if bias_tmpvar_inputs and bias_kernel_dims_str:
        bias_name = bias_tmpvar_inputs[1]
        bias_dram_name = bias_name
        bias_sram_dim = bias_kernel_dims_str[0]
        bias_load_dim = '0::' + ','.join(bias_kernel_dims_str)
        output_string += '''
        val ''' + bias_name + '''_SRAM = SRAM[T](''' + bias_sram_dim + ''')
        ''' + bias_name + '''_SRAM load ''' + bias_dram_name + '''_DRAM(''' + bias_load_dim + ''')'''
      
      output_string += '''
        Foreach(''' + out_channels + ''' by ''' + B_str_global + ''') { outCh_i => // out channels
          val nr = ''' + data_dims_str[0] + '''
          val nc = ''' + data_dims_str[1] + '''
          val kr = ''' + kernel_dims_str[0] + '''
          val kc = ''' + kernel_dims_str[1] + '''
          val or = ''' + out_dims_str[0] + '''
          val oc = ''' + out_dims_str[1] + '''
          val inCh = ''' + kernel_dims_str[2]
      if weight_blocking and not conv_input_output_from_DRAM:
        output_string += '''
          val B = ''' + B_str_global
      output_string += '''
          val ''' + out_name + '''_SRAM_conv = SRAM[T](or, oc''' + B_sram_dim_end + ''')'''
    
      # Check if we need to load from DRAM. If so, do like SAME padding below:
      # use a line buffer for large images, otherwise load each 2D fmap from DRAM
      if conv_input_output_from_DRAM:
      
        import math
        if use_line_buffer(data_dims_str):
          kr_ignore = str( int( math.ceil( float(int(kernel_dims_str[0]))/float(stride) ) - 1 ) )
        else:
          kr_ignore = '0'
        half_kernel_size = int(int(kernel_dims_str[0])/2)
      
        output_string += '''
          val kr_ignore = ''' + kr_ignore + '''
          val s = ''' + stride_str
      
        weights_small_B_no_reshape = skip_weight_reshape(kernel_dims_str)
        
        if weights_small_B_no_reshape:
          output_string += '''
        
          // Outside loop since small inCh. Can also move inside loop, e.g. SRAM(B, 1, kr*kc).  
          // Can also load inCh*kr*kc*B (single 1D load) since B small as well.
          val ''' + weight_name + '''_SRAM = SRAM[T](B, inCh, kr*kc).hierarchical.noduplicate
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i::outCh_i.to[Int] + B, 0::inCh, 0::kr*kc)
        '''

        output_string += '''
          Foreach(''' + memreduce_iter + ''' by 1) { inCh_i => // in channels'''
      
        if not weights_small_B_no_reshape:
          layer_WLP = 8
          layer_DLP = 4
          output_string += '''
            
            val ''' + weight_name + '''_SRAM = SRAM[T](B*kr*kc).flat.noduplicate'''
          if make_reuse_def:
            output_string += '''
            ''' + weight_name + '''_SRAM load ''' + weight_dram_name + '''_DRAM(weights_start_idx + inCh_i.to[Int], kr*kc*outCh_i.to[Int]::kr*kc*(outCh_i.to[Int]+B) par WLP_L''' + str(global_val_number) + ''')'''
          else:
            output_string += '''
            ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(inCh_i.to[Int], kr*kc*outCh_i.to[Int]::kr*kc*(outCh_i.to[Int]+B) par WLP_L''' + str(global_val_number) + ''')'''
          output_string += '''
            val ''' + weight_name + '''_SRAM_reshape = SRAM[T](B,kr*kc).hierarchical.noduplicate
            Foreach(kr*kc by 1, B by 1 par WLP_L''' + str(global_val_number) + '''/2) { (ij, b) =>
              ''' + weight_name + '''_SRAM_reshape(b, ij) = ''' + weight_name + '''_SRAM(b*kr*kc + ij)
            }'''
      
        # This is a special-case of SAME padding and the code is similar so can re-use code path from below
      
        assert int(stride) in [1,2]
        
        # For larger images, use line buffer
        if use_line_buffer(data_dims_str):
          
          assert not make_reuse_def # Can later handle reuse
          
          layer_B  = 2
          layer_IP = 1
          layer_DLP = 2
          
          import math
          total_number_loaded_px = int( math.ceil(float(kernel_dims_str[0])/float(stride)) ) * int(stride)
          extra_loaded_px = total_number_loaded_px - int(kernel_dims_str[0])
          
          if int(stride) == 1:
            output_string += '''
            val lb''' + str(global_val_number) + ''' = LineBuffer[T](kr, ''' + line_in_cols_sram  + ''')'''
          else:
            output_string += '''
            val lb''' + str(global_val_number) + ''' = LineBuffer.strided[T](kr+''' + str(extra_loaded_px) + ''', ''' + line_in_cols_sram + ''', s)'''
          output_string += '''
            Foreach(0 until or + kr_ignore) { r =>
              val row_to_load_from = min(r.to[Int]*s.to[Int], nr.to[Int]-s.to[Int])'''  # Can use mux here
          if int(stride) == 1:
            output_string += '''
              lb''' + str(global_val_number) + ''' load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i, row_to_load_from, 0::nc par DLP_L''' + str(global_val_number) + ''')'''
          else:
            output_string += '''
              lb''' + str(global_val_number) + ''' load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i, row_to_load_from::row_to_load_from+s, 0::nc par DLP_L''' + str(global_val_number) + ''')'''
          in_name = 'lb' + str(global_val_number)
          if weights_small_B_no_reshape:
            weight_read_str = weight_name + '_SRAM(b, inCh_i, kc*i.to[Int] + j.to[Int])'
          else:
            weight_read_str = weight_name + '_SRAM_reshape(b, kc*i.to[Int] + j.to[Int])'
          output_string += generate_sliding_window(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
            half_kernel_size, '1', True, True, True)
        
        # Otherwise, load entire 2D
        else:
          layer_OP = '{{{REUSE_NAME}}}_OP'
          output_string += '''
            val img2D = SRAM[T](''' + img2D_size + ''')'''
          if make_reuse_def:
            output_string += '''
            img2D load tmp_DRAM(load_idx_0, inCh_i*in2D :: inCh_i*in2D + in2D par DLP_L''' + str(global_val_number) + ''')'''
          else:
            output_string += '''
            img2D load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i*in2D :: inCh_i*in2D + in2D par DLP_L''' + str(global_val_number) + ''')'''
          B_par = 'IP_L' + str(global_val_number)
          weight_read_str = weight_name + '_SRAM_reshape(b, kc*i.to[Int] + j.to[Int])'
          output_string += generate_sliding_window('img2D', out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
            half_kernel_size, B_par)
      # Load from SRAM
      else:
        output_string += '''
          val ''' + weight_name + '''_SRAM = SRAM[T](''' + B_sram_dim + '''inCh*kr*kc).flat.noduplicate'''
        if weight_blocking:
          output_string += '''
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i :: outCh_i + ''' + B_str + ''', 0::inCh*kr*kc par WLP_L''' + str(global_val_number) + ''')'''
        else:
          output_string += '''
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i, 0::inCh*kr*kc par WLP_L''' + str(global_val_number) + ''')'''
        # If inCh=1, no partial sum accumulation is needed and can eliminate inCh loop
        if int(kernel_dims_str[2]) == 1:
          layer_WLP = 1
          if weight_blocking:
            output_string += '''
          Foreach(0 until or, 0 until oc, 0 until ''' + B_str + ''' par ''' + B_str + ''') { (r,c,b) =>'''
          else:
            output_string += '''
          Foreach(0 until or, 0 until oc) { (r,c) =>'''
          output_string += '''
            val kernel = List.tabulate(kr){i => List.tabulate(kc){j => 
              ''' + tmpvar_inputs[1] + '''_SRAM(''' + b_idx + '''i.to[Int]*kc + j.to[Int])
            }}.flatten
            val pixels = List.tabulate(kr){i => List.tabulate(kc){j => 
              ''' + tmpvar_inputs[0] + '''_SRAM(0.to[Int], r.to[Int]+i.to[Int],c.to[Int]+j.to[Int])
            }}.flatten
            ''' + tmpvar + '''_SRAM_conv(r, c) = ReduceTree(pixels.zip(kernel).map{case (data_i, kernel_i) => data_i * kernel_i} :_*){_+_}
          }'''
        else:
          assert not skip_weight_reshape(kernel_dims_str)
          layer_WLP = 4
          output_string += '''
          val ''' + weight_name + '''_SRAM_reshape = SRAM[T](''' + B_sram_dim + '''inCh,kr,kc).hierarchical.noduplicate'''
          if weight_blocking:
            output_string += '''
          Foreach(0 until inCh, 0 until kr, 0 until kc, 0 until ''' + B_str + ''' par WLP_L''' + str(global_val_number) + ''') { case List(inCh_i,ki,kj,b) =>
            ''' + weight_name + '''_SRAM_reshape(''' + b_idx + '''inCh_i, ki, kj) = ''' + weight_name + '''_SRAM(''' + b_idx + '''inCh_i*kr*kc + ki*kc + kj)
          }
          Pipe.II(1).Foreach(inCh by 1, 0 until or, 0 until oc, 0 until ''' + B_str + ''' par ''' + B_str + ''') { case List(inCh_i,r,c,b) => // in channels'''
          else:
            output_string += '''
          Foreach(0 until inCh par WLP_L''' + str(global_val_number) + ''', 0 until kr, 0 until kc par 1) { (inCh_i,ki,kj) =>
            ''' + weight_name + '''_SRAM_reshape(''' + B_sram_dim + '''inCh_i,ki,kj) = ''' + weight_name + '''_SRAM(inCh_i*kr*kc + ki*kc + kj)
          }
          Pipe.II(1).Foreach(inCh by 1, 0 until or, 0 until oc) { case List(inCh_i, r, c) => // in channels'''
              
          # Can refactor below to use generate_sliding_window.
          # First have to add a case in it to ignore the += partial_sum for inCh=1
          """
          in_name = tmpvar_inputs[0]
          weight_read_str = weight_name + '_SRAM_reshape(' + b_idx + 'inCh_i, i, j)'        
          output_string += generate_sliding_window(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
            half_kernel_size, 'B', False, weight_blocking)
          """
          
          output_string += '''
            val kernel = List.tabulate(kr){i => List.tabulate(kc){j => 
              ''' + tmpvar_inputs[1] + '''_SRAM_reshape(''' + b_idx + '''inCh_i, i, j)
            }}.flatten
            val pixels = List.tabulate(kr){i => List.tabulate(kc){j => 
              ''' + tmpvar_inputs[0] + '''_SRAM(inCh_i, r.to[Int]+i.to[Int],c.to[Int]+j.to[Int])
            }}.flatten
            val partial_sum = ReduceTree(pixels.zip(kernel).map{case (data_i, kernel_i) => data_i * kernel_i} :_*){_+_}
            val start_from_0 = (inCh_i == 0)
            ''' + tmpvar + '''_SRAM_conv(r, c''' + b_idx_end + ''') = partial_sum + mux(start_from_0, 0.to[T], ''' + tmpvar + '''_SRAM_conv(r, c''' + b_idx_end + '''))
          }'''
        
    # --------------------------
    # SAME or Custom padding
    # --------------------------
    else:
      if padding == 'SAME':
        pad_top, pad_left, pad_bottom, pad_right = get_same_padding(int(data_dims_str[0]),
          int(data_dims_str[1]), strides, int(kernel_dims_str[0]), int(kernel_dims_str[1]))
      else:
        p = int(padding)
        pad_top = p
        pad_left = p
        pad_bottom = p
        pad_right = p
        
      # Given the amount to pad, the amount of iterations to skip is
      # kernel size minus this, minus 1 (minus 1 since on the final iteration
      # we get the last piece of data so we don't need to skip that iteration).
      # Then finally, divide by stride (since we load s at a time).
      if use_line_buffer(data_dims_str):
        kr_ignore = str( int(int(kernel_dims_str[0]) - 1 - pad_top )/int(stride) )
        # The equation is ceil((k-p)/s) - 1, as a sanity check compute it the other way:
        import math
        kr_ignore_check = str( int( math.ceil( float(int(kernel_dims_str[0]) - pad_top )/float(stride) ) - 1 ))
        assert kr_ignore_check == kr_ignore
      else:
        kr_ignore = str( pad_top ) #'((kr - s)/2).to[Int]'
      kc_ignore = str( pad_left ) #'((kc - s)/2).to[Int]'
      half_kernel_size = int(int(kernel_dims_str[0])/2)
      # print kernel_dims_str[0] + ':  kr_ignore = ' + kr_ignore + ', kc_ignore = ' + kc_ignore
      
      if bias_tmpvar_inputs and bias_kernel_dims_str:
        if make_reuse_def:
          bias_name = 'bias'
          bias_dram_name = '{{{REUSE_NAME}}}_bias_concat'
          bias_sram_dim = 'MAX__out_channels'
          bias_load_dim = 'bias_start_idx :: bias_start_idx + out_channels'
        else:
          bias_name = bias_tmpvar_inputs[1]
          bias_dram_name = bias_name
          bias_sram_dim = bias_kernel_dims_str[0]
          bias_load_dim = '0::' + ','.join(bias_kernel_dims_str)
          # No in2D_aligned or load_offset needed
        output_string += '''
        val ''' + bias_name + '''_SRAM = SRAM[T](''' + bias_sram_dim + ''')
        ''' + bias_name + '''_SRAM load ''' + bias_dram_name + '''_DRAM(''' + bias_load_dim + ''')'''
      
      if conv_input_output_from_DRAM:
        output_string += '''
        Foreach(''' + out_channels + ''' by B) { outCh_i => // out channels
        '''
      else:
        if weight_blocking:
          output_string += '''
        Foreach(''' + out_channels + ''' by B_L''' + str(global_val_number) + ''') { outCh_i => // out channels
        
          val B = B_L''' + str(global_val_number)
        else:
          output_string += '''
        Foreach(''' + out_channels + ''' by 1) { outCh_i => // out channels
        '''
      if not make_reuse_def:
        output_string += '''
          val nr = ''' + data_dims_str[0] + '''
          val nc = ''' + data_dims_str[1] + '''
          val or = ''' + out_dims_str[0] + '''
          val oc = ''' + out_dims_str[1] + '''
          val inCh = ''' + kernel_dims_str[2]
      output_string += '''
          val kr = ''' + kernel_dims_str[0] + '''
          val kc = ''' + kernel_dims_str[1] + '''
          val kr_ignore = ''' + kr_ignore + '''
          val kc_ignore = ''' + kc_ignore + '''
          val s = ''' + stride_str
           # If no Pad(), should be (k-s)/2, and same for kr_ignore if no LB
      if conv_input_output_from_DRAM:
        output_string += '''
        
          val ''' + out_name + '''_SRAM_conv = SRAM[T](''' + out_sram_rows + ''', ''' + out_sram_cols  + ''', B)'''
      else:
        if weight_blocking:
          output_string += '''
        
          val ''' + out_name + '''_SRAM_conv = SRAM[T](''' + out_sram_rows + ''', ''' + out_sram_cols  + ''', B)'''
        else:
          output_string += '''
        
          val ''' + out_name + '''_SRAM_conv = SRAM[T](''' + out_sram_rows + ''', ''' + out_sram_cols  + ''')'''
      
      # If small W then no need to par load and if no B then no need to reshape
      weights_small_B_no_reshape = skip_weight_reshape(kernel_dims_str)
      
      # Normally 4D tensor weights are loaded one in channel at a time (inside in_channels loop), B*k*k.
      # For small # channels or if no weight blocking needed (since all data on-chip), can load all channels at once.
      
      # Case: On-chip data, small or no B
      if not conv_input_output_from_DRAM:
        if weight_blocking:
          output_string += '''
        
          val ''' + weight_name + '''_SRAM = SRAM[T](B, inCh*kr*kc)
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i :: outCh_i + B, 0::inCh*kr*kc par WLP_L''' + str(global_val_number) + ''')
        '''
        else:
          output_string += '''
        
          val ''' + weight_name + '''_SRAM = SRAM[T](inCh*kr*kc)
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i, 0::inCh*kr*kc par WLP_L''' + str(global_val_number) + ''')
        '''
        
        if not weights_small_B_no_reshape:
          if weight_blocking:
            output_string += '''
          val ''' + weight_name + '''_SRAM_reshape = SRAM[T](B,inCh,kr,kc).hierarchical.noduplicate
          Foreach(0 until inCh, 0 until kr, 0 until kc, 0 until B par WLP_L''' + str(global_val_number) + ''') { case List(inCh_i,ki,kj,b) =>
            ''' + weight_name + '''_SRAM_reshape(b,inCh_i,ki,kj) = ''' + weight_name + '''_SRAM(b,inCh_i*kr*kc + ki*kc + kj)
          }'''
          else:
            output_string += '''
          val ''' + weight_name + '''_SRAM_reshape = SRAM[T](inCh,kr,kc).hierarchical.noduplicate
          Foreach(0 until inCh par WLP_L''' + str(global_val_number) + ''', 0 until kr, 0 until kc) { (inCh_i,ki,kj) =>
            ''' + weight_name + '''_SRAM_reshape(inCh_i,ki,kj) = ''' + weight_name + '''_SRAM(inCh_i*kr*kc + ki*kc + kj)
          }'''
        
      # Case: Off-chip data, small B
      elif weights_small_B_no_reshape:
        output_string += '''
        
          // Outside loop since small inCh. Can also move inside loop, e.g. SRAM(B, 1, kr*kc).  
          // Can also load inCh*kr*kc*B (single 1D load) since B small as well.
          val ''' + weight_name + '''_SRAM = SRAM[T](B, inCh, kr*kc).hierarchical.noduplicate
          ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(outCh_i::outCh_i.to[Int] + B, 0::inCh, 0::kr*kc)
        '''
      
      # Otherwise, load 4D tensor weights one in channel at a time
      
      if conv_input_output_from_DRAM:
        output_string += '''
          Foreach(''' + memreduce_iter + ''' by 1) { inCh_i => // in channels'''
      else:
        if weight_blocking:
          output_string += '''
          Pipe.II(1).Foreach(''' + memreduce_iter + ''' by 1, 0 until or, 0 until oc, 0 until B par B) { case List(inCh_i,r,c,b) =>'''
        else:
          output_string += '''
          Pipe.II(1).Foreach(''' + memreduce_iter + ''' by 1, 0 until or, 0 until oc) { (inCh_i,r,c) =>'''
      
      # If data is on-chip, no need to re-load from DRAM so fuse the in channels loop with r/c/b and
      # keep the weight load for all in channels outside
      if conv_input_output_from_DRAM and not weights_small_B_no_reshape:
        layer_WLP = 8
        layer_DLP = 4
        output_string += '''
          
            val ''' + weight_name + '''_SRAM = SRAM[T](B*kr*kc).flat.noduplicate'''
        if make_reuse_def:
          output_string += '''
            ''' + weight_name + '''_SRAM load ''' + weight_dram_name + '''_DRAM(weights_start_idx + inCh_i.to[Int], kr*kc*outCh_i.to[Int]::kr*kc*(outCh_i.to[Int]+B) par WLP_L''' + str(global_val_number) + ''')'''
        else:
          output_string += '''
            ''' + weight_name + '''_SRAM load ''' + weight_dram_name + '''_DRAM(inCh_i.to[Int], kr*kc*outCh_i.to[Int]::kr*kc*(outCh_i.to[Int]+B) par WLP_L''' + str(global_val_number) + ''')'''
        output_string += '''
            val ''' + weight_name + '''_SRAM_reshape = SRAM[T](B,kr*kc).hierarchical.noduplicate
            Foreach(kr*kc by 1, B by 1 par WLP_L''' + str(global_val_number) + '''/2) { (ij, b) =>
              ''' + weight_name + '''_SRAM_reshape(b, ij) = ''' + weight_name + '''_SRAM(b*kr*kc + ij)
            }'''
    
      # Check if we need to load from DRAM. If so, use a line buffer or load single channels.
      if conv_input_output_from_DRAM:
      
        assert int(stride) in [1,2]
        
        # For larger images, use line buffer
        if use_line_buffer(data_dims_str):
          
          assert not make_reuse_def # Can later handle reuse
          
          layer_B  = 2
          layer_IP = 1
          layer_DLP = 2
          
          import math
          total_number_loaded_px = int( math.ceil(float(kernel_dims_str[0])/float(stride)) ) * int(stride)
          extra_loaded_px = total_number_loaded_px - int(kernel_dims_str[0])
          
          if int(stride) == 1:
            output_string += '''
            val lb''' + str(global_val_number) + ''' = LineBuffer[T](kr, ''' + line_in_cols_sram  + ''')'''
          else:
            output_string += '''
            val lb''' + str(global_val_number) + ''' = LineBuffer.strided[T](kr+''' + str(extra_loaded_px) + ''', ''' + line_in_cols_sram  + ''', s)'''
          output_string += '''
            Foreach(0 until nr/s + kr_ignore) { r =>
              val row_to_load_from = min(r.to[Int]*s.to[Int], nr.to[Int]-s.to[Int])'''  # Can use mux here
          if int(stride) == 1:
            output_string += '''
              lb''' + str(global_val_number) + ''' load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i, row_to_load_from, 0::nc par DLP_L''' + str(global_val_number) + ''')'''
          else:
            output_string += '''
              lb''' + str(global_val_number) + ''' load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i, row_to_load_from::row_to_load_from+s, 0::nc par DLP_L''' + str(global_val_number) + ''')'''
          in_name = 'lb' + str(global_val_number)
          if weights_small_B_no_reshape:
            weight_read_str = weight_name + '_SRAM(b, inCh_i, kc*i.to[Int] + j.to[Int])'
          else:
            weight_read_str = weight_name + '_SRAM_reshape(b, kc*i.to[Int] + j.to[Int])'
          output_string += generate_sliding_window(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
            half_kernel_size, '1', True, True, True)
        # Otherwise, load entire 2D
        else:
          layer_OP = '{{{REUSE_NAME}}}_OP'
          output_string += '''
            val img2D = SRAM[T](''' + img2D_size + ''')'''
          if make_reuse_def:
            output_string += '''
            img2D load tmp_DRAM(load_idx_0, inCh_i*in2D :: inCh_i*in2D + in2D par DLP_L''' + str(global_val_number) + ''')'''
          else:
            output_string += '''
            img2D load ''' + tmpvar_inputs[0] + '''_DRAM(inCh_i*in2D :: inCh_i*in2D + in2D par DLP_L''' + str(global_val_number) + ''')'''
          B_par = 'IP_L' + str(global_val_number)
          weight_read_str = weight_name + '_SRAM_reshape(b, kc*i.to[Int] + j.to[Int])'
          output_string += generate_sliding_window('img2D', out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
            half_kernel_size, B_par)
      # Load from SRAM
      else:
        
        b_idx = ''
        b_idx_end = ''
        if weight_blocking:
          b_idx = 'b,'
          b_idx_end = ', b'
        
        layer_IP = 1
        layer_OP = 4
        layer_WLP = 4

        in_name = tmpvar_inputs[0]
        if weights_small_B_no_reshape:
          weight_read_str = weight_name + '_SRAM(' + b_idx + 'i.to[Int]*kc + j.to[Int])'
        else:
          weight_read_str = weight_name + '_SRAM_reshape(' + b_idx + 'inCh_i, i, j)'        
        output_string += generate_sliding_window(in_name, out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
          half_kernel_size, 'B', False, weight_blocking)

    if make_reuse_def:
      dse_string += '  // {{{REUSE_NAME}}}' + "\n"
    else:
      dse_string += '  // Layer ' + str(global_val_number) + "\n"
    if weight_blocking:
      dse_string += '  val B_L' + str(global_val_number) + '   = ' + str(layer_B ) + "\n"
    dse_string += '  val WLP_L' + str(global_val_number) + ' = ' + str(layer_WLP) + "\n"
    if conv_input_output_from_DRAM:
      dse_string += '  val IP_L' + str(global_val_number) + '  = ' + str(layer_IP) + "\n"
      dse_string += '  val OP_L' + str(global_val_number) + '  = ' + str(layer_OP) + "\n"
      dse_string += '  val DLP_L' + str(global_val_number) + ' = ' + str(layer_DLP) + "\n"
      dse_string += '  val SP_L' + str(global_val_number) + '  = ' + str(layer_SP) + "\n"
    dse_string += "\n"

    return output_string, dse_string, weight_blocking, layer_IP
  
  # Convolution node prior to other fused operations
  # Note: Can merge w/ conv_before_fusion
  def conv_before_fusion_depthwise(is_input, tmpvar, tmpvar_inputs, bias_tmpvar_inputs, bias_kernel_dims_str, \
    data_dims_str, kernel_dims_str, strides, padding, out_dims_str, final_out_size, make_reuse_def):
  
    # Initializations
    output_string = ''
    dse_string = ''
    layer_IP = 1
    layer_OP = 1
    layer_WLP = 1
    # layer_DLP = 1
    # layer_SP = 1
    
    assert strides[1] == strides[2]
    stride = strides[1]
    
    # Depthwise conv is currently implemented for the most common cases
    # (SAME padding, k>1, inputs/outputs to DRAM). To handle all cases,
    # i.e. if the assertions below fail, then merge this def with 
    # conv_before_fusion, since then they would be similar.
    
    assert make_reuse_def
    if make_reuse_def:
      # out_channels = 'out_channels'   # Currently assumed to be same as in_channels
      out_name = 'out'
      img2D_size = 'MAX__in2D_aligned'
      line_in_cols_sram = 'MAX__nc' #'MAX__nc_aligned'
      weight_name = 'weight'
      weight_dram_name = '{{{REUSE_NAME}}}_weights_concat'
      memreduce_iter = 'in_channels'
      out_sram_cols = 'MAX__oc'
      out_sram_rows = 'MAX__or'
      out_compute_cols = 'oc'
      out_compute_rows = 'or'
      weight_sram_depth = 'MAX__in_channels'
      stride_str = 'stride'
    
    assert conv_input_output_from_DRAM
    
    special_case_k1 = int(kernel_dims_str[0]) == 1 and int(kernel_dims_str[1]) == 1
    assert not special_case_k1

    assert padding == 'SAME'
    if padding == 'SAME':
      pad_top, pad_left, pad_bottom, pad_right = get_same_padding(int(data_dims_str[0]),
        int(data_dims_str[1]), strides, int(kernel_dims_str[0]), int(kernel_dims_str[1]))
      # Currently no Line Buffers for dw conv because there are no output channels so
      # partial sums are not stored in large blocks.
      kr_ignore = '((kr - s)/2).to[Int]' #str( pad_top )
      kc_ignore = '((kc - s)/2).to[Int]' #str( pad_left )
      half_kernel_size = int(int(kernel_dims_str[0])/2)
      
      if bias_tmpvar_inputs and bias_kernel_dims_str:
        if make_reuse_def:
          bias_name = 'bias'
          bias_dram_name = '{{{REUSE_NAME}}}_bias_concat'
          bias_sram_dim = 'MAX__in_channels'
          bias_load_dim = 'bias_start_idx :: bias_start_idx + in_channels'
        output_string += '''
        val ''' + bias_name + '''_SRAM = SRAM[T](''' + bias_sram_dim + ''')
        ''' + bias_name + '''_SRAM load ''' + bias_dram_name + '''_DRAM(''' + bias_load_dim + ''')'''
      
      output_string += '''
        val kr = ''' + kernel_dims_str[0] + '''
        val kc = ''' + kernel_dims_str[1] + '''
        val s = ''' + stride_str + '''
        val kr_ignore = ''' + kr_ignore + '''
        val kc_ignore = ''' + kc_ignore # Should be (k-s)/2, and same for kr_ignore if no LB
      
      if conv_input_output_from_DRAM:
        output_string += '''
        Foreach(''' + memreduce_iter + ''' by inB par OP_L''' + str(global_val_number) + ''') { inCh_i => // in channels'''
      
      if conv_input_output_from_DRAM:
        output_string += '''
        
          val ''' + out_name + '''_SRAM_conv = SRAM[T](''' + out_sram_rows + ''', ''' + out_sram_cols  + ''', inB)'''
      
      # Setting inB = 1 for depthwise, can increase to load larger blocks of channels / input data
      
      # Note: Normally for kxk convolution, weights are stored (in channels, out channels * k * k).
      # But because out channels = 1 for depthwise, might want to move in_channels in with k*k.
      
      # Currently inB=1 so loading 1 in channel at a time, but for few in_channels can load all at once,
      # or can increase inB
      
      #output_string += '''
      #    val ''' + weight_name + '''_SRAM = SRAM[T](inB, kr*kc).hierarchical.noduplicate
      #    ''' + weight_name + '''_SRAM load ''' + weight_name + '''_DRAM(inCh_i :: inCh_i + inB, 0::kr*kc)
      #    '''

      if conv_input_output_from_DRAM:# and not weights_small_B_no_reshape:
        layer_WLP = 1
        # layer_DLP = 4
        output_string += '''
          
          val ''' + weight_name + '''_SRAM = SRAM[T](inB, kr*kc).hierarchical.noduplicate'''
        if make_reuse_def:
          output_string += '''
          ''' + weight_name + '''_SRAM load ''' + weight_dram_name + \
          '''_DRAM(weights_start_idx + inCh_i :: weights_start_idx + inCh_i + inB, 0::kr*kc par WLP_L''' + \
          str(global_val_number) + ''')'''
        
        #output_string += '''
        #  
        #  val ''' + weight_name + '''_SRAM = SRAM[T](inB*kr*kc).flat.noduplicate'''
        #if make_reuse_def:
        #  output_string += '''
        #  ''' + weight_name + '''_SRAM load ''' + weight_dram_name + '''_DRAM(weights_start_idx, kr*kc*inCh_i.to[Int]::kr*kc*(inCh_i.to[Int]+inB) par WLP_L''' + str(global_val_number) + ''')'''
        #output_string += '''
        #  val ''' + weight_name + '''_SRAM_reshape = SRAM[T](inB,kr*kc).hierarchical.noduplicate
        #  Foreach(kr*kc by 1, inB by 1 par WLP_L''' + str(global_val_number) + '''/2) { (ij, ib) =>
        #    ''' + weight_name + '''_SRAM_reshape(ib, ij) = ''' + weight_name + '''_SRAM(ib*kr*kc + ij)
        #  }'''
    
      assert int(stride) in [1,2]
      layer_OP = '{{{REUSE_NAME}}}_OP'
      output_string += '''
          val img2D = SRAM[T](''' + img2D_size + ''')'''
      if make_reuse_def:
        output_string += '''
          img2D load tmp_DRAM(load_idx_0, inCh_i*in2D :: inCh_i*in2D + inB*in2D par DLP_L''' + str(global_val_number) + ''')'''
      c_par = 'IP_L' + str(global_val_number)
      # weight_read_str = weight_name + '_SRAM_reshape(ib, kc*i.to[Int] + j.to[Int])'
      weight_read_str = weight_name + '_SRAM(ib, kc*i.to[Int] + j.to[Int])'
      output_string += generate_sliding_window_depthwise('img2D', out_name, padding, out_compute_rows, out_compute_cols, weight_read_str, \
        half_kernel_size, c_par)

    if make_reuse_def:
      dse_string += '  // {{{REUSE_NAME}}}' + "\n"
    dse_string += '  val WLP_L' + str(global_val_number) + ' = ' + str(layer_WLP) + "\n"
    if conv_input_output_from_DRAM:
      dse_string += '  val IP_L' + str(global_val_number) + '  = ' + str(layer_IP) + "\n"
      dse_string += '  val OP_L' + str(global_val_number) + '  = ' + str(layer_OP) + "\n"
      dse_string += '  val DLP_L' + str(global_val_number) + ' = ' + str(layer_IP*4) + "\n"
      dse_string += '  val SP_L' + str(global_val_number) + '  = ' + str(layer_IP*4) + "\n"
    dse_string += "\n"

    return output_string, dse_string, weight_blocking, layer_IP
  
  tmpvar = 'tmp' + str(global_val_number)
  name_to_tmpvar[node.name] = tmpvar
  tmpvar_inputs = get_inputs(node, name_to_tmpvar)
  
  # Generate each op
  if node.op == 'MatMul':
    fc_section = True
    if not reuse_FC:
      accel_function += '        // ' + node.op + "\n"
  else:
    if not reuse:
      accel_function += '        // ' + node.op + "\n"
  # Final MatMul check: if this is a Conv2D but with 1x1 input, use MatMul here too.
  # Sometimes instead of using a MatMul for the final layer, a 1x1 Conv is used.
  # E.g. this is done by MobileNet. In this case, special-case to MatMul. In the 
  # future, could re-use Conv2D 1x1 here since blocked 1x1 would still be efficient.
  process_node_as_Conv2D = False
  process_node_as_MatMul = False
  if node.op == 'Conv2D':
    data_dims_str = get_data_dims_str(node, frz_sess)
    if data_dims_str[0] == '1' and data_dims_str[1] == '1':
      process_node_as_MatMul = True
    else:
      process_node_as_Conv2D = True
  elif node.op == 'MatMul':
    process_node_as_MatMul = True
  
  # -----------------------------------------------------
  # Convolution
  # -----------------------------------------------------
  if process_node_as_Conv2D:

    # Get input and kernel sizes, which are not properties of op but of tensor inputs to op    
    assert len(node.input) == 2      
    data_dims_str = get_data_dims_str(node, frz_sess)
    kernel_dims_str = get_kernel_dims_str(node, frz_sess)
    
    # Now that we have dimensions, also get padding and stride
    out_height, out_width, out_size, strides, padding = get_output_dim(node, kernel_dims_str, data_dims_str, kernel_dims_str[3])
    final_out_size = out_size
    
    if node.input[0] in extra_paddings.keys():
      data_dims_str = unpadded_dims[node.input[0]]
      previous_paddings = extra_paddings[node.input[0]]
      # Handle cases of tf.pad
      supported = False
      # If all prev pads are 0, then that pad was useless
      if all(pad == 0 for pad in previous_paddings):
        supported = True
        # Same and valid are identical for k=1 when no extra padding, so default to same
        if int(kernel_dims_str[0]) == 1 and int(kernel_dims_str[1]) == 1:
          padding = 'SAME'
      # Otherwise, check the padding is of the form [0 0 p p p p 0 0]
      elif previous_paddings[0] == 0 and previous_paddings[1] == 0 and previous_paddings[6] == 0 and previous_paddings[7] == 0 and \
           previous_paddings[2] == previous_paddings[3] and previous_paddings[2] == previous_paddings[4] and previous_paddings[2] == previous_paddings[5]:
        # In this case, rather than use SAME padding, the user chose to pad manually and use VALID padding
        # We can just convert this to SAME padding
        # https://www.tensorflow.org/versions/r1.12/api_guides/python/nn#Convolution
        # Note: e.g. SAME on a 3x3 w/ s=1 is border of 1, but SAME on a 3x3 w/ s=2 is border of 1 on bottom/right only
        if previous_paddings[2] == (int(kernel_dims_str[0])-1)/2:
          assert padding == 'VALID'
          supported = True
          padding = str(previous_paddings[2])
      if not supported:
        print 'Currently this pad is not supported'
        assert False

    # If the data is an input, do a setMem
    is_input = False
    if tmpvar_inputs[0][0] == 'i':
      # Declare a DRAM for the input to conv
      if tmpvar_inputs[0] not in input_dram_already_created:
        input_dram_already_created.add(tmpvar_inputs[0])
        data_mem_declarations += '    val ' + tmpvar_inputs[0] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(data_dims_str))   + ')' + "\n"
        data_mem_set += '    setMem(' + tmpvar_inputs[0] + '_DRAM, ' + tmpvar_inputs[0] + get_reshape_string(tmpvar_inputs[0]) + ")\n"
      is_input = True
    
    if reuse:
      in2D_size = int(data_dims_str[0])*int(data_dims_str[1])
      def_arg_values = {}
      def_arg_values['nr: Int'              ] = data_dims_str[0]
      def_arg_values['nc: Int'              ] = data_dims_str[1]
      def_arg_values['or: Int'              ] = str(out_height)
      def_arg_values['oc: Int'              ] = str(out_height)
      def_arg_values['in_channels: Int'     ] = kernel_dims_str[2]
      def_arg_values['out_channels: Int'    ] = kernel_dims_str[3]
      def_arg_values['stride: Int'          ] = strides[1]
      def_arg_values['bias_start_idx : Int'  ] = None # Gets updated later
      def_arg_values['weights_start_idx : Int'  ] = None # Gets updated later
      def_arg_values['store_idx : Int'        ] = None # Gets updated later
      def_arg_values['load_idx_0 : Int'         ] = None # Gets updated later
      def_arg_values['MAX__or: Int'          ] = data_dims_str[0]
      def_arg_values['MAX__oc: Int'          ] = data_dims_str[1]
      def_arg_values['in2D: Int'        ] = str(in2D_size)
      def_arg_values['MAX__in2D_aligned: Int'   ] = str(burst_align(in2D_size))
      def_arg_values['out2D: Int'       ] = str((out_height*out_width))
      def_arg_values['MAX__out2D_aligned: Int'  ] = str(burst_align(out_height*out_width))
      def_arg_values['MAX__in_channels: Int' ] = kernel_dims_str[2]
      def_arg_values['MAX__out_channels: Int'] = kernel_dims_str[3]
      def_arg_values['use_relu: Boolean'    ] = str('false') # Gets updated later
            
      # Calculate in buffer size
      inChannel_block_size = 1
      if block_input_channels(kernel_dims_str):
        inChannel_block_size = closest_pow_2(device_params['image_buffer_size'] / in2D_size)
      def_arg_values['inB : Int'            ] = str(inChannel_block_size)
    
    # Fusion
    match_nodes1 = fusion_optimization_match(['Conv2D', 'BiasAdd', 'Relu', 'MaxPool'], output_graph_def.node, node_idx-1)
    match_nodes2 = fusion_optimization_match(['Conv2D', 'BiasAdd', 'Relu'], output_graph_def.node, node_idx-1)
    match_nodes3 = fusion_optimization_match(['Conv2D', 'BiasAdd'], output_graph_def.node, node_idx-1)
    
    # ['Conv2D', 'BiasAdd', 'Relu', 'MaxPool']
    if match_nodes1:
      match_nodes = match_nodes1
      for node in match_nodes:
        nodes_already_processed_through_fusion.add(node.name)
      
      # Get bias add parameters
      bias_node = match_nodes[1]
      assert len(bias_node.input) == 2      
      bias_tmpvar_inputs = get_inputs(bias_node, name_to_tmpvar)
      bias_kernel_dims_str = get_kernel_dims_str(bias_node, frz_sess)
      
      # Biases are weights from DRAM
      weight_mem_declarations_no_reuse += '    val ' + tmpvar_inputs[1] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(kernel_dims_str)) + ')' + "\n"
      weight_mem_declarations_no_reuse += '    setMem(' + tmpvar_inputs[1] + '_DRAM, ' + tmpvar_inputs[1] + get_reshape_string(tmpvar_inputs[1]) + ")\n"
      weight_mem_declarations_no_reuse += '    val ' + bias_tmpvar_inputs[1] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(bias_kernel_dims_str)) + ')' + "\n"
      weight_mem_declarations_no_reuse += '    setMem(' + bias_tmpvar_inputs[1] + '_DRAM, ' + bias_tmpvar_inputs[1] + ')' + "\n"
      
      # Get pooling parameters
      pool_node = match_nodes[3]
      pool_kernel_dims = get_pool_kernel_dims(pool_node)
      pool_input_dims = [out_height, out_width, int(kernel_dims_str[3])]
      pool_out_height, pool_out_width, pool_out_size, pool_strides, pool_padding = get_output_dim(pool_node, pool_kernel_dims, pool_input_dims, kernel_dims_str[3])
      final_out_size = pool_out_size

      accel_function_out, dse_string_out, weight_blocking, layer_IP = conv_before_fusion(is_input, tmpvar, tmpvar_inputs, bias_tmpvar_inputs, \
        bias_kernel_dims_str, data_dims_str, kernel_dims_str, strides, padding, out_size.split(','), final_out_size, False)
      accel_function += accel_function_out
      file_opening   += dse_string_out
      
      unique_op_name = 'Fused_Conv2D_BiasAdd_MaxPool_k' + kernel_dims_str[0]
      total_ops = out_height*out_width*int(kernel_dims_str[0])*int(kernel_dims_str[1])*int(kernel_dims_str[2])*int(kernel_dims_str[3])
      kxk = int(kernel_dims_str[0])*int(kernel_dims_str[1])
      register_layer_ops(unique_op_name, total_ops, layer_IP, kxk)
      
      # Check if we need to handle edges
      if pool_padding == 'SAME':
        pad_top, pad_left, pad_bottom, pad_right = get_same_padding(int(out_size.split(',')[0]),
          int(out_size.split(',')[1]), pool_strides, int(pool_kernel_dims[1]), int(pool_kernel_dims[2]))
        assert pad_top == pad_left
        assert pad_bottom == pad_right
        assert pad_top == 0
        assert pad_bottom in [0,1]
      # VALID is a special-case of SAME when padding is 0
      else:
        # No padding for VALID
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
      
      pool_output_2D = str(burst_align(pool_out_height*pool_out_width))
      
      # If output is to DRAM, create the tile SRAM
      if conv_input_output_from_DRAM:
        accel_function += '''
          // Fused BiasAdd
          Foreach(B by 1) { b =>
            val ''' + tmpvar + '''_SRAM_pool = SRAM[T](''' + str(pool_output_2D) + ''')
            Foreach(''' + str(pool_out_height) + ''' by 1, ''' + str(pool_out_width) + ''' by 1) { (i,j) =>'''
      # Otherwise we can fuse the loops because no need to perform any stores
      else:
        if weight_blocking:
          accel_function += '''
          // Fused BiasAdd
            Foreach(''' + str(pool_out_height) + ''' by 1, ''' + str(pool_out_width) + ''' by 1, B by 1) { (i,j,b) =>'''
        else:
          accel_function += '''
          // Fused BiasAdd
            Foreach(''' + str(pool_out_height) + ''' by 1, ''' + str(pool_out_width) + ''' by 1) { (i,j) =>'''
      accel_function += '''
              val pool_elements: List[T] = List.tabulate(''' + pool_kernel_dims[1] + '''){ii => List.tabulate(''' + pool_kernel_dims[2] + '''){jj =>'''
      if pad_bottom == 0:
        if weight_blocking:
          accel_function += '''
                max(0.to[T], ''' + tmpvar + '''_SRAM_conv(i*''' + str(pool_strides[1]) + ''' + ii, j*''' + str(pool_strides[2]) + ''' + jj, b) + ''' + bias_tmpvar_inputs[1] + '''_SRAM(outCh_i + b))'''
        else:
          accel_function += '''
                max(0.to[T], ''' + tmpvar + '''_SRAM_conv(i*''' + str(pool_strides[1]) + ''' + ii, j*''' + str(pool_strides[2]) + ''' + jj) + ''' + bias_tmpvar_inputs[1] + '''_SRAM(outCh_i))'''
      else:
        accel_function += '''
                mux(
                    i.to[Int]*''' + str(pool_strides[1]) + ''' + ii.to[Int] >= ''' + out_size.split(',')[0] + \
             ''' || j.to[Int]*''' + str(pool_strides[2]) + ''' + jj.to[Int] >= ''' + out_size.split(',')[1] + ''',
                    0.to[T],
                    max(
                        0.to[T], 
                        ''' + tmpvar + '''_SRAM_conv(
                            min(''' + out_size.split(',')[0] + '''-1, i.to[Int]*''' + str(pool_strides[1]) + ''' + ii.to[Int]),
                            min(''' + out_size.split(',')[1] + '''-1, j.to[Int]*''' + str(pool_strides[2]) + ''' + jj.to[Int])'''
        if conv_input_output_from_DRAM or weight_blocking:
          accel_function += ''',
                            b'''
        accel_function += '''
                        ) + ''' + bias_tmpvar_inputs[1] + '''_SRAM(outCh_i.to[Int] + b.to[Int])
                    )
                )'''
      accel_function += '''
              }}.flatten'''
      
      # If output is to DRAM, write to the tile SRAM
      if conv_input_output_from_DRAM:
        accel_function += '''
              ''' + tmpvar + '''_SRAM_pool(i.to[Int]*''' + str(pool_out_width) + ''' + j.to[Int]) = ReduceTree(pool_elements :_*){max(_,_)}
            }'''
        if reuse:
          # Can also support reuse for this node, and look up the buffer of tmp_DRAM from the parent.
          # E.g. create tmp_DRAM_3D for initial layers if LBs are needed and load input into that.
          # Currently asserting this is the first node.
          assert not reuse_tmp_dram_ordered
          accel_function += '''
            tmp_DRAM(0.to[Int], (outCh_i.to[Int] + b.to[Int])*''' + pool_output_2D + ''' :: (outCh_i.to[Int] + b.to[Int])*''' + pool_output_2D + ' + ' + str(pool_output_2D) + ''') store ''' + tmpvar + '''_SRAM_pool'''
        else:
          accel_function += '''
            ''' + tmpvar + '''_DRAM(outCh_i.to[Int] + b.to[Int])*''' + pool_output_2D + ''' :: (outCh_i.to[Int] + b.to[Int])*''' + pool_output_2D + ' + ' + str(pool_output_2D) + ''') store ''' + tmpvar + '''_SRAM_pool'''
        accel_function += '''
          }'''
      # Otherwise output SRAM has been declared before the convolution, write to that
      else:
        if weight_blocking:
          accel_function += '''
              ''' + tmpvar + '''_SRAM(outCh_i + b, i, j) = ReduceTree(pool_elements :_*){max(_,_)}
            }'''
        else:
          accel_function += '''
              ''' + tmpvar + '''_SRAM(outCh_i, i, j) = ReduceTree(pool_elements :_*){max(_,_)}
            }'''
      
      accel_function += '''
        }
        // Optimization: BiasAdd was merged into Conv2D above
        // Optimization: ReLU was merged into Conv2D above
        // Optimization: MaxPool was merged into Conv2D above
'''
      name_to_tmpvar[match_nodes[3].name] = tmpvar

    # ['Conv2D', 'BiasAdd', 'Relu'],
    # ['Conv2D', 'BiasAdd']
    # Implement both here since only difference is a mux
    elif match_nodes2 or match_nodes3:
    
      if match_nodes2:
        use_relu = True
        match_nodes = match_nodes2
      else:
        use_relu = False
        match_nodes = match_nodes3
        
      for node in match_nodes:
        nodes_already_processed_through_fusion.add(node.name)
       
      # Get bias add parameters
      bias_node = match_nodes[1]
      assert len(bias_node.input) == 2      
      bias_tmpvar_inputs = get_inputs(bias_node, name_to_tmpvar)
      bias_kernel_dims_str = get_kernel_dims_str(bias_node, frz_sess)
      
      # For now, only re-use if no Line Buffer needed
      # Later can add reuse support for LB as well
      reuse_supported = False
      if reuse and not use_line_buffer(data_dims_str):
        reuse_supported = True
      # Special case: if this is a 1x1 conv, it is ok to re-use, as long as it is not the first layer
      # This is because Line Buffer is not actually used for 1x1, so if it is not the first layer (dealing with 3D input)
      # there is no need for this to deal with 3D at all
      if reuse and use_line_buffer(data_dims_str):
        if int(kernel_dims_str[0]) == 1 and int(kernel_dims_str[1]) == 1:
          if tmpvar_inputs[0][0] != 'i':
            reuse_supported = True
      
      if not reuse_supported:
        # Kernel in DRAM
        weight_mem_declarations_no_reuse += '    val ' + tmpvar_inputs[1] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(kernel_dims_str)) + ')' + "\n"
        weight_mem_declarations_no_reuse += '    setMem(' + tmpvar_inputs[1] + '_DRAM, ' + tmpvar_inputs[1] + get_reshape_string(tmpvar_inputs[1]) + ")\n"
        # Biases in DRAM
        weight_mem_declarations_no_reuse += '    val ' + bias_tmpvar_inputs[1] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(bias_kernel_dims_str)) + ')' + "\n"
        weight_mem_declarations_no_reuse += '    setMem(' + bias_tmpvar_inputs[1] + '_DRAM, ' + bias_tmpvar_inputs[1] + ')' + "\n"
      
      hw_block, dse_string_out, weight_blocking, layer_IP = conv_before_fusion(is_input, tmpvar, tmpvar_inputs, bias_tmpvar_inputs, bias_kernel_dims_str, data_dims_str, kernel_dims_str, \
        strides, padding, out_size.split(','), final_out_size, reuse_supported)

      # If output is to DRAM
      if conv_input_output_from_DRAM:
        if reuse_supported:
          if use_relu:
            def_arg_values['use_relu: Boolean'] = str('true') # Default is false
          out_name = 'out'
          out_dims = 'MAX__out2D_aligned'
          out_r = 'or'
          out_c = 'oc'
          bias_sram_name = 'bias'
          use_relu_name = 'use_relu'
        else:
          out_name = tmpvar
          out_r = out_size.split(',')[0]
          out_c = out_size.split(',')[1]
          out_dims = str(burst_align(int(out_r)*int(out_c))) # Currently not used
          bias_sram_name = bias_tmpvar_inputs[1]
          if use_relu:
            use_relu_name = 'true'
          else:
            use_relu_name = 'false'

        store_block_size = '1'
        # Can check if beneficial to block stores, if so can uncomment this check
        # and then also add 1 more loop iterator below for when this is true 
        #if block_input_channels(kernel_dims_str):
        #  store_block_size = 'ofmap_block_store_size'

        # The par here should be > the store par below and also match the par of the conv above.
        # E.g. instead of 8, the lowest common dimension (e.g. 7) may work better.
        hw_block += '''
          // Fused BiasAdd
          Foreach(B by ''' + store_block_size + ''') { b =>
            val ''' + out_name + '''_SRAM_bias = SRAM[T](''' + out_dims + ''')
            Foreach(0 until ''' + out_r + ''', 0 until ''' + out_c + ''' par SP_L''' + str(global_val_number) + ''') { (r,c) =>'''
        bias_load_inside = kernel_dims_str[0] == '1'
        if bias_load_inside:
          hw_block += '''
              val sum = ''' + out_name + '''_SRAM_conv(r,c,b) + ''' + bias_sram_name + '''_SRAM(b)'''
        else:
          hw_block += '''
              val sum = ''' + out_name + '''_SRAM_conv(r,c,b) + ''' + bias_sram_name + '''_SRAM(outCh_i.to[Int] + b.to[Int])'''
        if use_relu6:
          hw_block += '''
              ''' + out_name + '''_SRAM_bias(r.to[Int]*''' + out_c + ''' + c.to[Int]) = mux( ''' + use_relu_name + ''', min(max(sum, 0.to[T]), 6.to[T]), sum )
            }'''
        else:
          hw_block += '''
              ''' + out_name + '''_SRAM_bias(r.to[Int]*''' + out_c + ''' + c.to[Int]) = mux( ''' + use_relu_name + ''' && (sum < 0.to[T]), 0.to[T], sum )
            }'''
        if reuse_supported:
          hw_block += '''
            tmp_DRAM(store_idx, (outCh_i.to[Int] + b.to[Int])*out2D :: (outCh_i.to[Int] + b.to[Int] + ''' + store_block_size + ''')*out2D par SP_L''' + str(global_val_number) + ''') store ''' + out_name + '''_SRAM_bias'''
        else:
          # Can also support reuse for this node, and look up the buffer of tmp_DRAM from the parent.
          # E.g. create tmp_DRAM_3D for initial layers if LBs are needed and load input into that.
          # Currently asserting this is the first node.
          assert not reuse_tmp_dram_ordered
          hw_block += '''
            tmp_DRAM(0.to[Int], (outCh_i.to[Int] + b.to[Int])*''' + str(burst_align(out_height*out_width)) + ''' :: (outCh_i.to[Int] + b.to[Int])*''' + str(burst_align(out_height*out_width)) + ' + ' + str(burst_align(out_height*out_width)) + ''' par SP_L''' + str(global_val_number) + ''') store ''' + out_name + '''_SRAM_bias'''
          #hw_block += '''
          #  ''' + out_name + '''_DRAM(outCh_i.to[Int] + b.to[Int], 0::''' + str(burst_align(out_height*out_width)) + ''' par SP_L''' + str(global_val_number) + ''') store ''' + out_name + '''_SRAM_bias'''
        hw_block += '''
          }
        }'''
      # If output is to SRAM
      else:
        hw_block += '''
          // Fused BiasAdd
          Foreach(''' + str(out_height) + ''' by 1, ''' + str(out_width) + ''' by 1) { (i,j) =>'''
        if use_relu:
          hw_block += '''
            ''' + tmpvar + '''_SRAM(outCh_i, i, j) = max(0.to[T], ''' + tmpvar + '''_SRAM_conv(i,j) + ''' + bias_tmpvar_inputs[1] + '''_SRAM(outCh_i))'''
        else:
          hw_block += '''
            ''' + tmpvar + '''_SRAM(outCh_i, i, j) = ''' + tmpvar + '''_SRAM_conv(i,j) + ''' + bias_tmpvar_inputs[1] + '''_SRAM(outCh_i)'''
        hw_block += '''
          }
        }'''
      hw_block += '''
        // Optimization: BiasAdd was merged into Conv2D above'''
      if reuse_supported or use_relu:
        hw_block += '''
        // Optimization: ReLU was merged into Conv2D above'''
      hw_block += '''
'''
      if use_relu:
        name_to_tmpvar[match_nodes[2].name] = tmpvar
      else:
        name_to_tmpvar[match_nodes[1].name] = tmpvar

      # Note Relu is being omitted because a mux can be used instead
      large_identifier = ''
      if use_line_buffer(data_dims_str):
        large_identifier = '_large'
      unique_op_name = 'Fused_Conv2D_BiasAdd_k' + kernel_dims_str[1] + large_identifier# + '_s' + strides[1]

      # If reuse, check to see whether this is the first occurrence or not
      if reuse_supported:
        hw_block = hw_block.replace('{{{REUSE_NAME}}}', unique_op_name)
        dse_string_out = dse_string_out.replace('{{{REUSE_NAME}}}', unique_op_name)
        register_new_reused_processor(unique_op_name, def_arg_values, hw_block, dse_string_out, \
          True, bias_tmpvar_inputs[1], tmpvar_inputs[1])          
      else:
        file_opening   += dse_string_out
        accel_function += hw_block

      total_ops = out_height*out_width*int(kernel_dims_str[0])*int(kernel_dims_str[1])*int(kernel_dims_str[2])*int(kernel_dims_str[3])
      kxk = int(kernel_dims_str[0])*int(kernel_dims_str[1])
      register_layer_ops(unique_op_name, total_ops, layer_IP, kxk)

    # Otherwise it is unfused convolution
    else:
      accel_function_out, dse_string_out, weight_blocking, layer_IP = conv_before_fusion(is_input, tmpvar, tmpvar_inputs, None, None, data_dims_str, kernel_dims_str, \
        strides, padding, out_size.split(','), final_out_size, False)
      accel_function += accel_function_out
      file_opening   += dse_string_out
      # If output is to DRAM
      if conv_input_output_from_DRAM:
        accel_function += '''
          ''' + tmpvar + '''_DRAM(outCh_i, 0::''' + str(out_height) + ''', 0::''' + str(burst_align(out_width)) + ''') store ''' + tmpvar + '''_SRAM_conv
        }'''
      # If output is to SRAM
      else:
        accel_function += '''
        }
'''
    # If output is to DRAM, declare it here using final_out_size
    if conv_input_output_from_DRAM:
      out_size_align_last = reformat_memory(final_out_size.split(','))
      out_size_align_last[-1] = str(burst_align(int(out_size_align_last[-1])))
      if reuse:
        reuse_tmp_dram_ordered.append(tmpvar)
        reuse_tmp_dram_dims[tmpvar] = out_size_align_last
        if tmpvar_inputs[0] not in reuse_tmp_dram_children.keys():
          reuse_tmp_dram_children[tmpvar_inputs[0]] = []
        reuse_tmp_dram_children[tmpvar_inputs[0]].append(tmpvar)
        reuse_tmp_dram_parents[tmpvar] = [tmpvar_inputs[0]]
      else:
        tmp_mem_declarations_no_reuse += '    val ' + tmpvar           + '_DRAM = DRAM[T](' + ','.join(out_size_align_last) + ')' + "\n\n"

  elif node.op == 'Pad':
    # There are 2 types of padding:
    #  - SAME
    #  - VALID
    # However, there is also a Pad op in TensorFlow which explicitly pads an image
    # Here we fuse this into the node after it, by storing the paddings for the tensor
    name_to_tmpvar[node.name] = tmpvar_inputs[0]
    unpadded_dims [node.name] = get_data_dims_str(node, frz_sess)
    extra_paddings[node.name] = extra_paddings[node.input[1]]
  
  # Note: this code is similar to Conv2D above, can merge the 2 code paths
  elif node.op == 'DepthwiseConv2dNative':

    # Get input and kernel sizes, which are not properties of op but of tensor inputs to op    
    assert len(node.input) == 2      
    data_dims_str = get_data_dims_str(node, frz_sess)
    kernel_dims_str = get_kernel_dims_str(node, frz_sess)
    
    # Now that we have dimensions, also get padding and stride
    out_height, out_width, out_size, strides, padding = get_output_dim(node, kernel_dims_str, data_dims_str, kernel_dims_str[3])
    final_out_size = out_size
    
    # If these asserts fail, copy the relevant code from Conv2D above
    assert node.input[0] not in extra_paddings.keys()
    assert not tmpvar_inputs[0][0] == 'i'
    assert conv_input_output_from_DRAM
    assert reuse
    
    if reuse:
      in2D_size = int(data_dims_str[0])*int(data_dims_str[1])
      def_arg_values = {}
      def_arg_values['nr: Int'              ] = data_dims_str[0]
      def_arg_values['nc: Int'              ] = data_dims_str[1]
      def_arg_values['or: Int'              ] = str(out_height)
      def_arg_values['oc: Int'              ] = str(out_height)
      def_arg_values['in_channels: Int'     ] = kernel_dims_str[2]
      def_arg_values['stride: Int'          ] = strides[1]
      def_arg_values['bias_start_idx : Int'  ] = None # Gets updated later
      def_arg_values['weights_start_idx : Int'  ] = None # Gets updated later
      def_arg_values['store_idx : Int'        ] = None # Gets updated later
      def_arg_values['load_idx_0 : Int'         ] = None # Gets updated later
      def_arg_values['MAX__or: Int'          ] = data_dims_str[0]
      def_arg_values['MAX__oc: Int'          ] = data_dims_str[1]
      def_arg_values['in2D: Int'        ] = str(in2D_size)
      def_arg_values['MAX__in2D_aligned: Int'   ] = str(burst_align(in2D_size))
      def_arg_values['out2D: Int'       ] = str((out_height*out_width))
      def_arg_values['MAX__out2D_aligned: Int'  ] = str(burst_align(out_height*out_width))
      def_arg_values['MAX__in_channels: Int' ] = kernel_dims_str[2]
      def_arg_values['use_relu: Boolean'    ] = str('false') # Gets updated later
            
      # Calculate in buffer size
      # Because there are no out channels and therefore no block of partial sums, load
      # the entire 2D feature map. If this ends up using too much memory, can use line
      # buffer like in Conv2D.
      if not max_depthwise_conv_input:
        for node in output_graph_def.node:
          if node.op == 'DepthwiseConv2dNative':
            input_dims = get_data_dims_str(node, frz_sess)
            input_2D_size = int(input_dims[0])*int(input_dims[1])
            max_depthwise_conv_input = max(max_depthwise_conv_input, input_2D_size)
      
      # Currently making block size 1, can increase later based on above
      # Also change bias loop below since assumes 1 now
      inChannel_block_size = 1
      #if block_input_channels(kernel_dims_str):
      #  inChannel_block_size = closest_pow_2(device_params['image_buffer_size'] / in2D_size)
      def_arg_values['inB : Int'            ] = str(inChannel_block_size)
    
    # Fusion
    match_nodes1 = fusion_optimization_match(['DepthwiseConv2dNative', 'BiasAdd', 'Relu'], output_graph_def.node, node_idx-1)
    match_nodes2 = fusion_optimization_match(['DepthwiseConv2dNative', 'BiasAdd'], output_graph_def.node, node_idx-1)
    
    # ['DepthwiseConv2dNative', 'BiasAdd', 'Relu'],
    # ['DepthwiseConv2dNative', 'BiasAdd']
    # Implement both here since only difference is a mux
    if match_nodes1 or match_nodes2:
    
      if match_nodes1:
        use_relu = True
        match_nodes = match_nodes1
      else:
        use_relu = False
        match_nodes = match_nodes2
        
      for node in match_nodes:
        nodes_already_processed_through_fusion.add(node.name)
       
      # Get bias add parameters
      bias_node = match_nodes[1]
      assert len(bias_node.input) == 2      
      bias_tmpvar_inputs = get_inputs(bias_node, name_to_tmpvar)
      bias_kernel_dims_str = get_kernel_dims_str(bias_node, frz_sess)
      
      reuse_supported = True
      
      hw_block, dse_string_out, weight_blocking, layer_IP = conv_before_fusion_depthwise(is_input, tmpvar, tmpvar_inputs, bias_tmpvar_inputs, bias_kernel_dims_str, data_dims_str, kernel_dims_str, \
        strides, padding, out_size.split(','), final_out_size, reuse_supported)
      
      if conv_input_output_from_DRAM:
        if reuse_supported:
          if use_relu:
            def_arg_values['use_relu: Boolean'] = str('true') # Default is false
          out_name = 'out'
          out_dims = 'MAX__out2D_aligned'
          out_r = 'or'
          out_c = 'oc'

        store_block_size = '1'
        # Can check if beneficial to block stores, if so can uncomment this check
        # and then also add 1 more loop iterator below for when this is true 
        #if block_input_channels(kernel_dims_str):
        #  store_block_size = 'ofmap_block_store_size'

        # The par here should be > the store par below and also match the par of the conv above.
        # E.g. instead of 8, the lowest common dimension (e.g. 7) may work better.
        hw_block += '''
          // Fused BiasAdd
          // Note: For now inB=1 so remove this loop
          // Foreach(inB by ''' + store_block_size + ''') { ib =>
            val ''' + out_name + '''_SRAM_bias = SRAM[T](''' + out_dims + ''')
            Foreach(0 until ''' + out_r + ''', 0 until ''' + out_c + ''' par 1/*SP_L''' + str(global_val_number) + '''*/) { (r,c) =>'''
        bias_load_inside = kernel_dims_str[0] == '1'
        if bias_load_inside:
          hw_block += '''
              val sum = ''' + out_name + '''_SRAM_conv(r,c,0.to[Int]/*ib*/) + bias_SRAM(0.to[Int]/*ib*/)'''
        else:
          hw_block += '''
              val sum = ''' + out_name + '''_SRAM_conv(r,c,0.to[Int]/*ib*/) + bias_SRAM(inCh_i.to[Int]/* + ib.to[Int]*/)'''
        if use_relu6:
          hw_block += '''
              ''' + out_name + '''_SRAM_bias(r.to[Int]*''' + out_c + ''' + c.to[Int]) = mux( use_relu, min(max(sum, 0.to[T]), 6.to[T]), sum )
            }'''
        else:
          hw_block += '''
              ''' + out_name + '''_SRAM_bias(r.to[Int]*''' + out_c + ''' + c.to[Int]) = mux( use_relu && (sum < 0.to[T]), 0.to[T], sum )
            }'''
        if reuse_supported:
          hw_block += '''
            tmp_DRAM(store_idx, (inCh_i.to[Int]/* + ib.to[Int]*/)*out2D :: (inCh_i.to[Int] + /*ib.to[Int] +*/ ''' + store_block_size + ''')*out2D par SP_L''' + str(global_val_number) + ''') store ''' + out_name + '''_SRAM_bias'''
        hw_block += '''
          // }'''
      hw_block += '''
          // Optimization: BiasAdd was merged into Conv2D above'''
      if reuse_supported or use_relu:
        hw_block += '''
          // Optimization: ReLU was merged into Conv2D above'''
      hw_block += '''
        }
'''
      if use_relu:
        name_to_tmpvar[match_nodes[2].name] = tmpvar
      else:
        name_to_tmpvar[match_nodes[1].name] = tmpvar
      
      # If reuse, check to see whether this is the first occurrence or not
      if reuse_supported:
        # Note Relu is being omitted because a mux can be used instead
        large_identifier = ''
        #if use_line_buffer(data_dims_str):
        #  large_identifier = '_large'
        unique_op_name = 'Fused_DepthwiseConv_BiasAdd_k' + kernel_dims_str[1] + large_identifier# + '_s' + strides[1]
        hw_block = hw_block.replace('{{{REUSE_NAME}}}', unique_op_name)
        dse_string_out = dse_string_out.replace('{{{REUSE_NAME}}}', unique_op_name)
        register_new_reused_processor(unique_op_name, def_arg_values, hw_block, dse_string_out, \
          True, bias_tmpvar_inputs[1], tmpvar_inputs[1])          
        total_ops = out_height*out_width*int(kernel_dims_str[0])*int(kernel_dims_str[1])*int(kernel_dims_str[2])*int(kernel_dims_str[3])
        kxk = int(kernel_dims_str[0])*int(kernel_dims_str[1])
        register_layer_ops(unique_op_name, total_ops, layer_IP, kxk)

    # Otherwise it is unfused convolution
    else:
      accel_function_out, dse_string_out, weight_blocking, layer_IP = conv_before_fusion_depthwise(is_input, tmpvar, tmpvar_inputs, None, None, data_dims_str, kernel_dims_str, \
        strides, padding, out_size.split(','), final_out_size, False)
      accel_function += accel_function_out
      file_opening   += dse_string_out
      # If output is to DRAM
      if conv_input_output_from_DRAM:
        accel_function += '''
            ''' + tmpvar + '''_DRAM(inCh_i, 0::''' + str(out_height) + ''', 0::''' + str(burst_align(out_width)) + ''') store ''' + tmpvar + '''_SRAM_conv
          }
        }'''
    # If output is to DRAM, declare it here using final_out_size
    if conv_input_output_from_DRAM:
      out_size_align_last = reformat_memory(final_out_size.split(','))
      out_size_align_last[-1] = str(burst_align(int(out_size_align_last[-1])))
      if reuse:
        reuse_tmp_dram_ordered.append(tmpvar)
        reuse_tmp_dram_dims[tmpvar] = out_size_align_last
        if tmpvar_inputs[0] not in reuse_tmp_dram_children.keys():
          reuse_tmp_dram_children[tmpvar_inputs[0]] = []
        reuse_tmp_dram_children[tmpvar_inputs[0]].append(tmpvar)
        reuse_tmp_dram_parents[tmpvar] = [tmpvar_inputs[0]]
      else:
        tmp_mem_declarations_no_reuse += '    val ' + tmpvar           + '_DRAM = DRAM[T](' + ','.join(out_size_align_last) + ')' + "\n\n"
  
  # -----------------------------------------------------
  # Reshape
  # -----------------------------------------------------
  elif node.op == 'Reshape':

    assert len(node.input) == 2
    data_dims_str = get_data_dims_str(node, frz_sess)
    
    reshape_dims = tensor_util.MakeNdarray(name_to_node[node.input[1]].attr['value'].tensor).tolist()
    print 'Reshape with target dims ' + str( reshape_dims ) + ', input dims ' + str(data_dims_str)
    
    if node.input[0] in name_to_node.keys() and name_to_node[node.input[0]].op in input_ops:
      input_name = tmpvar_inputs[0]
      assert len(reshape_dims) in [1,2,4]
      name_to_tmpvar[node.name] = tmpvar_inputs[0]
      if len(reshape_dims) == 4:
        reshape_dims = reshape_dims[1:]
        tmpvar_to_reshape_string[input_name] = str(reshape_dims[2]) + ',' + str(reshape_dims[0]) + ',' + str(reshape_dims[1])
        accel_function += '        // Done on host' + "\n"
      elif len(reshape_dims) == 1:
        assert reshape_dims[0] == -1
        accel_function += "        // Skipping, already 1D\n"
        # No reshape needed, 1D already and same size
      
    elif len(data_dims_str) > 1:
      # If input is DRAM
      if conv_input_output_from_DRAM:
        accel_function += '''
        val ''' + tmpvar + '''_SRAM = SRAM[T](''' + '*'.join(data_dims_str) + ''')
        Foreach(''' + data_dims_str[2] + ''' by 1) { j =>
          Foreach(''' + data_dims_str[1] + ''' by 1) { i =>
            val row = SRAM[T](''' + data_dims_str[0] + ''')
            row load ''' + tmpvar_inputs[0] + '''_DRAM(j, i, 0::''' + data_dims_str[0] + ''')
            Foreach(''' + data_dims_str[0] + ''' by 1) { k =>
              ''' + tmpvar + '''_SRAM(k*''' + data_dims_str[2] + ''' + i*''' + data_dims_str[1] + '''*''' + data_dims_str[2] + ''' + j) = row(k)
            }
          }
        }
'''
      # If input is SRAM
      else:
        accel_function += '''
        val ''' + tmpvar + '''_SRAM = SRAM[T](''' + '*'.join(data_dims_str) + ''')
        Foreach(''' + data_dims_str[2] + ''' by 1, ''' + data_dims_str[1] + ''' by 1, ''' + data_dims_str[0] + ''' by 1) { (j,i,k) =>
          ''' + tmpvar + '''_SRAM(k*''' + data_dims_str[2] + ''' + i*''' + data_dims_str[1] + '''*''' + data_dims_str[2] + ''' + j) = ''' + tmpvar_inputs[0] + '''_SRAM(j, i, k)
        }
'''
    else:
      accel_function += '''        // Skipping reshape since ''' + tmpvar + ' and ' + tmpvar_inputs[0] + ''' already 1d
'''
      name_to_tmpvar[node.name] = tmpvar_inputs[0]
      
  # -----------------------------------------------------
  # Mean
  # -----------------------------------------------------
  elif node.op in ['Mean', 'AvgPool']:
  
    data_dims_str = get_data_dims_str(node, frz_sess)
    rc_unaligned = str(int(data_dims_str[0]) * int(data_dims_str[0]))
    rc_size = str(int(data_dims_str[0]) * int(data_dims_str[0]))
    rc_aligned = str(burst_align(int(data_dims_str[0]) * int(data_dims_str[0])))
    
    # If input is DRAM
    if conv_input_output_from_DRAM:
      accel_function += '''        val ''' + tmpvar + '''_SRAM = SRAM[T](''' + data_dims_str[2] + ''')
        Foreach(''' + data_dims_str[2] + ''' by 1) { channel =>
          val feature_map = SRAM[T](''' + rc_aligned + ''')'''
      if reuse:
        # Use LUT to find the output
        # Minus 1 since we omit the 1st element from the LUT (always 0).
        stored_idx_of_parent = 'store_idx_args(' + str(reuse_tmp_dram_ordered.index(tmpvar_inputs[0])-1) + ')'
        accel_function += '''
          feature_map load tmp_DRAM(''' + stored_idx_of_parent + ''', channel*''' + rc_size + ''' :: channel*''' + rc_size + ''' + ''' + rc_size + ''')'''
      else:
        accel_function += '''
          feature_map load ''' + tmpvar_inputs[0] + '''_DRAM(channel*''' + rc_size + ''' :: channel*''' + rc_size + ''' + ''' + rc_size + ''')'''
      accel_function += '''
          val sum = Reduce(Reg[T](0.to[T]))(''' + rc_unaligned + ''' by 1 par 1){ i => feature_map(i)*''' + str( 1.0 / ( float(data_dims_str[0])*float(data_dims_str[1]) ) ) + '''.to[T] }{_+_}
          ''' + tmpvar + '''_SRAM(channel) = sum.value
        }
'''
    # If input is SRAM
    else:
      accel_function += '''        val ''' + tmpvar + '''_SRAM = SRAM[T](''' + '*'.join(data_dims_str) + ''')
        Foreach(''' + data_dims_str[2] + ''' by 1) { channel =>
          val sum = Reduce(Reg[T](0.to[T]))(''' + rc_unaligned + ''' by 1 par 1){ (i,j) => ''' + tmpvar_inputs[0] + '''_SRAM(channel, i,j) }{_+_}
          ''' + tmpvar + '''_SRAM(channel) = sum.value * ''' + str( 1.0 / ( float(data_dims_str[0])*float(data_dims_str[1]) ) ) + '''.to[T]
        }
'''
    accel_function += "\n"

  # -----------------------------------------------------
  # Add
  # -----------------------------------------------------
  elif node.op == 'Add':
    # This is a general add of 2 tensors
    assert len(node.input) == 2
    assert conv_input_output_from_DRAM
    
    T1_dims_str = get_dims_str(node.input[0], frz_sess, 1)
    T2_dims_str = get_dims_str(node.input[1], frz_sess, 1)
    assert T1_dims_str == T2_dims_str

    # Fusion
    match_nodes1 = fusion_optimization_match(['Add', 'Relu'], output_graph_def.node, node_idx-1)
    
    # ['Add', 'Relu']
    if match_nodes1:
      match_nodes = match_nodes1
      for node in match_nodes:
        nodes_already_processed_through_fusion.add(node.name)

      if len(T1_dims_str) == 3:
        if reuse:
          in_channels = 'in_channels'
          img2D_SRAM_size = 'MAX__in2D_aligned'
          img2D_size = 'in2D'
          def_arg_values = {}
          def_arg_values['in_channels: Int'] = T1_dims_str[2]
          def_arg_values['MAX__in2D_aligned: Int'] = str(burst_align(int(T1_dims_str[0])*int(T1_dims_str[1])))
          def_arg_values['in2D: Int'] = str(int(T1_dims_str[0])*int(T1_dims_str[1]))
          def_arg_values['store_idx : Int'        ] = None # Gets updated later
          def_arg_values['load_idx_0 : Int'        ] = None # Gets updated later
          def_arg_values['load_idx_1 : Int'        ] = None # Gets updated later
        else:
          in_channels = T1_dims_str[2]
          img2D_size = str(int(T1_dims_str[0])*int(T1_dims_str[1]))
          img2D_SRAM_size = img2D_size
        hw_block = '''        val block_size = ''' + img2D_SRAM_size + '''
        Foreach(''' + in_channels + '*' + img2D_size + ''' by block_size) { idx =>
          val feature_map_a = SRAM[T](block_size)'''
        if reuse:
          hw_block += '''
          feature_map_a load tmp_DRAM(load_idx_0, idx :: idx + block_size par 16)'''
        else:
          hw_block += '''
          feature_map_a load ''' + tmpvar_inputs[0] + '''_DRAM(idx :: idx + block_size par 16)'''
        hw_block += '''
          
          val feature_map_b = SRAM[T](block_size)'''
        if reuse:
          hw_block += '''
          feature_map_b load tmp_DRAM(load_idx_1, idx :: idx + block_size par 16)'''
        else:
          hw_block += '''
          feature_map_b load ''' + tmpvar_inputs[1] + '''_DRAM(idx :: idx + block_size par 16)'''
        hw_block += '''
          
          val feature_map_sum = SRAM[T](block_size)
          Foreach(block_size by 1 par 16) { i => feature_map_sum(i) = max(0.to[T], feature_map_a(i) + feature_map_b(i)) }
          '''
        if reuse:
          hw_block += '''
          tmp_DRAM(store_idx, idx :: idx + block_size par 16) store feature_map_sum'''
        else:
          hw_block += '''
          ''' + tmpvar + '''_DRAM(idx :: idx + block_size par 16) store feature_map_sum'''
        hw_block += '''
        }
'''
        # If reuse, check to see whether this is the first occurrence or not
        if reuse:
          register_new_reused_processor('Add3D', def_arg_values, hw_block)
        else:
          accel_function += hw_block

      name_to_tmpvar[match_nodes[1].name] = tmpvar

    # If output is to DRAM, declare it here
    if conv_input_output_from_DRAM:
      out_size_align_last = reformat_memory(T1_dims_str)
      out_size_align_last[-1] = str(burst_align(int(out_size_align_last[-1])))
      if reuse:
        reuse_tmp_dram_ordered.append(tmpvar)
        reuse_tmp_dram_dims[tmpvar] = out_size_align_last
        if tmpvar_inputs[0] not in reuse_tmp_dram_children.keys():
          reuse_tmp_dram_children[tmpvar_inputs[0]] = []
        if tmpvar_inputs[1] not in reuse_tmp_dram_children.keys():
          reuse_tmp_dram_children[tmpvar_inputs[1]] = []
        reuse_tmp_dram_children[tmpvar_inputs[0]].append(tmpvar)
        reuse_tmp_dram_children[tmpvar_inputs[1]].append(tmpvar)
        reuse_tmp_dram_parents[tmpvar] = [tmpvar_inputs[0], tmpvar_inputs[1]]
      else:
        tmp_mem_declarations_no_reuse += '    val ' + tmpvar           + '_DRAM = DRAM[T](' + ','.join(out_size_align_last) + ')' + "\n\n"
      
  # -----------------------------------------------------
  # MatMul
  # -----------------------------------------------------
  elif process_node_as_MatMul:
    
    # If reuse_FC is true, these will be called in a loop. Because input is in SRAM, it will
    # not make a def like reuse with DRAM.
    
    # Get input and kernel sizes, which are not properties of op but of tensor inputs to op
    assert len(node.input) == 2
    kernel_dims_str = get_kernel_dims_str(node, frz_sess)    
    data_dims_str = get_data_dims_str(node, frz_sess)
    
    # This might be a convolution with input 1x1, if so make it a MatMul here
    conv_as_matmul = False
    if node.op == 'Conv2D':
      assert data_dims_str[0] == '1' and data_dims_str[1] == '1'
      assert not reuse_FC # Can handle this case later
      conv_as_matmul = True
      node.op = 'MatMul'
    
    if not reuse_FC:
      weight_mem_declarations_no_reuse += '    val ' + tmpvar_inputs[1] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(kernel_dims_str)) + ')' + "\n"
      weight_mem_declarations_no_reuse += '    setMem(' + tmpvar_inputs[1] + '_DRAM, ' + tmpvar_inputs[1] + get_reshape_string(tmpvar_inputs[1]) + ")\n"
    
    is_input = False
    if tmpvar_inputs[0][0] == 'i':
      # Declare a DRAM for the input to fc
      if tmpvar_inputs[0] not in input_dram_already_created:
        input_dram_already_created.add(tmpvar_inputs[0])
        data_mem_declarations += '    val ' + tmpvar_inputs[0] + '_DRAM = DRAM[T](' + ','.join(reformat_memory(data_dims_str))   + ')' + "\n"
        data_mem_set += '    setMem(' + tmpvar_inputs[0] + '_DRAM, ' + tmpvar_inputs[0] + get_reshape_string(tmpvar_inputs[0]) + ")\n"
      is_input = True
    
    # Fusion
    match_nodes1 = fusion_optimization_match(['MatMul', 'BiasAdd', 'Relu'], output_graph_def.node, node_idx-1)
    match_nodes2 = fusion_optimization_match(['MatMul', 'BiasAdd', 'Sigmoid'], output_graph_def.node, node_idx-1)
    match_nodes3 = fusion_optimization_match(['MatMul', 'BiasAdd'], output_graph_def.node, node_idx-1)
    
    # ['MatMul', 'BiasAdd', 'Relu']
    match_nodes = None
    if match_nodes1:
      match_nodes = match_nodes1
    # ['MatMul', 'BiasAdd', 'Sigmoid']
    elif match_nodes2:
      match_nodes = match_nodes2
      include_sigmoid_initialization = True
    # ['MatMul', 'BiasAdd']
    elif match_nodes3:
      match_nodes = match_nodes3
    assert match_nodes
      
    for node in match_nodes:
      nodes_already_processed_through_fusion.add(node.name)      
    
    # Get bias add parameters
    bias_node = match_nodes[1]
    assert len(bias_node.input) == 2      
    bias_tmpvar_inputs = get_inputs(bias_node, name_to_tmpvar)
    bias_data_dims_str = get_data_dims_str(bias_node, frz_sess)
    bias_kernel_dims_str = get_kernel_dims_str(bias_node, frz_sess)
    # If Conv2D, remove dimensions of size 1
    if conv_as_matmul:
      data_dims_str = kernel_dims_str[2:]
      kernel_dims_str = kernel_dims_str[2:]
      bias_data_dims_str = [bias_data_dims_str[-1]]
    assert len(bias_data_dims_str) == 1
    assert len(bias_kernel_dims_str) == 1
    assert bias_data_dims_str[0] == bias_kernel_dims_str[0]
    bias_size = bias_kernel_dims_str[0]
    
    if reuse_FC:
      unique_op_name = 'Fused_FC_BiasAdd'
      if reuse_FC_name:
        assert reuse_FC_name == unique_op_name
      else:
        reuse_FC_name = unique_op_name
        new_dram_map = {}
        new_dram_map['bias'] = []
        new_dram_map['weights'] = []
        reuse_fc_weight_dram[unique_op_name] = new_dram_map
    
        # Make an SRAM which is double-buffered and of the size of this input (biggest FC 1st)
        accel_function += '''
        val data0_SRAM = SRAM[T](max_fc_weight_cols).nonbuffer
        val data1_SRAM = SRAM[T](max_fc_weight_cols).nonbuffer'''
        if is_input:
          accel_function += '''
        data0_SRAM load ''' + tmpvar_inputs[0] + '''_DRAM(0::''' + kernel_dims_str[0] + ''')'''
        else:
          utils.error_exit('Currently, reuse FC is only supported when the first fc input is from DRAM')
        accel_function += '''

        Sequential.Foreach(0 until num_fc_layers) { L =>
          val bias_SRAM = SRAM[T](max_fc_bias_size)
          bias_SRAM load ''' + unique_op_name + '''_bias_concat_DRAM(bias_offset(L) :: bias_offset(L) + weight_rows(L))
          Foreach(weight_rows(L) by 1 par OP_L''' + str(global_val_number) + ''') { r =>
            val weight_SRAM = SRAM[T2](max_fc_weight_cols)
            weight_SRAM load ''' + unique_op_name + '''_weights_concat_DRAM(weights_offset(L) + r.to[Int]*weight_cols(L) :: weights_offset(L) + (r.to[Int]+1)*weight_cols(L) par IP_L''' + str(global_val_number) + ''')
            
            val prod = Reduce(Reg[T](0.to[T]))(weight_cols(L) by 1 par IP_L''' + str(global_val_number) + '''){ c => 
              val data = 
                mux(L % 2 == 0, 
                  data0_SRAM(c), 
                  data1_SRAM(c)
                )
              data * weight_SRAM(c).to[T]
            }{_+_}
            val prod_plus_bias = prod.value + bias_SRAM(r)'''
        if match_nodes[-1].op == 'Relu':
          accel_function += '''
            val out = mux( L == num_fc_layers-1, prod_plus_bias, max( 0.to[T], prod_plus_bias ) )'''
        elif match_nodes[-1].op == 'Sigmoid':
          accel_function += '''
            val out = mux( L == num_fc_layers-1, prod_plus_bias, Sigmoid( prod_plus_bias ) )'''
        else:
          accel_function += '''
            val out = prod_plus_bias'''
        accel_function += '''
            if (L % 2 == 0) {
              data1_SRAM(r) = out
            } else {
              data0_SRAM(r) = out
            }
          }
        }
        // Optimization: BiasAdd was merged into MatMul above'''
        if match_nodes[-1].op == 'Relu':
          accel_function += '''
        // Optimization: ReLU was merged into MatMul above'''
        elif match_nodes[-1].op == 'Sigmoid':
          accel_function += '''
        // Optimization: Sigmoid was merged into MatMul above
        '''
        accel_function += "\n"
        file_opening += '  // FC' + "\n"
        file_opening += '  val OP_L' + str(global_val_number) + ' = 4' + "\n"
        file_opening += '  val IP_L' + str(global_val_number) + ' = 16' + "\n"
      
      # Add these weights to list
      reuse_fc_weight_dram[unique_op_name]['bias'].append(bias_tmpvar_inputs[1])
      reuse_fc_weight_dram[unique_op_name]['weights'].append(tmpvar_inputs[1])
      weight_files_to_concat.append(bias_tmpvar_inputs[1])
      weight_files_to_concat.append(tmpvar_inputs[1])      
      if final_node_SRAM == 'data1':
        final_node_SRAM = 'data0'
      else:
        final_node_SRAM = 'data1'
    
    # No FC reuse
    else:
      matrix_size = int(kernel_dims_str[0]) * int(kernel_dims_str[1])
      ip = 4
      op = 1
      # Currently using these heuristics but can also count FC ops in the static model, and if it 
      # is significant compared to conv (conv from DRAM used below, can be either) then use more par here
      if not conv_input_output_from_DRAM:
        if matrix_size > 10**5:
          ip = 16
          op = 2
        elif matrix_size > 10**4:
          ip = 8
          op = 1
        else:
          ip = 1
      file_opening += '  val OP_L' + str(global_val_number) + ' = ' + str(op) + "\n"
      file_opening += '  val IP_L' + str(global_val_number) + ' = ' + str(ip) + "\n"
      
      compute_ratio = ''
      #if ip >= 8:
      #  compute_ratio = '/2'
      if is_input:
        accel_function += '''        val ''' + tmpvar_inputs[0] + '''_SRAM = SRAM[T](''' + kernel_dims_str[0] + ''')
        ''' + tmpvar_inputs[0] + '''_SRAM load ''' + tmpvar_inputs[0] + '''_DRAM(0::''' + kernel_dims_str[0] + ')' + "\n"
      
      accel_function += '''        val ''' + bias_tmpvar_inputs[1] + '''_SRAM = SRAM[T](''' + str(burst_align(bias_size)) + ''')
        ''' + bias_tmpvar_inputs[1] + '''_SRAM load ''' + bias_tmpvar_inputs[1] + '''_DRAM(0::''' + str(burst_align(bias_size)) + ''')
        val ''' + tmpvar + '''_SRAM = SRAM[T](''' + str(burst_align(kernel_dims_str[1])) + ''')
        Foreach(''' + kernel_dims_str[1] + ''' by 1 par OP_L''' + str(global_val_number) + '''){out_i =>
          val ''' + tmpvar_inputs[1] + '''_row_SRAM = SRAM[T](''' + kernel_dims_str[0] + ''')
          ''' + tmpvar_inputs[1] + '''_row_SRAM load ''' + tmpvar_inputs[1] + '''_DRAM(out_i, 0::''' + kernel_dims_str[0] + ''' par IP_L''' + str(global_val_number) + ''')
          val prod = Reduce(Reg[T](0.to[T]))(''' + kernel_dims_str[0] + ''' by 1 par IP_L''' + str(global_val_number) + compute_ratio + '''){ in_i => ''' + tmpvar_inputs[0] + '''_SRAM(in_i) * ''' + tmpvar_inputs[1] + '''_row_SRAM(in_i) }{_+_}'''
      if match_nodes[-1].op == 'Relu':
        accel_function += '''
          ''' + tmpvar + '''_SRAM(out_i) = max(0.to[T], prod.value + ''' + bias_tmpvar_inputs[1] + '''_SRAM(out_i))'''
      elif match_nodes[-1].op == 'Sigmoid':
        accel_function += '''
          ''' + tmpvar + '''_SRAM(out_i) = Sigmoid(prod.value + ''' + bias_tmpvar_inputs[1] + '''_SRAM(out_i))'''
      else:
        accel_function += '''
          ''' + tmpvar + '''_SRAM(out_i) = prod.value + ''' + bias_tmpvar_inputs[1] + '''_SRAM(out_i)'''
      accel_function += '''
        }
        // Optimization: BiasAdd was merged into MatMul above'''
      if match_nodes[-1].op == 'Relu':
        accel_function += '''
        // Optimization: ReLU was merged into MatMul above'''
      elif match_nodes[-1].op == 'Sigmoid':
        accel_function += '''
        // Optimization: Sigmoid was merged into MatMul above'''
      accel_function += "\n"
      weight_mem_declarations_no_reuse += '    val ' + bias_tmpvar_inputs[1] + '_DRAM = DRAM[T](' + str(burst_align(bias_size)) + ')' + "\n"
      weight_mem_declarations_no_reuse += '    setMem(' + bias_tmpvar_inputs[1] + '_DRAM, ' + bias_tmpvar_inputs[1] + ')' + "\n"    
      final_node_SRAM = tmpvar
    
    name_to_tmpvar[match_nodes[-1].name] = tmpvar
    num_classes = bias_size
    final_node = tmpvar
      
  # -----------------------------------------------------
  # Other
  # -----------------------------------------------------
  else:
    accel_function += '        val ' + tmpvar + ' = ' + str(node.op) + '(' + ', '.join(tmpvar_inputs) + ')' + "\n"
  
  if fc_section:
    if not reuse_FC:
      accel_function += "\n"
  else:
    if not reuse:  
      accel_function += "\n"
  global_val_number += 1


# ========================================================================================================
# Assign Reused Data DRAM Buffers
# ========================================================================================================

# If there is at least 1 data DRAM in the reused hardware
if reuse and reuse_tmp_dram_ordered:
  # Assign outputs to locations in DRAM buffer space.
  # To release a parent, need to wait until all its consumers are done with it.
  tmp_DRAMs_in_use = []
  max_DRAM_assigned = 0
  def get_next_available_reused_DRAM():
    global tmp_DRAMs_in_use
    global max_DRAM_assigned
    next_available_DRAM = 0
    while True:
      if next_available_DRAM not in tmp_DRAMs_in_use:
        break
      next_available_DRAM += 1
    tmp_DRAMs_in_use.append(next_available_DRAM)
    # print 'ASSIGNING ' + str(next_available_DRAM)
    max_DRAM_assigned = max(max_DRAM_assigned, next_available_DRAM)
    return next_available_DRAM

  def release_reused_DRAM(id):
    global tmp_DRAMs_in_use
    # print 'RELEASING ' + str(id)
    tmp_DRAMs_in_use.remove(id)

  # Count the number of children
  num_children = {}
  for dram in reuse_tmp_dram_ordered:
    if dram in reuse_tmp_dram_children.keys():
      # print '#children of ' + dram + ' = ' + str(len(reuse_tmp_dram_children[dram]))
      num_children[dram] = len(reuse_tmp_dram_children[dram])

  # Assign DRAMs
  reuse_tmp_dram_id = {}
  for dram in reuse_tmp_dram_ordered:
    reuse_tmp_dram_id[dram] = get_next_available_reused_DRAM()
    # Now check if all your parents are done. If so, release them.
    for parent in reuse_tmp_dram_parents[dram]:
      if parent in num_children.keys():  # First DRAM not in tmp DRAM, so no need to release
        assert num_children[parent] >= 1
        num_children[parent] -= 1
        if num_children[parent] == 0:
          release_reused_DRAM(reuse_tmp_dram_id[parent])

  # Assert the input is in DRAM0
  assert reuse_tmp_dram_id[reuse_tmp_dram_ordered[0]] == 0
  
  # Now we have the assignments for all 
  # Next step is to create the LUTs for inputs and outputs
  reuse_layer_list_unique = set(reuse_layer_list)
  for layer in reuse_layer_list_unique:
    reuse_args[layer]['store_idx : Int'] = []
    if 'load_idx_0 : Int' in reuse_args[layer].keys():
      reuse_args[layer]['load_idx_0 : Int'] = []
    if 'load_idx_1 : Int' in reuse_args[layer].keys():
      reuse_args[layer]['load_idx_1 : Int'] = []
  idx = 0
  for layer in reuse_layer_list:
    # Get the DRAM for this layer
    # Skip the first one due to assertion above
    dram = reuse_tmp_dram_ordered[idx+1]
    idx += 1
    reuse_args[layer]['store_idx : Int'].append(str(reuse_tmp_dram_id[dram]))
    if 'load_idx_0 : Int' in reuse_args[layer].keys():
      reuse_args[layer]['load_idx_0 : Int'].append(str(reuse_tmp_dram_id[reuse_tmp_dram_parents[dram][0]]))
    if 'load_idx_1 : Int' in reuse_args[layer].keys():
      reuse_args[layer]['load_idx_1 : Int'].append(str(reuse_tmp_dram_id[reuse_tmp_dram_parents[dram][1]]))

  # Find the largest data dimension
  # For 3D tensors, concatenate rows/cols
  max_channels = 0
  max_rc = 0
  max_rc_x_channels = 0
  for dram in reuse_tmp_dram_ordered:
    dims = reuse_tmp_dram_dims[dram]
    max_channels = max(max_channels, int(dims[0]))
    max_rc = max(max_rc, int(dims[1])*int(dims[1]))
    max_rc_x_channels = max(max_rc_x_channels, int(dims[0])*int(dims[1])*int(dims[1]))
  # tmp_mem_declarations += "\n" + '    val tmp_DRAM = DRAM[T](' + str(max_DRAM_assigned+1) + ', ' + str(max_channels) + ', ' + str(max_rc) + ')' + "\n"
  tmp_mem_declarations += "\n" + '    val tmp_DRAM = DRAM[T](' + str(max_DRAM_assigned+1) + ', ' + str(max_rc_x_channels) + ')' + "\n"

# For non-reuse tmp DRAMs
tmp_mem_declarations += tmp_mem_declarations_no_reuse

# ========================================================================================================
# Global data structures used by above nodes
# ========================================================================================================

if include_sigmoid_initialization:
  accel_global_LUTs += '''      def Sigmoid(x : T) : T = {
        val sig = LUT[T](128)(
          0.01798621.to[T],  0.019124037.to[T], 0.020332353.to[T], 0.021615333.to[T], 0.02297737.to[T],  0.02442309.to[T],  0.025957357.to[T], 0.027585282.to[T], 
          0.029312231.to[T], 0.031143831.to[T], 0.033085978.to[T], 0.035144846.to[T], 0.037326887.to[T], 0.039638839.to[T], 0.042087728.to[T], 0.04468087.to[T], 
          0.047425873.to[T], 0.050330633.to[T], 0.05340333.to[T],  0.056652425.to[T], 0.06008665.to[T],  0.063714994.to[T], 0.067546691.to[T], 0.071591199.to[T], 
          0.07585818.to[T],  0.080357469.to[T], 0.085099045.to[T], 0.090092994.to[T], 0.095349465.to[T], 0.100878623.to[T], 0.106690594.to[T], 0.112795406.to[T], 
          0.119202922.to[T], 0.125922765.to[T], 0.13296424.to[T],  0.140336249.to[T], 0.148047198.to[T], 0.156104897.to[T], 0.164516463.to[T], 0.173288206.to[T], 
          0.182425524.to[T], 0.191932786.to[T], 0.201813222.to[T], 0.212068804.to[T], 0.222700139.to[T], 0.233706357.to[T], 0.245085013.to[T], 0.256831991.to[T], 
          0.268941421.to[T], 0.281405607.to[T], 0.294214972.to[T], 0.307358017.to[T], 0.320821301.to[T], 0.334589441.to[T], 0.348645135.to[T], 0.362969206.to[T], 
          0.377540669.to[T], 0.39233683.to[T],  0.4073334.to[T],   0.422504635.to[T], 0.437823499.to[T], 0.453261848.to[T], 0.468790627.to[T], 0.484380084.to[T], 
          0.5.to[T],         0.515619916.to[T], 0.531209373.to[T], 0.546738152.to[T], 0.562176501.to[T], 0.577495365.to[T], 0.5926666.to[T],   0.60766317.to[T], 
          0.622459331.to[T], 0.637030794.to[T], 0.651354865.to[T], 0.665410559.to[T], 0.679178699.to[T], 0.692641983.to[T], 0.705785028.to[T], 0.718594393.to[T], 
          0.731058579.to[T], 0.743168009.to[T], 0.754914987.to[T], 0.766293643.to[T], 0.777299861.to[T], 0.787931196.to[T], 0.798186778.to[T], 0.808067214.to[T], 
          0.817574476.to[T], 0.826711794.to[T], 0.835483537.to[T], 0.843895103.to[T], 0.851952802.to[T], 0.859663751.to[T], 0.86703576.to[T],  0.874077235.to[T], 
          0.880797078.to[T], 0.887204594.to[T], 0.893309406.to[T], 0.899121377.to[T], 0.904650535.to[T], 0.909907006.to[T], 0.914900955.to[T], 0.919642531.to[T], 
          0.92414182.to[T],  0.928408801.to[T], 0.932453309.to[T], 0.936285006.to[T], 0.93991335.to[T],  0.943347575.to[T], 0.94659667.to[T],  0.949669367.to[T], 
          0.952574127.to[T], 0.95531913.to[T],  0.957912272.to[T], 0.960361161.to[T], 0.962673113.to[T], 0.964855154.to[T], 0.966914022.to[T], 0.968856169.to[T], 
          0.970687769.to[T], 0.972414718.to[T], 0.974042643.to[T], 0.97557691.to[T],  0.97702263.to[T],  0.978384667.to[T], 0.979667647.to[T], 0.980875963.to[T]    
        )
        mux(x < -4.to[T], 0.to[T], mux(x > 4.to[T], 1.to[T], sig((x*16.to[T] - 0.5.to[T]).to[Int] + 64.to[Int])))
      }

'''

# ========================================================================================================
# Reused Weight DRAM
# ========================================================================================================

# For reuse weights
for layer in reuse_weight_dram.keys():
  for dram in reuse_weight_dram[layer].keys():
    # Concatenate all the weights into one large weight
    # First, get the number of dimensions of these DRAMs
    num_dims = None
    for const in reuse_weight_dram[layer][dram]:
      if num_dims:
        assert num_dims == len(tmpvar_to_reshape_string[const].split(','))
      else:
        num_dims = len(tmpvar_to_reshape_string[const].split(','))
    
    # Special-case: 1D weights (e.g. bias)
    if num_dims == 1:
    
      # Create file for concat weights
      array_name = layer + '_' + dram + '_concat'
      const_fname = weight_path + array_name + '.bin'
      read_fname = 'args(args.length-1) + "/' + array_name + '.bin"'
      const_file = open(const_fname, 'wb')  # Write in binary mode      
      start_idx_list = []
      running_total = 0
      for const in reuse_weight_dram[layer][dram]:
        # Calculate arg offset
        start_idx_list.append(str(running_total))
        running_total += int(tmpvar_to_reshape_string[const])
        # Get the const which needs to be written to disk
        node = constants_to_write_to_disk[const]
        const_file.write(node.attr['value'].tensor.tensor_content)
        dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
      const_file.close()
      
      # Now read this file in host
      weight_mem_declarations += "\n" + '    // Bias DRAM' + "\n"
      weight_mem_declarations += '    val ' + array_name + ' = loadBinary[Float](' + read_fname + \
        ').map{e => e.to[' + dtype + ']}' + "\n"
      weight_mem_declarations += '    val ' + array_name + '_DRAM = DRAM[T](' + str(running_total) + ')' + "\n"
      weight_mem_declarations += '    setMem(' + array_name + '_DRAM, ' + array_name + ')' + "\n"
      
      reuse_args[layer][dram + '_start_idx : Int'] = start_idx_list
    
    # Special-case: 2D weights (e.g. 1x1 conv)
    elif num_dims == 2:
      # Make a file for each 2nd dim
      inner_dims = []
      for const in reuse_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        if dims[1] not in inner_dims:
          inner_dims.append(dims[1])
      
      const_files = {}
      file_sizes = {}
      for inner_dim in inner_dims:
        array_name = layer + '_' + dram + '_' + inner_dim + '_concat'
        const_fname = weight_path + array_name + '.bin'
        const_file = open(const_fname, 'wb')  # Write in binary mode      
        const_files[inner_dim] = const_file
        file_sizes[inner_dim]  = 0
          
      # Write each file
      for const in reuse_weight_dram[layer][dram]:
        # Get the const which needs to be written to disk
        dims = tmpvar_to_reshape_string[const].split(',')
        node = constants_to_write_to_disk[const]
        inner_dim = dims[1]
        
        import numpy as np
        # Default storage is [k, k, inCh, outCh], so invert this (transpose) to [outCh, inCh, k, k]
        const_files[inner_dim].write(np.transpose(tensor_util.MakeNdarray(node.attr['value'].tensor)).tobytes())
        
        file_sizes[inner_dim] += int(dims[0])
        dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
      
      # Close each file
      weight_mem_declarations += "\n" + '    // Weight DRAM' + "\n"
      for inner_dim in inner_dims:
        # Read this file in host
        array_name = layer + '_' + dram + '_' + inner_dim + '_concat'
        read_fname = 'args(args.length-1) + "/' + array_name + '.bin"'
        weight_mem_declarations += '    val ' + array_name + ' = loadBinary[Float](' + read_fname + \
          ').map{e => e.to[' + dtype + ']}.reshape(' + str(file_sizes[inner_dim]) + ',' + inner_dim + ')' + "\n"
        const_files[inner_dim].close()
      
      inner_dims_sorted = sorted(map(int, inner_dims))
      
      # Concat these weights into a single, larger array then generate the LUT arg offsets
      # Also get the max inner dim and sum of outer dims
      start_idx_list = []
      inner_dim_to_running_total = {}
      sum_outer_dims = 0
      max_inner_dim  = 0
      for inner_dim in inner_dims_sorted:
        inner_dim_to_running_total[str(inner_dim)] = sum_outer_dims
        sum_outer_dims += file_sizes[str(inner_dim)]
        max_inner_dim = max(max_inner_dim, inner_dim)
      for const in reuse_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        inner_dim = dims[1]
        start_idx_list.append(str(inner_dim_to_running_total[inner_dim]))
        inner_dim_to_running_total[inner_dim] += int(dims[0])
      reuse_args[layer][dram + '_start_idx : Int'] = start_idx_list
      
      array_name = layer + '_' + dram + '_concat'
      weight_mem_declarations += '''    val ''' + array_name + '''_DRAM = DRAM[T](''' + str(sum_outer_dims) + ''',''' + str(max_inner_dim) + ''')
    val ''' + array_name + '''_host = (0::''' + str(sum_outer_dims) + ''', 0::''' + str(max_inner_dim) + ''') { (i,j) =>
'''
      first_iter = True
      last_iter = False
      sum_outer_dims = 0
      for inner_dim in inner_dims_sorted:
        if inner_dim == max_inner_dim:
          last_iter = True
        if first_iter and last_iter:
          weight_mem_declarations += ''
        elif first_iter:
          weight_mem_declarations += '      if'
        elif last_iter:
          weight_mem_declarations += ' else '
        else:
          weight_mem_declarations += ' else if'
        
        if first_iter and last_iter:
          weight_mem_declarations += '''
          ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(i-(''' + str(sum_outer_dims) + '''), j)'''
        elif last_iter:
          weight_mem_declarations += ''' {
          ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(i-(''' + str(sum_outer_dims) + '''), j)
      }'''
        else:
          weight_mem_declarations += ''' (i < ''' + str(sum_outer_dims + file_sizes[str(inner_dim)]) + ''') {
        if (j < ''' + str(inner_dim) + ''') {
          ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(i-(''' + str(sum_outer_dims) + '''), j)
        } else {
          0.to[T]
        }
      }'''
        sum_outer_dims += file_sizes[str(inner_dim)]
        if first_iter:
          first_iter = False
      weight_mem_declarations += '''
    }
    setMem(''' + array_name + '''_DRAM, ''' + array_name + '''_host)
'''   
    
    # Special-case: 3D weights (e.g. k x k conv, concatenated last 2)
    elif num_dims == 3:
      
      # Make a file for each 2nd dim (in_channels)
      inner_dims = []
      outer_dims = []
      for const in reuse_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        if dims[1] not in inner_dims:
          inner_dims.append(dims[1])
        if dims[0] not in outer_dims:
          outer_dims.append(dims[0])
        kernel_size = dims[2]
      
      # We also need to map the number of in channels to the number of out channels.
      in_to_out_map = {}
      for const in reuse_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        inner_dim = dims[1]
        outer_dim = dims[0]
        if inner_dim not in in_to_out_map.keys():
          in_to_out_map[inner_dim] = outer_dim
        elif in_to_out_map[inner_dim] != outer_dim:
          # In this case, rather than a constant number of out channels for a given number
          # of in channels, it will vary. This means the reshaping on the host needs to keep
          # track of the number of out channels per weight tensor. All this information exists
          # in the current script, but not in the generated scala host. So instead of reshaping
          # there, the reshape would happen here.
          utils.error_exit('Mismatching in/out channels is currently unsupported for k>1, but not hard to add. Please contact the code author.')
      
      inner_dim_to_cumulative_outer_dims = {}
      for inner_dim in inner_dims:
        inner_dim_to_cumulative_outer_dims[inner_dim] = 0
      inner_dim_to_cumulative_inner_dims = {}
      for inner_dim in inner_dims:
        inner_dim_to_cumulative_inner_dims[inner_dim] = 0
      outer_dim_to_cumulative_inner_dims = {}
      for outer_dim in outer_dims:
        outer_dim_to_cumulative_inner_dims[outer_dim] = 0
      
      const_files = {}
      for inner_dim in inner_dims:
        array_name = layer + '_' + dram + '_' + inner_dim + '_concat'
        const_fname = weight_path + array_name + '.bin'
        const_file = open(const_fname, 'wb')  # Write in binary mode      
        const_files[inner_dim] = const_file
          
      # Write each file
      for const in reuse_weight_dram[layer][dram]:
        # Get the const which needs to be written to disk
        dims = tmpvar_to_reshape_string[const].split(',')
        node = constants_to_write_to_disk[const]
        outer_dim = dims[0]
        inner_dim = dims[1]
        inner_dim_to_cumulative_outer_dims[inner_dim] += int(outer_dim)
        inner_dim_to_cumulative_inner_dims[inner_dim] += int(inner_dim)
        outer_dim_to_cumulative_inner_dims[outer_dim] += int(inner_dim)
        import numpy as np
        # Default storage is [k, k, inCh, outCh], so invert this (transpose) to [outCh, inCh, k, k]
        const_files[inner_dim].write(np.transpose(tensor_util.MakeNdarray(node.attr['value'].tensor), [3, 2, 0, 1]).tobytes())        
        dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
      
      # Close each file
      weight_mem_declarations += "\n" + '    // Weight DRAM' + "\n"
      for inner_dim in inner_dims:
        # Read this file in host
        array_name = layer + '_' + dram + '_' + inner_dim + '_concat'
        read_fname = 'args(args.length-1) + "/' + array_name + '.bin"'
        weight_mem_declarations += '    val ' + array_name + ' = loadBinary[Float](' + read_fname + \
          ').map{e => e.to[' + dtype + ']}.reshape(' + str(inner_dim_to_cumulative_outer_dims[inner_dim]) + ',' + inner_dim + ',' + kernel_size + ')' + "\n"
        const_files[inner_dim].close()
      
      outer_dims_sorted = sorted(map(int, outer_dims))
      inner_dims_sorted = sorted(map(int, inner_dims))
      
      # Now concat these weights into a single, larger array, and then generate the LUT arg offsets
      # Also get the max outer dim and sum of inner dims
      start_idx_list = []
      outer_dim_to_running_total = {}
      sum_inner_dims = 0
      max_inner_dim  = 0
      for inner_dim in inner_dims_sorted:
        max_inner_dim = max(max_inner_dim, inner_dim)
      max_outer_dim  = 0
      for outer_dim in outer_dims_sorted:
        outer_dim_to_running_total[str(outer_dim)] = sum_inner_dims
        sum_inner_dims += outer_dim_to_cumulative_inner_dims[str(outer_dim)]
        max_outer_dim = max(max_outer_dim, outer_dim)
      for const in reuse_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        outer_dim = dims[0]
        start_idx_list.append(str(outer_dim_to_running_total[outer_dim]))
        outer_dim_to_running_total[outer_dim] += int(dims[1])
      reuse_args[layer][dram + '_start_idx : Int'] = start_idx_list      
      
      array_name = layer + '_' + dram + '_concat'
      weight_mem_declarations += '''    val ''' + array_name + '''_DRAM = DRAM[T](''' + str(sum_inner_dims) + ''',''' + str(max_outer_dim) + '*' + kernel_size + ''')
    val ''' + array_name + '''_host = (0::''' + str(sum_inner_dims) + ''', 0::''' + str(max_outer_dim) + '*' + kernel_size + ''') { (i,j) =>
'''
      first_iter = True
      last_iter = False
      sum_inner_dims = 0
      for inner_dim in inner_dims_sorted:
        if inner_dim == max_inner_dim:
          last_iter = True
        if first_iter and last_iter:
          weight_mem_declarations += ''
        elif first_iter:
          weight_mem_declarations += '      if'
        elif last_iter:
          weight_mem_declarations += ' else '
        else:
          weight_mem_declarations += ' else if'
        
        if first_iter and last_iter:
          weight_mem_declarations += '''
        val i_range = i-(''' + str(sum_inner_dims) + ''')
        val in_channel = i_range%''' + str(inner_dim) + '''
        val out_channel = j/(''' + kernel_size + ''') + (i_range/''' + str(inner_dim) + ''')*''' + in_to_out_map[str(inner_dim)] + '''
        val kernel = j%(''' + kernel_size + ''')
        ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(out_channel, in_channel, kernel)'''
          # if/else 0 can be added here too, but not necessary
        elif last_iter:
          weight_mem_declarations += ''' {
        val i_range = i-(''' + str(sum_inner_dims) + ''')
        val in_channel = i_range%''' + str(inner_dim) + '''
        val out_channel = j/(''' + kernel_size + ''') + (i_range/''' + str(inner_dim) + ''')*''' + in_to_out_map[str(inner_dim)] + '''
        val kernel = j%(''' + kernel_size + ''')
        ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(out_channel, in_channel, kernel)
      }'''
          # if/else 0 can be added here too, but not necessary
        else:
          weight_mem_declarations += ''' (i < ''' + str(sum_inner_dims + inner_dim_to_cumulative_inner_dims[str(inner_dim)]) + ''') {
        val i_range = i-(''' + str(sum_inner_dims) + ''')
        val in_channel = i_range%''' + str(inner_dim) + '''
        val out_channel = j/(''' + kernel_size + ''') + (i_range/''' + str(inner_dim) + ''')*''' + in_to_out_map[str(inner_dim)] + '''
        val kernel = j%(''' + kernel_size + ''')
        if (j/(''' + kernel_size + ''') < ''' + str(in_to_out_map[str(inner_dim)]) + ''') {
          ''' + layer + '_' + dram + '_' + str(inner_dim) + '_concat' + '''(out_channel, in_channel, kernel)
        } else {
          0.to[T]
        }
      }'''
        sum_inner_dims += inner_dim_to_cumulative_inner_dims[str(inner_dim)]
        if first_iter:
          first_iter = False
      weight_mem_declarations += '''
    }
    setMem(''' + array_name + '''_DRAM, ''' + array_name + '''_host)
'''   
    
    else:
      weight_mem_declarations += "\n" + '    // Create ' + layer + '_' + dram + '_DRAM of ' + str(num_dims) + ' dims by concatenating: ' + str(reuse_weight_dram[layer][dram])
      weight_mem_declarations += "\n" + '    // Dims of above:'
      for const in reuse_weight_dram[layer][dram]:
        weight_mem_declarations += "\n" + '    //     ' + const + ': ' + tmpvar_to_reshape_string[const]
      weight_mem_declarations += "\n"
    
weight_mem_declarations += "\n"

# For non-reuse weights
for const in constants_to_write_to_disk.keys():
  if const not in weight_files_to_concat:
    # Get the const which needs to be written to disk
    node = constants_to_write_to_disk[const]
    input_sizes = get_const_dims(node)
    # Now write this to file
    const_fname = weight_path + const + '.bin'
    read_fname = 'args(args.length-1) + "/' + const + '.bin"'
    # Reformat if necessary
    if len(input_sizes) == 4:
      import numpy as np
      # Default storage is [k, k, inCh, outCh], so invert this (transpose) to [outCh, inCh, k, k]
      np.transpose(tensor_util.MakeNdarray(node.attr['value'].tensor), [3, 2, 0, 1]).tofile(const_fname)
    elif len(input_sizes) == 2:
      import numpy as np
      # Default storage is [k, k, inCh, outCh], so invert this (transpose) to [outCh, inCh, k, k]
      np.transpose(tensor_util.MakeNdarray(node.attr['value'].tensor)).tofile(const_fname)
    else:
      const_file = open(const_fname, 'wb')
      const_file.write(node.attr['value'].tensor.tensor_content)
      const_file.close()

    dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
    weight_mem_declarations += '    val ' + const + ' = loadBinary[Float](' + read_fname + \
      ').map{e => e.to[' + dtype + ']}'
    weight_mem_declarations += ' // ' + node.name
    weight_mem_declarations += "\n"
  
weight_mem_declarations += weight_mem_declarations_no_reuse


# ========================================================================================================
# Reused FC Weight DRAM
# ========================================================================================================

if reuse_FC:
  layer = reuse_FC_name
  max_bias_size = 0
  max_cols = 0
  for dram in reuse_fc_weight_dram[layer].keys():
    
    # Concatenate the weights
    # First, get the number of dimensions of these DRAMs
    num_dims = None
    for const in reuse_fc_weight_dram[layer][dram]:
      if num_dims:
        assert num_dims == len(tmpvar_to_reshape_string[const].split(','))
      else:
        num_dims = len(tmpvar_to_reshape_string[const].split(','))
    
    # Special-case: 1D weights (e.g. bias)
    if num_dims == 1:
    
      # Create file for concat weights
      array_name = layer + '_' + dram + '_concat'
      const_fname = weight_path + array_name + '.bin'
      read_fname = 'args(args.length-1) + "/' + array_name + '.bin"'
      const_file = open(const_fname, 'wb')  # Write in binary mode      
      start_idx_list = []
      running_total = 0
      for const in reuse_fc_weight_dram[layer][dram]:
        # Calculate arg offset
        start_idx_list.append(str(running_total))
        running_total += int(tmpvar_to_reshape_string[const])
        # Get the const which needs to be written to disk
        node = constants_to_write_to_disk[const]
        const_file.write(node.attr['value'].tensor.tensor_content)
        dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
        max_bias_size = max(max_bias_size, int(tmpvar_to_reshape_string[const]))
      const_file.close()
      
      # Now read this file in host
      weight_mem_declarations += '    // Bias DRAM' + "\n"
      weight_mem_declarations += '    val ' + array_name + ' = loadBinary[Float](' + read_fname + \
        ').map{e => e.to[' + dtype + ']}' + "\n"
      weight_mem_declarations += '    val ' + array_name + '_DRAM = DRAM[T](' + str(running_total) + ')' + "\n"
      weight_mem_declarations += '    setMem(' + array_name + '_DRAM, ' + array_name + ')' + "\n"
      
      # LUT of bias start idx
      accel_global_LUTs += '      val bias_offset     = LUT[Int](' + str(len(start_idx_list)) + ')( ' + ', '.join(start_idx_list) + ' )' + "\n"
      accel_global_LUTs += "\n"
    
    # Special-case: 2D weights (matmul)
    elif num_dims == 2:    
          
      # Write weights to file
      array_name = layer + '_' + dram + '_concat'
      const_fname = weight_path + array_name + '.bin'
      read_fname = 'args(args.length-1) + "/' + array_name + '.bin"'
      const_file = open(const_fname, 'wb')  # Write in binary mode              
      max_abs_value = 0
      for const in reuse_fc_weight_dram[layer][dram]:
        # Get the const which needs to be written to disk
        node = constants_to_write_to_disk[const]
        import numpy as np
        # Default storage is [inCh, outCh], so invert this (transpose) to [outCh, inCh]
        weight_array = tensor_util.MakeNdarray(node.attr['value'].tensor)
        max_abs_value = max(max_abs_value, np.abs(weight_array).max())
        const_file.write(np.transpose(weight_array).tobytes())
        dtype = proto_type_to_spatial_type[node.attr['value'].tensor.dtype]
      const_file.close()
      
      # Keep track of sizes
      start_idx_list = []
      rows_list      = []
      cols_list      = []
      running_total = 0
      for const in reuse_fc_weight_dram[layer][dram]:
        dims = tmpvar_to_reshape_string[const].split(',')
        cols_list.append(dims[1])
        rows_list.append(dims[0])
        start_idx_list.append(str(running_total))
        running_total += int(dims[0])*int(dims[1])
        max_cols = max(max_cols, int(dims[1]))
      
      # LUTs
      accel_global_LUTs += '      val weights_offset = LUT[Int](' + str(len(start_idx_list)) + ')( ' + ', '.join(start_idx_list) + ' )' + "\n"
      accel_global_LUTs += '      val weight_rows    = LUT[Int](' + str(len(rows_list))      + ')( ' + ', '.join(rows_list)      + ' )' + "\n"
      accel_global_LUTs += '      val weight_cols    = LUT[Int](' + str(len(cols_list))      + ')( ' + ', '.join(cols_list)      + ' )' + "\n"
      accel_global_LUTs += "\n"
      
      # FC weights are large in size, so if range is small reduce precision
      import numpy
      import math
      num_integer_bits = int(math.ceil(numpy.log2(max(1,max_abs_value+0.0001)))) + 1 # +1 for sign
      if num_integer_bits < 6:
        dtype = 'T2'
        file_opening += '''
  type T2 = FixPt[TRUE,_''' + str(num_integer_bits) + ''',_''' + str(16-num_integer_bits) + ''']
'''
        
      # Read this file in host
      weight_mem_declarations += "\n" + '    // Weight DRAM' + "\n"
      weight_mem_declarations += '    val ' + array_name + ' = loadBinary[Float](' + read_fname + \
        ').map{e => e.to[' + dtype + ']}' + "\n"
      weight_mem_declarations += '''    val ''' + array_name + '''_DRAM = DRAM[''' + dtype + '''](''' + str(running_total) + ''')
    setMem(''' + array_name + '''_DRAM, ''' + array_name + ''')'''   

  accel_global_LUTs += '      val num_fc_layers      = ' + str(len(reuse_fc_weight_dram[layer]['bias'])) + "\n"
  accel_global_LUTs += '      val max_fc_bias_size   = ' + str(max_bias_size) + "\n"
  accel_global_LUTs += '      val max_fc_weight_cols = ' + str(max_cols) + "\n\n"
  weight_mem_declarations += "\n"


# ========================================================================================================
# Print Reuse loop
# ========================================================================================================

# There are 4 optimizations that can be made here for class size:
#  1. Move LUTs inside the check, or e.g. for only LUTs used in only one place
#     - this is done below for LUTs used in 1 def
#     - can extend to LUTs used in 2, or just always move LUTs inside the check
#  2. Omit last check and use else {}
#     - this is done below
#  3. If 2 LUTs are the same, reuse the same one twice
#     - this is not done yet
#  4. If a LUT is always optimized to a const, delete the LUT
#     - this is not done yet
#     - in Spatial: optimizing unused LUTs away was not done by Spatial automatically as 
#       of April 2019, i.e. an unread LUT still contributed to class size

# For reuse, call each def
if reuse and reuse_layer_list:  # If there is at least one reused processor
  reuse_loop = ''
  num_layers = len(reuse_layer_list)
  reuse_layer_list_unique = set(reuse_layer_list)
  
  # Make LUTs for when to call each def
  count = 0
  for layer in reuse_layer_list_unique:
    check_LUT = ['false']*num_layers
    for state in reuse_schedule[layer]:
      check_LUT[int(state)] = 'true'
    count += 1
    if count == len(reuse_layer_list_unique):
      reuse_loop += '''
        // val check_''' + layer + ''' = LUT[Bit](''' + str(num_layers) + ''')(''' +  \
        ', '.join(check_LUT) + ''')'''
    else:
      reuse_loop += '''
        val check_''' + layer + ''' = LUT[Bit](''' + str(num_layers) + ''')(''' +  \
        ', '.join(check_LUT) + ''')'''
  reuse_loop += "\n"
  
  # Now print arg LUTs
  args_processed = []
  arg_map_combined = {}
  arg_types = {}
  optimize_arg_to_const = {}
  default_arg_type = { 'Int' : '-1', 'Boolean' : 'false' }
  for layer in reuse_layer_list_unique:
    optimize_arg_to_const[layer] = {}
    for arg in reuse_args[layer].keys():
      dtype = arg.split(':')[-1].strip()
      # Deal with MAX later, since those can be optimized to be constants
      if 'MAX__' in arg:
        assert dtype == 'Int'
        optimize_arg_to_const[layer][arg] = str(max(map(int, reuse_args[layer][arg])))
        continue
      # Next check if this always has the same value
      elif len(set(reuse_args[layer][arg])) == 1:
        optimize_arg_to_const[layer][arg] = reuse_args[layer][arg][0]
        continue
      if arg not in args_processed:
        args_processed.append(arg)
        arg_map_combined[arg] = [default_arg_type[dtype]]*num_layers
        arg_types[arg] = dtype
      for idx, value in enumerate(reuse_args[layer][arg]):
        state = int(reuse_schedule[layer][idx])
        assert arg_map_combined[arg][state] == default_arg_type[arg_types[arg]]
        arg_map_combined[arg][state] = reuse_args[layer][arg][idx]
        
  # Class size optimization: find args only used by one layer
  args_used_by_one_layer = {}
  for arg in args_processed:
    num_layers_with_this_arg = 0
    layer_with_arg = None
    for layer in reuse_layer_list_unique:
      # If this arg is only used in one layer, excluding layers where it is optimized to
      # a constant, then move it inside the check
      if arg in reuse_args[layer].keys() and arg not in optimize_arg_to_const[layer].keys():
        layer_with_arg = layer
        num_layers_with_this_arg += 1
    if num_layers_with_this_arg == 1:
      assert layer_with_arg
      args_used_by_one_layer[arg] = layer_with_arg
  
  for arg in arg_map_combined.keys():
    if arg not in args_used_by_one_layer.keys():
      reuse_loop += '''
        val ''' + arg.split(':')[0].strip() + '''_args = LUT[''' + arg_types[arg] + '''](''' + str(num_layers) + ''')(''' +  \
        ', '.join(arg_map_combined[arg]) + ''')'''
  reuse_loop += "\n"

  # Print loop with each state
  assert '{{{INSERT_REUSE_LOOP_HERE}}}' in accel_function
  reuse_loop += '''
        Sequential.Foreach(0 until ''' + str(num_layers) + ''') { L =>'''
  count = 0
  for layer in reuse_layer_list_unique:
    arg_list_LUTs = []
    for arg in sorted(reuse_args[layer].keys()):
      if arg in optimize_arg_to_const[layer].keys():
        arg_list_LUTs.append(optimize_arg_to_const[layer][arg])
      else:
        arg_list_LUTs.append(arg.split(':')[0].strip() + '_args(L)')
    count += 1
    if len(reuse_layer_list_unique) == 1:
      reuse_loop += '''
          if (true) {'''
    elif count == 1:
      reuse_loop += '''
          if (check_''' + layer + '''(L)) {'''
    elif count == len(reuse_layer_list_unique):
      reuse_loop += '''
          } else { // if (check_''' + layer + '''(L)) {'''
    else:
      reuse_loop += '''
          } else if (check_''' + layer + '''(L)) {'''
    for arg in args_used_by_one_layer.keys():
      if args_used_by_one_layer[arg] == layer:
        reuse_loop += '''
            val ''' + arg.split(':')[0].strip() + '''_args = LUT[''' + arg_types[arg] + '''](''' + str(num_layers) + ''')(''' +  \
        ', '.join(arg_map_combined[arg]) + ''')'''
    reuse_loop += '''
            ''' + layer + '''(''' + ', '.join(arg_list_LUTs) + ''', L)'''
  reuse_loop += '''
          }
        }
'''
  accel_function = accel_function.replace('{{{INSERT_REUSE_LOOP_HERE}}}', reuse_loop)

  # If store_idx LUT was optimized away, replace accesses to it with the corresponding constants
  store_arg = 'store_idx : Int'
  if (store_arg not in arg_map_combined.keys()) or (store_arg in args_used_by_one_layer.keys()):
    store_LUT = [0]*num_layers
    for layer in reuse_layer_list_unique:
      if store_arg in reuse_args[layer].keys():
        for idx, value in enumerate(reuse_args[layer][store_arg]):
          state = int(reuse_schedule[layer][idx])
          store_LUT[state] = reuse_args[layer][store_arg][idx]
    for lut_idx in range(num_layers):
      match_string = 'store_idx_args(' + str(lut_idx) + ')'
      if match_string in accel_function:
        new_store_idx = str(store_LUT[lut_idx]) + '.to[Int]'
        accel_function = accel_function.replace(match_string, new_store_idx)

# ========================================================================================================
# Static Model for DSP Assignment
# ========================================================================================================
if reuse_layer_to_ops.keys():
  total_ops = 0
  max_name_length = 0
  for layer in reuse_layer_to_ops.keys():
    total_ops += reuse_layer_to_ops[layer]
    max_name_length = max(max_name_length, len(layer))
  print
  
  dsp_util_target = 1.0
  mults_to_assign = int( dsp_util_target * float(device_params['num_dsps']) / float( device_params['dsp_usage_per_32b_mul'] ) )
  
  print 'MAC % by op:'
  for layer in reuse_layer_to_ops.keys():
    spaces = ' '*(max_name_length - len(layer)) + ' : '
    percent_for_op = float(reuse_layer_to_ops[layer])/float(total_ops)
    print '  ' + layer + spaces + str(100.*percent_for_op)[0:5] + '%  (' + str(reuse_layer_to_ops[layer]) + ' / ' + str(total_ops) + ')'
    if layer in reuse_layer_to_IP.keys() and layer in reuse_layer_to_kxk.keys():
      # Spatial currently has JVM code size errors for large outer pars. The line below should be used here:
      # par_alloc = str(closest_pow_2(mults_to_assign * percent_for_op / reuse_layer_to_IP[layer] / reuse_layer_to_kxk[layer]))
      # But until that is fixed, limiting the outer par to 8 here:
      par_alloc = str(min(8, closest_pow_2(mults_to_assign * percent_for_op / reuse_layer_to_IP[layer] / reuse_layer_to_kxk[layer])))
      file_opening = file_opening.replace(layer + '_OP', par_alloc)
  print
  

# ========================================================================================================
# Close host code block
# ========================================================================================================

host_after_accel = '''
    val output = getMem(''' + final_node + '''_DRAM)'''
# "Some ResNet models represent 1000 categories, and some represent all 1001, with
#  the 0th category being 'background'."
if int(num_classes) == 1001:
  host_after_accel += '''
    val output_no_extra = Array.tabulate(''' + str(1000) + '''){i => output(i+1)}'''
else:
  host_after_accel += '''
    val output_no_extra = Array.tabulate(''' + str(num_classes) + '''){i => output(i)}'''
host_after_accel += '''
    printArray(output_no_extra, "output")
    /* Used for debugging
    val gold = loadCSV1D[T]("data_out.csv", "\\n")
    printArray(gold, "gold")
    val margin = 0.005.to[T]
  	val cksum = gold.zip(output_no_extra){(a,b) => abs(a-b) < margin}.reduce{_&&_}
  	println("PASS: " + cksum)
    */
'''

if include_imagenet_classification:
  host_after_accel  += '''
    println("\\n")
    val imagenet_classes = loadCSV1D[String](args(args.length-2), "\\n")
    println("Top 5 Predictions: ")
    val zipped_with_idx__1 = Array.tabulate(1000){i => pack(output_no_extra(i),i) }
    val max_idx__1         = zipped_with_idx__1.reduce{(a,b) => if (a._1 > b._1) a else b}._2
    val zipped_with_idx__2 = Array.tabulate(1000){i => if (i == max_idx__1) pack(-1.to[T],i) else pack(output_no_extra(i),i) }
    val max_idx__2         = zipped_with_idx__2.reduce{(a,b) => if (a._1 > b._1) a else b}._2
    val zipped_with_idx__3 = Array.tabulate(1000){i => if (i == max_idx__1 || i == max_idx__2) pack(-1.to[T],i) else pack(output_no_extra(i),i) }
    val max_idx__3         = zipped_with_idx__3.reduce{(a,b) => if (a._1 > b._1) a else b}._2
    val zipped_with_idx__4 = Array.tabulate(1000){i => if (i == max_idx__1 || i == max_idx__2 || i == max_idx__3) pack(-1.to[T],i) else pack(output_no_extra(i),i) }
    val max_idx__4         = zipped_with_idx__4.reduce{(a,b) => if (a._1 > b._1) a else b}._2
    val zipped_with_idx__5 = Array.tabulate(1000){i => if (i == max_idx__1 || i == max_idx__2 || i == max_idx__3 || i == max_idx__4) pack(-1.to[T],i) else pack(output_no_extra(i),i) }
    val max_idx__5         = zipped_with_idx__5.reduce{(a,b) => if (a._1 > b._1) a else b}._2
    println("  1. " + imagenet_classes(max_idx__1))
    println("  2. " + imagenet_classes(max_idx__2))
    println("  3. " + imagenet_classes(max_idx__3))
    println("  4. " + imagenet_classes(max_idx__4))
    println("  5. " + imagenet_classes(max_idx__5))
    println("")
'''

file_closing    = '''  }
}
'''

# ========================================================================================================
# Write to file
# ========================================================================================================

# Assemble file string
spatial_output = ''
spatial_output += file_opening + "\n"
spatial_output += var_declarations
spatial_output += host_before_accel
spatial_output += data_mem_declarations + data_mem_set
spatial_output += tmp_mem_declarations
spatial_output += weight_mem_declarations
spatial_output += '''
    Accel {

''' + accel_global_LUTs + accel_defs + '''//    Sequential.Foreach(batch_size by 1) { batch_idx =>
''' + accel_function + '''//    } Sequential over all images
    }
'''
spatial_output += host_after_accel
spatial_output += file_closing

# File string now complete, write to file
output_fname = app_name + '.scala'
f = open(output_fname, 'w')
f.write(spatial_output + "\n")
f.close()
print 'Output written to ' + output_fname
print 'Compile and synthesize in Spatial by placing this file in spatial/test/spatial/tests/apps/'
print 'and running the following command in the spatial-dnn directory:'
if use_relu6:
  print '  $ bin/spatial ' + app_name + ' --synth --noBindParallels --fpga=AWS_F1 && cd gen/' + app_name + ' && make aws-F1'
else:
  print '  $ bin/spatial ' + app_name + ' --synth --forceFuseFMA --noBindParallels --fpga=AWS_F1 && cd gen/' + app_name + ' && make aws-F1'
print 
print 'The format for inputs is currently .csv. E.g. to convert an image file to .csv, use data/img_to_csv.py'

# Close the tf session with the frozen graph
frz_sess.close()
