# ------------------------------------------------------------------------------
# 
# Utility Functions for scripts
# 
# ------------------------------------------------------------------------------

import sys

def get_args(argc, err_msg):
  if argc == 0:
    return None
  if len(sys.argv) != argc + 1:
    print 'Script usage:   $ python ' + err_msg
    sys.exit(0)
  return sys.argv[1:]

def error_exit(msg):
  print "\n" + 'Message for user:' + "\n  " + msg + "\n" + 'Exiting' + "\n"
  sys.exit(0)

def read_config_file(fname):
  print 'Reading config file: ' + fname
  f = open(fname)
  key_to_value = {}
  for line in f:
    line = line.strip()
    try:
      key = line.split()[0]
    except:
      continue
    try:
      val = int(line.split()[1])
    except:
      val = None
    key_to_value[key] = val
  f.close()
  return key_to_value

# Write a summary file of an imported graph
def write_summary_file(name, sess, graph_def, imported=False):

  f = open(name, 'w')
  for node in graph_def.node:
    f.write(node.name + "\n")
    prefix = ''
    if imported:
      prefix = 'import/'
    if node.op not in ['NoOp', 'SaveV2']:
      f.write('  op = ' + node.op + "\n")
      try:
        f.write('  output size = ' + str(sess.graph.get_tensor_by_name(prefix + node.name + ':0').get_shape()) + "\n")
      except:
        f.write('  output size = 0' + "\n")
      if node.op in ['Conv2D', 'DepthwiseConv2dNative']:
        f.write('  padding = ' + str(node.attr['padding'].s) + "\n")
        f.write('  stride = ' + str(node.attr['strides'].list.i) + "\n")
        # f.write('  stride = ' + str(node.attr['strides'].list.i[1]) + "\n")
        # f.write('  k = ' + str(sess.graph.get_tensor_by_name(prefix + node.input[1] + ':0').get_shape().as_list()[1]) + "\n")
        f.write('  k = ' + str(sess.graph.get_tensor_by_name(prefix + node.input[1] + ':0').get_shape()) + "\n")
      elif node.op == 'MaxPool':
        f.write('  stride = ' + str(node.attr['strides'].list.i) + "\n")
        f.write('  k = ' + str(node.attr['ksize'].list.i) + "\n")
      input_idx = 0
      for input in node.input:
        f.write('  in' + str(input_idx) + ' = ' + str(input) + "\n")
        input_idx += 1
  f.close()  
