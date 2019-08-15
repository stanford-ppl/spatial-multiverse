# This example contains the newer implementation of ResNet released by the TensorFlow team
# https://github.com/tensorflow/models/tree/master/official/resnet

# Download and extract the model
# Could use saved model or checkpoint, here we show saved model
wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NHWC.tar.gz
tar -zxvf resnet_v1_fp32_savedmodel_NHWC.tar.gz
