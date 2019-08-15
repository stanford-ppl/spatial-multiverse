This is the LeNet example from TensorFlow 1.12, found [here](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/examples/tutorials/mnist/mnist_deep.py).
TensorFlow 1.13 has a very similar example, see [here](https://www.tensorflow.org/tutorials/estimators/cnn).
The model in this directory is already optimized for inference. To run the example:

1. Generate the Spatial program using `dnn_to_spatial.py` on lenet.pb
2. Follow the instructions printed by the script to compile through Spatial
3. Once Spatial compilation finishes, follow these [instructions](../../docs/aws.md) to load the generated AFI to your EC2 F1 instance and run the inference. For instructions to target other devices supported by Spatial, see [here](https://spatial-lang.org/targetting-devices).
