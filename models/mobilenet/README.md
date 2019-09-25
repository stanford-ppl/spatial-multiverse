This directory contains steps to compile the MobileNet example provided by TensorFlow Slim.

Note: This example requires newer versions of the TensorFlow graph optimizer which folds BN into depthwise convolution.
We tested with TensorFlow 1.14.

1. Download the model from [here](https://github.com/tensorflow/models/tree/master/research/slim). We used `mobilenet_v1_1.0_224`:

```
bash get_mobilenet.sh
```

2. Run the optimizer. The input and output nodes are listed in the archive downloaded above.

```
python  optimize_inference_graph.py  models/mobilenet/mobilenet_v1_1.0_224_frozen.pb  input  MobilenetV1/Predictions/Reshape_1  224,224,3
```

Note: you could have also used the output before the SoftMax rather than the output listed in the TensorFlow archive (result is the same), e.g.:

```
python  optimize_inference_graph.py  models/mobilenet/mobilenet_v1_1.0_224_frozen.pb  input  MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd  224,224,3
```

3. Follow the instructions in the [ResNet example](../resnet). E.g. run `python dnn_to_spatial.py models/mobilenet/mobilenet_v1_1.0_224_frozen_opt2.pb`, then follow the instructions to compile the application. The call to `img_to_csv.py` is also the same as that example, i.e. no preprocessing is required.
