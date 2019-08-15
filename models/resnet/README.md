This directory contains the pre-trained ResNet example with the optimization scripts already run.
It is the original ResNet model used by He et al., converted to TensorFlow [here](https://github.com/ry/tensorflow-resnet).
To run this network,

1. Call `dnn_to_spatial.py` on this graph, e.g. `python dnn_to_spatial.py models/resnet/ResNet50_opt.pb`

2. Follow the instructions printed to move the generated file `resnet50opt.scala` to the Spatial apps directory and compile using Spatial

3. Select an input .jpg image and convert it to a .csv format using the `data/img_to_csv.py` script. When running the Spatial `Top` executable, pass the .csv file as an argument.

4. Once Spatial compilation finishes, follow these [instructions](../../docs/aws.md) to load the generated AFI to your EC2 F1 instance and run the inference. For instructions to target other devices supported by Spatial, see [here](https://spatial-lang.org/targetting-devices).
