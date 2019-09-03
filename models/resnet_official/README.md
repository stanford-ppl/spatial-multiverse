This example contains the newer implementation of ResNet released by the [TensorFlow team](https://github.com/tensorflow/models/tree/master/official/resnet).

1. First, download the model using: 

```
bash get_resnet_official.sh
```

2. Then run the following commands from the root directory:

```
python create_inference_graph.py saved_model models/resnet/resnet_v1_fp32_savedmodel_NHWC/1538686669 softmax_tensor  models/resnet_official/ resnet_official

python  optimize_inference_graph.py  models/resnet_official/resnet_official.pb  input_tensor  softmax_tensor  224,224,3

# You may also want to print ImageNet classification, if so see the instructions [here](../../docs/demo.md)

python dnn_to_spatial.py models/resnet_official/resnet_official_opt2.pb
```

3. Follow the instructions printed to move the generated file `resnetofficialopt2.scala` to the Spatial apps directory and compile using Spatial

4. Select an input .jpg image and convert it to a .csv format using the `data/img_to_csv.py` script. When running the Spatial `Top` executable, pass the .csv file as an argument.

5. Once Spatial compilation finishes, follow these [instructions](../../docs/aws.md) to load the generated AFI to your EC2 F1 instance and run the inference. For instructions to target other devices supported by Spatial, see [here](https://spatial-lang.org/targetting-devices).
