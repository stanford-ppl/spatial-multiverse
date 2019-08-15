To compile the example,

1. Run `python get_djinnasr.py` to generate the model
2. Follow the instructions printed by the script to create the inference graph, and then to optimize the graph
3. Call `dnn_to_spatial.py` for the optimized graph and follow the command printed to compile through Spatial
4. Once Spatial compilation finishes, follow these [instructions](../../docs/aws.md) to load the generated AFI to your EC2 F1 instance and run the inference. For instructions to target other devices supported by Spatial, see [here](https://spatial-lang.org/targetting-devices).
