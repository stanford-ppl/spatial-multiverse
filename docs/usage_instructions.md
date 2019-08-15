# Usage Instructions

The scripts should be run in the following order. 
See the `models` directory for some examples.

#### To create the optimized inference graph,

  1. If the input is a saved TensorFlow checkpoint or SavedModel, run `create_inference_graph.py`. You can skip this if you have the inference graph already.
  2. To optimize an inference graph, run `optimize_inference_graph.py`

#### To generate a design from the TensorFlow model,

  1. Run `dnn_to_spatial.py` and pass the inference graph as input
  2. Follow the printed instructions to run the Spatial compiler for the generated application
  3. Once Spatial compilation finishes, follow the [aws instructions](aws.md) to load the generated AFI to your EC2 F1 instance and run the inference

#### To give input data to Spatial,

  1. Run `data/img_to_csv.py` on the input image
  2. Pass the .csv file as an argument to the Spatial Top executable
