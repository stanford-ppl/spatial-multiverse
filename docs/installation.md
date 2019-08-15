# Installation

1. Clone `spatial-multiverse` and initialize the `spatial` submodule using:

```
git clone git://github.com/stanford-ppl/spatial-multiverse.git
cd spatial-multiverse/spatial
git submodule init
git submodule update
cd ..
```

2. Install [TensorFlow](https://github.com/tensorflow/tensorflow) from source. The compiler can work by installing only the python API, however the 
[graph transform tools](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) used by the first optimization level need to be built with bazel and require installation from source. We built and tested using version 1.12.0 and Python 2.7.6.

3. Compile Spatial in the `spatial` subdirectory of this repository by following the installation instructions [here](https://github.com/stanford-ppl/spatial#getting-started). Note: you do not need to clone Spatial, only compile it. This repository already contains Spatial as a submodule that points to the correct branch.

4. If you want to run on the Amazon F1, you will need to set up [aws-fpga](https://github.com/aws/aws-fpga). Modify your `.bashrc` to set `AWS_HOME` and to source `hdk_setup.sh` and `sdk_setup.sh`. For more details you can read the AWS tutorial [here](aws.md).
