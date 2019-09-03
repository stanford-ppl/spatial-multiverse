# Installation

1. Clone `spatial-multiverse` and initialize the `spatial` submodule using:

```
git clone git://github.com/stanford-ppl/spatial-multiverse.git
cd spatial-multiverse/spatial
git submodule init
git submodule update
make install
cd ..
```

Then add the following to your `.bashrc`, which prevents Java stack overflow and other errors.
The sizes can be further increased if needed.

```
export _JAVA_OPTIONS="-Xms1024m -Xss256m -Xmx16g -XX:MaxMetaspaceSize=16g"
```

2. Install [TensorFlow](https://github.com/tensorflow/tensorflow) using:

```
sudo pip install tensorflow==1.14.0
```

Any recent build should work, but we originally built and tested using version 1.12.0 and Python 2.7.6.
When later adding MobileNet support, we had moved to version 1.14.0.

3. If you want to run on the Amazon F1, you will need to set up [aws-fpga](https://github.com/aws/aws-fpga). Modify your `.bashrc` to set `AWS_HOME` and to source `hdk_setup.sh` and `sdk_setup.sh`. For more details you can read the AWS tutorial [here](aws.md).
