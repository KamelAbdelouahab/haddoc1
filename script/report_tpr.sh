caffe test -model caffe/networks/caffe-lenet_train.prototxt -weights caffe/networks/caffe-lenet.caffemodel -iterations 500 -gpu 0
caffe test -model caffe/networks/caffe-lenet_test.prototxt -weights caffe/networks/caffe-lenet.caffemodel -iterations 500 -gpu 0
