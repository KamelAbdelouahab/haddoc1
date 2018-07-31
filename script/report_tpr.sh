caffe test -solver caffe/networks/caffe-lenet_train.prototxt \
           -weights caffe/networks/caffe-lenet.caffemodel \
           - gpu 0 \
           - iterations 100 \
