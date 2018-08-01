# caffe test -model caffe/networks/caffe-lenet_train.prototxt -weights caffe/networks/caffe-lenet.caffemodel -iterations 500 -gpu 0
# caffe test -model caffe/networks/caffe-lenet_test.prototxt -weights caffe/networks/caffe-lenet.caffemodel -iterations 500 -gpu 0
# caffe test -model caffe/networks/lecun-lenet_train.prototxt -weights caffe/networks/lecun-lenet.caffemodel -iterations 500 -gpu 0
# caffe test -model caffe/networks/lecun-lenet_test.prototxt -weights caffe/networks/lecun-lenet.caffemodel -iterations 500 -gpu 0
caffe test -model caffe/networks/dream-lenet_train.prototxt -weights caffe/networks/dream-lenet.caffemodel -iterations 500 -gpu 0
caffe test -model caffe/networks/dream-lenet_test.prototxt -weights caffe/networks/dream-lenet.caffemodel -iterations 500 -gpu 0
