# caffe train --solver=caffe/networks/caffe-lenet_solver.prototxt
# caffe train --solver=caffe/networks/lecun-lenet_solver.prototxt
caffe train --solver=caffe/networks/dream-lenet_solver.prototxt
mv caffe/networks/dream-lenet_iter_10000.caffemodel caffe/networks/dream-lenet.caffemodel
# caffe train --solver=caffe/networks/caffe3-lenet_solver.prototxt
