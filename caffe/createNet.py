import os
import sys
import subprocess

HOME                = os.environ['HOME']
CAFFE_PATH           = HOME + '/caffe'
sys.path.insert(0, CAFFE_PATH +'/python')
CURRENT_PATH        = os.path.dirname(os.path.realpath(__file__))

os.environ["GLOG_minloglevel"] = "1"
import caffe
os.environ["GLOG_minloglevel"] = "0"

from pylab import *
from caffe import layers as L
from caffe import params as P

def create_Net (lmdb,batch_size,C1,C2,C3):
    os.environ["GLOG_minloglevel"] = "1"
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                            transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=C1, weight_filler=dict(type='xavier'))
    n.Relu1 = L.ReLU(n.conv1, in_place=True)
    n.s1 = L.Pooling(n.Relu1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.s1, kernel_size=3, num_output=C2, weight_filler=dict(type='xavier'))
    n.Relu2 = L.ReLU(n.conv2, in_place=True)
    n.s2 = L.Pooling(n.Relu2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.s2, kernel_size=3, num_output=C3, weight_filler=dict(type='xavier'))
    n.Relu3 = L.ReLU(n.conv3, in_place=True)
    n.fc = L.InnerProduct(n.Relu3, num_output=10, weight_filler=dict(type='xavier'))
    n.cla = L.SoftmaxWithLoss(n.fc,n.label)
    os.environ["GLOG_minloglevel"] = "0"
    return n.to_proto()


def create_solver(solver_prototxt,train_prototxt,test_prototxt):
    with open(solver_prototxt,'w') as f:
        f.write ("train_net: \"" + train_prototxt  + "\" " + "\n")
        f.write ("test_net: \""  + test_prototxt   + "\" " + "\n")
        f.write ("test_iter: 1" + "\n")
        f.write ("test_interval: 1000" + "\n")
        f.write ("test_compute_loss: true" + "\n")
        f.write ("base_lr: 0.01" + "\n")
        f.write ("lr_policy: \"step\"" + "\n")
        f.write ("gamma: 0.1" + "\n")
        f.write ("stepsize: 1000" + "\n")
        f.write ("display: 100" + "\n")
        f.write ("max_iter: 10000" + "\n")
        f.write ("weight_decay: 0.001" + "\n")
        f.write ("momentum: 0.9" + "\n")
        f.write ("solver_mode: CPU" + "\n")
        f.close()


#=========================== MAIN ============================#


DREAMNET_PATH                =     CURRENT_PATH  + "/dreamnet"
DATASET_PATH                =     CURRENT_PATH  + "/dataset"

train_dataset                =     DATASET_PATH  + "/mnist_train"
#~ test_dataset                =     DATASET_PATH  + "/mnist_test"
test_dataset                =     DATASET_PATH  + "/usps_test"

dreamnet_solver_prototxt    =    DREAMNET_PATH + "/solver.prototxt"
dreamnet_train_prototxt     =    DREAMNET_PATH + "/train.prototxt"
dreamnet_test_prototxt        =    DREAMNET_PATH + "/test.prototxt"

print subprocess.Popen("rm -rf net", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
print subprocess.Popen("rm -rf " + DREAMNET_PATH , shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()


print subprocess.Popen("mkdir net", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
print subprocess.Popen("mkdir " + DREAMNET_PATH , shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()

create_solver(dreamnet_solver_prototxt,dreamnet_train_prototxt,dreamnet_test_prototxt)

if (len(sys.argv) != 4):
    sys.exit("Not enough args!")
else:
    C1    =    int(sys.argv[1])
    C2    =    int(sys.argv[2])
    C3    =    int(sys.argv[3])

print + " Creating Convolutional Neural Network with topology :\n " + ''

print "\t Layer 1 |" + str(C1) + "\t | Convolution of 3x3 kernel"
print "\t Layer 2 |" + str(C1) + "\t | Max pool susbsampling layer"
print "\t Layer 3 |" + str(C2) + "\t | Convolution of 3x3 kernel"
print "\t Layer 4 |" + str(C2) + "\t | Max pool susbsampling layer"
print "\t Layer 5 |" + str(C3) + "\t | Convolution of 3x3 kernel"
print "\t Layer 6 |" + str(10) + "\t | Fully connected layer \n \n"


print "\t Training set at: " + train_dataset
print "\t Testset at: " + test_dataset + "\n"

with open(dreamnet_train_prototxt, 'w') as f:
    f.write(str(create_Net(train_dataset,64,C1,C2,C3)))

with open(dreamnet_test_prototxt, 'w') as f:
    f.write(str(create_Net(test_dataset,1000,C1,C2,C3)))

print + "Succefully generated train and test prototxt files" + ''
