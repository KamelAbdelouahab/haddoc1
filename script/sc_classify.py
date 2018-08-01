# -*- coding: utf-8 -*-
import os
import sys

HOME                = os.environ['HOME']
CAFFE_DIRNAME       = HOME + '/caffe'
CAFFE_PYTHON_LIB    = CAFFE_DIRNAME+'/python'
C2V_DIRNAME         = os.path.dirname(os.path.realpath(__file__))
C2V_PYTHON_LIB      = C2V_DIRNAME + '/lib/python'
C2V_CAPH_LIB        = C2V_DIRNAME + '/lib/caph'
C2V_CPP_LIB         = C2V_DIRNAME + '/lib/cpp'
CAPH_GENERATED      = C2V_DIRNAME + '/../caph_generated'

sys.path.insert(0, CAFFE_PYTHON_LIB)
sys.path.insert(0, C2V_PYTHON_LIB)


from utils_lib      import *

if (len(sys.argv) == 1):
    print('\033[93m' + "Backdoor  : Using the DreamNet network ... " + '')
    prototxt    = '../example/dreamnet/train.prototxt'
    caffemodel  = '../example/dreamnet/dreamnet.caffemodel'
    feature_dir = CAPH_GENERATED

else:
    if (len(sys.argv) != 4):
        sys.exit("Not enough args!")
    else:
        prototxt    = sys.argv[1]
        caffemodel  = sys.argv[2]
        feature_dir = sys.argv[3]
print('\033[94m' + "     RUNNING CLASSIFICATION USING SYSTEMC SIM RESULTS ...  " + '')

#SUPRESS CAFFE DISPLAY WHEN CREATING NETWORK
os.environ["GLOG_minloglevel"] = "1"
import caffe
Network = caffe.Net(prototxt,caffemodel,caffe.TEST)
os.environ["GLOG_minloglevel"] = "0"

#nb_features = Network.blobs['fc'].data.shape[1]
capheeClassif(Network,feature_dir)
