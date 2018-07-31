# -*- coding: utf-8 -*-
import os
import sys

C2V_DIRNAME         = os.path.dirname(os.path.realpath(__file__))
C2V_PYTHON_LIB      = C2V_DIRNAME + '/lib/python'
C2V_CAPH_LIB        = C2V_DIRNAME + '/lib/caph'
C2V_CPP_LIB         = C2V_DIRNAME + '/lib/cpp'
CAPH_GENERATED      = C2V_DIRNAME + '/../caph_generated'


#BACKDOOR

sys.path.insert(0, CAFFE_PYTHON_LIB)
sys.path.insert(0, C2V_PYTHON_LIB)


from caph_net_lib   import *
from params_lib     import *
from utils_lib      import *
from copy           import deepcopy
import subprocess
import math


print('\033[4m' '\033[94m' "\n > THIS IS HADOC HLS TOOL \n" '\033[0m')

if (len(sys.argv) == 1):
    print('\033[92m' " \t Backdoor 1 : Using the DreamNet network with 6 bits for parameter representation " '\033[0m')
    prototxt    = '../example/dreamnet/train.prototxt'
    caffemodel  = '../example/dreamnet/dreamnet.caffemodel'
    nb_bits     = 6

else:
    if (len(sys.argv) == 2):
        prototxt    = '../example/dreamnet/train.prototxt'
        caffemodel  = '../example/dreamnet/dreamnet.caffemodel'
        nb_bits     = int(sys.argv[1])

    else:
        if (len(sys.argv) != 3):
            sys.exit("Not enough args!")
        else:
            prototxt    = sys.argv[1]
            caffemodel  = sys.argv[2]
            nb_bits     = sys.argv[3]

print('\033[94m' "\t Network used at: " + prototxt + '\033[0m')
print('\033[94m' "\t caffemodel at: " + caffemodel + '\033[0m')
print('\033[94m' "\t Parameter represented in : " + str(nb_bits) + " bits fixed point " +'\033[0m')


#SUPRESS CAFFE DISPLAY WHEN READING NETWORK
os.environ["GLOG_minloglevel"] = "1"
import caffe
Network = caffe.Net(prototxt,caffemodel,caffe.TEST)
os.environ["GLOG_minloglevel"] = "0"


shiftnorm = nb_bits - 1
scale_factor = math.pow(2,(nb_bits - 1)) - 1

#========================================================================================================
# ------------------------------   Generate weights caph file  ----------------------------
#=======================================================================================================

caph_dataype = "signed<32>"
caph_weights_filename = CAPH_GENERATED + '/weights.cph'

f= open(caph_weights_filename,'w')
f.write("-- File Generated by the Caphee HLS Tool\n------------------------------------------------\n");
f.close()


Blobs = Network.blobs
for b in list(Blobs.keys()):
  if 'label' in b or 'cla' in b or 'data' in b:
    del Blobs[b]

for k in list(Blobs.keys()):
    if 'conv' in k:
        if 'conv1'in k:
            GenerateFirstConvCst(caph_weights_filename,Network.params[k],scale_factor,caph_dataype)
        else :
            (filter_fixedpt,biais_fixept) = ConvLayerFixedpoint(Network.params[k],scale_factor)
            GenerateConvCst(caph_weights_filename,k,filter_fixedpt,biais_fixept,caph_dataype)
    if 'fc' in k:
        (filter_fixedpt,biais_fixept) = IPLayerFixedpoint(Network.params[k],scale_factor)
        GenerateIPCst(caph_weights_filename,k,Network.params,filter_fixedpt,biais_fixept, previous_layer,caph_dataype)
    previous_layer = k
print('\033[92m' "\n > Succefully extracted network parameters in: \n \t " + caph_weights_filename +'\033[0m')

#========================================================================================================
# -------------------------------------  Generate CAPH Network   ---------------------------------------
#=======================================================================================================

caph_net_filename = CAPH_GENERATED + '/cnn_generated.cph'

genCaph_Headers(caph_net_filename)
genCaph_CNN(Network,caph_net_filename,caph_dataype,"conv233c_wb_opt",shiftnorm)
genCaph_FC(Network,caph_net_filename,caph_dataype,C2V_CPP_LIB,C2V_DIRNAME)

print('\033[92m' " > Succefully generated caph network in: \n \t " + caph_net_filename +'\033[0m \n')
