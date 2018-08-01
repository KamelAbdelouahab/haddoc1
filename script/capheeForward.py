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

import caffe
import subprocess

# How to use : =======================================================================================
# Python capheeForward.py <caph_test_set_directory> <caph_test_set_index(or Filename without extention)>
# ====================================================================================================

#Create sample.txt from testset
print('\033[94m' + " ================ Caphee HLS tool ================ " '')
sample_dir = sys.argv[1]
sample_file = sys.argv[2]
print(subprocess.Popen("cp " + sample_dir + sample_file + ".txt" + " caphee/sample.txt", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())

Network = caffe.Net('./Net/DreamNet_test.prototxt','./Result/DreamNet_iter_10000.caffemodel',caffe.TEST)
Network.forward()
scale_factor = 32

os.chdir(CAPH_GENERATED)

print(subprocess.Popen("rm -rf ./fc_out.txt", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
print(subprocess.Popen("make clean", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
print(subprocess.Popen("make systemc.makefile", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
print(subprocess.Popen("make systemc", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())


capheeClassif(10,'./res')
caffeClassif(Network,scale_factor,int(sample_file))
