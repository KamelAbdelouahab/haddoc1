# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import math
import codecs
from contextlib import contextmanager
import os,sys


def softmax_func(x):
    e = np.exp(x-np.max(x))
    s = e / np.sum(e)
    return s

def np2caph(numpy_array,caph_filename):
    import os
    f = open ('./tmp.txt','w')
    f.write(str(numpy_array))
    f.close()

    f1 = open('./tmp.txt','r')
    f2 = open (caph_filename,'w')
    for line in f1:
        f2.write(line.replace('[[','< < ').replace(']]',' > >').replace('[',' < ').replace(']',' > ').replace('\n',' ').replace('    ','  '))
    f1.close()
    f2.close()
    os.remove('./tmp.txt')

def caph2np(caph_filename):
    import numpy as np
    caph_list = codecs.open(caph_filename,encoding='utf-8').read()
    caph_list = caph_list.replace('<','').replace('>','').replace(' ',',').split(',')
    caph_list = [x for x in caph_list if (x!='')]
    caph_list = [x for x in caph_list if (x!=',')]
    caph_int  = list(map(int,caph_list))
    caph_np   = np.array(caph_int,dtype=int)
    return caph_np


def caffeClassif(Network,image_index):
    import numpy as np
    LastLayer = np.array(Network.blobs['fc'].data[image_index])
    out_class = np.argmax(softmax_func(LastLayer))
    #print "Caffe Features : {}".format(LastLayer)
    print("Caffe Classification result : {}\n".format(out_class))
    return out_class

def capheeClassif(Network,feature_dir):
    import numpy as np
    feature_nb = Network.blobs['fc'].data.shape[1]
    fc_list = []
    for n in  range(0,feature_nb):
        fc_np = caph2np(feature_dir+'/w_fc'+str(n)+'.txt')
        fc_list.append(fc_np)

    LastLayer = np.array(fc_list,dtype=int).reshape(feature_nb)
    out_class = np.argmax(softmax_func(LastLayer))
    print("Caphee Classification result : {}\n".format(out_class))
    return out_class
