# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import math
import codecs
from contextlib import contextmanager
import os,sys


def genCaph_Headers(caph_net_filename):
    f= open(caph_net_filename,'w')
    f.write("#include \"dc.cph\"\n")
    f.write("#include \"mapfct.cph\"\n")
    f.write("#include \"conv_cnn.cph\"\n")
    f.write("#include \"relu.cph\"\n")
    f.write("#include \"pool.cph\"\n")
    f.write("#include \"repdc.cph\"\n")
    f.write("#include \"classif.cph\"\n")

    #-- Fichier genere par la librairie
    f.write("#include \"weights.cph\"\n")

    #-- Fichiers generes par le programme c++)
    f.write("#include \"dotdc.cph\"\n");
    f.write("#include \"fc_distri_act.cph\"\n")
    #-- f.write("#include \"fc_map_gen.cph\"\n")
    f.write("#include \"sumdc.cph\"\n \n")
    f.close()


def genCaph_CNN(network,caph_net_filename,caph_dataype,acteurconv,shiftnorm):
    f = open(caph_net_filename,'a')
    Blobs               = network.blobs
    Params              = network.params
    name_wire           =   "w_"
    name_wire_relu      =   "w_r"
    name_weights        =   "weights_"
    name_biais          =   "biais_"

    for b in list(Blobs.keys()):
        if 'label' in b or 'cla' in b or 'data' in b:
            del Blobs[b]
    for layer in list(Blobs.keys()):
        if 'conv' in layer:
            if 'conv1' in layer:
                generateFirstLayer(caph_net_filename,Params[layer][0].data,"i",acteurconv,name_weights+str(layer),shiftnorm,name_biais+str(layer),name_wire_relu,name_wire+str(layer))
                name_wire_previous = name_wire_relu
            else:
                generateConvLayer(caph_net_filename,Params[layer][0].data,name_wire_previous,acteurconv,name_weights+str(layer),shiftnorm,name_biais+str(layer),name_wire+str(layer),Params[layer][0].data.shape[1])
        if 'pool' in layer:
            generatePoolingLayer(caph_net_filename, name_wire_previous, Params[layer_previous][0].data.shape[0], name_wire+str(layer), 'pool');

        if 'conv1' not in layer:
            name_wire_previous = name_wire+str(layer)
        layer_previous     = layer
    f.close()

def genCaph_FC(network,caph_net_filename,caph_dataype,C2V_CPP_LIB,C2V_DIRNAME):
    import subprocess
    # Parcourir network jusqu'a FC : a FC recuprer
        #   le nombre de neurones de fc
        #   le nombre de features a son entrée
        #   la taille des features
        #   ... les trucs utiles pour genc
    Blobs   = network.blobs;
    nbclass = 10;
    for b in list(Blobs.keys()):
       if 'label' in b or 'cla' in b or 'data' in b:
         del Blobs[b]
    for layer in list(Blobs.keys()):
        if 'ip' not in layer:
            previous_layer = layer
        else:
            repsize         = network.blobs['conv1'].data.shape[1]
            nx_feat         = network.blobs[previous_layer].data.shape[2]
            ny_feat         = network.blobs[previous_layer].data.shape[3]
            nb_unit_fc      = network.blobs[layer].data.shape[1]
            nfeat           = network.blobs[previous_layer].data.shape[1]
            fc_wirename_in  = "w_" + previous_layer
            fc_wirename_out = "w_" + layer
            nbclass         = nb_unit_fc
            sizesum1        = network.blobs['conv1'].data.shape[1]
            sizesum2        = network.blobs['conv2'].data.shape[1]
            sizesum3        = network.blobs['conv3'].data.shape[1]

    f= open(caph_net_filename,'a')
    # # Include the FC layer generated file (with GENC.EXE)
    # f.write("\n #include \"fc_layer_gen.cph\"\n");
    # Output streams
    f.write("\nstream i:"+caph_dataype+"dc from \"sample.txt\";\n");
    for nb in range(sizesum3):
       f.write("stream w_pool3%d : " %nb)
       f.write("%s dc " %caph_dataype)
       f.write("to \"w_pool3%d.txt\";\n" %nb)
    f.close()

    # datatype= caph_dataype.replace("<","\<").replace(">","\>")
    datatype= "signed\<32\>"
    print('\033[94m' "\n > Lunching gen_cnn_code with parameters:"+ '')
    os.environ["GLOG_minloglevel"] = "1"
    print(subprocess.Popen(C2V_CPP_LIB + "/gen_cnn_code %d %d %d %d %d %d %d %s %s %d %s %s" %(repsize,sizesum1,sizesum2,nfeat,nx_feat,ny_feat,nb_unit_fc,fc_wirename_in,fc_wirename_out,nbclass,"y_",datatype), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
    os.environ["GLOG_minloglevel"] = "0"
    print(subprocess.Popen("cp -R  " + "/home/kamel/dev/haddoc1/cnn " +
                            "/home/kamel/dev/haddoc1/caph_generated" ,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT).stdout.read())
    print(subprocess.Popen("cp -R  " + "/home/kamel/dev/haddoc1/utils "+
                            "/home/kamel/dev/haddoc1/caph_generated" ,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT).stdout.read())
    #print(subprocess.Popen("rm -rf " + "/home/kamel/dev/haddoc1/cnn",
    #                        shell=True,
    #                        stdout=subprocess.PIPE,
    #                        stderr=subprocess.STDOUT).stdout.read())
    #print(subprocess.Popen("rm -rf " + "/home/kamel/dev/haddoc1/utils",
    #                       shell=True,
    #                       stdout=subprocess.PIPE,
    #                       stderr=subprocess.STDOUT).stdout.read())

# =======================================================================================================#
def generateFirstLayer(caph_net_filename,filters_conv1,name_wire_input,acteurconv,name_weights_conv1,shiftnorm,name_biais_conv1,name_wire_relu,name_wire_output_conv1):
    f= open(caph_net_filename,'a')
    f.write("net(")
    for nb in range(filters_conv1.shape[0]): # recuperer le nombre de kernels
       f.write("%s" %name_wire_output_conv1)
       f.write ("%d" %nb)
       if nb<filters_conv1.shape[0]-1:
            f.write (",")
    f.write(")=convs ")
    f.write("%s " %acteurconv)
    f.write("rep%d "%filters_conv1.shape[0] )
    f.write("%s " %name_weights_conv1)
    f.write("%d " %shiftnorm)
    f.write("%s " %name_biais_conv1)
    f.write("%s;" %name_wire_input)
    f.close()
    generateFactLayer(caph_net_filename, name_wire_output_conv1, filters_conv1.shape[0], name_wire_relu, 'relu');




def generateConvLayer(caph_net_filename, filters_conv2, name_wire_in, acteurconv, weights, shiftnorm, biais, name_wire_out, nconnect):
    with open(caph_net_filename, "a") as f:

        f.write("\n net(")
        for nb in range(filters_conv2.shape[0]): # recuperer le nombre de kernels
            f.write("%s" %name_wire_out)
            if (nb < filters_conv2.shape[0]-1):
                f.write ("%d," %nb)
            else:
                f.write ("%d)" %nb)

        #f.write("%s " %name_weights_conv2)       # todo N1 : doit être le numero de la couche ...

        f.write("= convlayer %s %s %d %s sum%d relu \n\t (" %(acteurconv,weights,shiftnorm,biais,nconnect))
        for n in range(filters_conv2.shape[0]): # sur chaque neurone
            f.write ("\t(")
            for nb in range(filters_conv2.shape[1]): # recuperer dentres de chaque neurone
                f.write("%s" %name_wire_in) # recuperer le nom des entres de la couche precedente
                if (nb < filters_conv2.shape[1]-1):
                    f.write ("%d," %nb)
                else:
                    if (n <filters_conv2.shape[0]-1):
                        f.write ("%d),\n" %nb)
                    else:
                        f.write ("%d));\n" %nb)

def generatePoolingLayer(caph_net_filename, name_wire_in, nb_fmv, name_wire_out, acteurpool):
    with open(caph_net_filename, "a") as f:

        #-- -- Couche S2: subsampling et activaton
        #net (t1,t2,t3,t4,t5,t6) = map (pool 2 2) (map (relu) (ts1,ts2,ts3,ts4,ts5,ts6));

        f.write("\nnet(")
        for nb in range(nb_fmv): # recuperer le nombre de kernels
            f.write("%s%d" %(name_wire_out,nb))
            if nb<nb_fmv-1:
                 f.write (",")
        f.write(")= map (%s 2 2) ( " %acteurpool)
        for nb in range(nb_fmv): # recuperer le nombre de kernels
            f.write("%s" %name_wire_in)
            f.write ("%d" %nb)
            if nb<nb_fmv-1:
                f.write (",")
            else:
                f.write(");\n")

def generateFactLayer(caph_net_filename, name_wire_in, nb_fmv, name_wire_out, acteur):
    with open(caph_net_filename, "a") as f:

        #-- -- Couche S2: subsampling et activaton
        #net (t1,t2,t3,t4,t5,t6) = map (pool 2 2) (map (relu) (ts1,ts2,ts3,ts4,ts5,ts6));

        f.write("\nnet(")
        for nb in range(nb_fmv): # recuperer le nombre de kernels
            f.write("%s%d" %(name_wire_out,nb))
            if nb<nb_fmv-1:
                 f.write (",")
        f.write(")= map %s ( " %acteur)
        for nb in range(nb_fmv): # recuperer le nombre de kernels
            f.write("%s" %name_wire_in)
            f.write ("%d" %nb)
            if nb<nb_fmv-1:
                f.write (",")
            else:
                f.write(");\n")
