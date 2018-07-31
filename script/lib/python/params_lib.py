# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import math
import codecs
from contextlib import contextmanager
import os,sys

# Convert caffe params to fixed point
def ConvLayerFixedpoint(layer,scale_factor):
    filters_conv = np.array(np.round(scale_factor*layer[0].data),dtype=int);
    biais_conv= np.array(np.round(scale_factor*layer[1].data),dtype=int);
    return (filters_conv,biais_conv)

def IPLayerFixedpoint(layer,scale_factor):
    filters_ip  = np.array(np.round(scale_factor*layer[0].data),dtype=int);
    biais_ip  = np.array(np.round(scale_factor*layer[1].data),dtype=int);
    return (filters_ip,biais_ip)
# ==========================================================================================================

# Generate the caph lib files
def GenerateIPCst(filename, name, layer_, filter_fixedpt, biais_fixedpt, name_previous_layer,caph_dataype):

    with open(filename, "a") as f:
        size_in = layer_[name_previous_layer][0].data.shape[0]

        featmap_nb = filter_fixedpt.shape[1]/size_in;
        f.write("const  weights_%s = [" %name)

        for nb in range(filter_fixedpt.shape[0]): #nombre de neurones dans la couche
            f.write("\n[")
            for kk in range(size_in):
                f.write("[")
                for k in range(featmap_nb):
                    f.write("%d" % filter_fixedpt[nb][k +(featmap_nb*kk)])
                    if(k < featmap_nb-1):
                        f.write(",")
                    else:
                        if (kk <size_in-1):
                            f.write("],")
                        else:
                            if(nb < filter_fixedpt.shape[0]-1):
                                f.write("]],")
                            else:
                                  f.write("]]")

        f.write("]:%s" %caph_dataype)
        f.write("array[%d][%d][%d];\n\n" % (filter_fixedpt.shape[0],size_in, filter_fixedpt.shape[1]/size_in))

        # Write the bias FC term
        f.write("const biais_fc = [")
        for i in range(biais_fixedpt.shape[0]):
            if (i < (biais_fixedpt.shape[0]-1)):
                f.write("%d, " %biais_fixedpt[i])
            else:
                f.write("%d]" % biais_fixedpt[i])
                f.write(":%s" %caph_dataype)
                f.write(" array[%d];" %biais_fixedpt.shape[0])

        f.write("\n\n")

def GenerateConvCst(filename, name, filter_fixedpt, biais_fixedpt,caph_dataype):
    with open(filename, "a") as f:
        f.write("const weights_%s = [" %name)

        for nb in range(filter_fixedpt.shape[0]): #nombre de neurones dans la couche
            f.write("\n[")
            for k in range(filter_fixedpt.shape[1]): #nombre de filtres dans le neurone

                f.write("[")
                for i in range(filter_fixedpt.shape[2]): # taille filtre conv x
                    for j in range(filter_fixedpt.shape[3]): # taille filtre conv y
                        f.write("%d" % filter_fixedpt[nb][k][filter_fixedpt.shape[2]-i-1][filter_fixedpt.shape[3]-j-1])
                        if ( i == (filter_fixedpt.shape[2]-1) and (j ==filter_fixedpt.shape[2]-1)):   # utiliser dimension du filtre -1
                            f.write("]")
                        else:
                            f.write(",")
                if(k < filter_fixedpt.shape[1]-1):
                    f.write(",")
                else:
                    if (nb < filter_fixedpt.shape[0]-1):
                        f.write("],")
                    else:
                        f.write("]")


        f.write("]:%s" %caph_dataype)
        f.write("array[%d]" %filter_fixedpt.shape[0])
        f.write("[%d][9];\n" %filter_fixedpt.shape[1])

        # Ecrit la premiere liste de biais
        f.write("\nconst biais_%s= [" %name)

        #print filters_conv1_b
        for i in range(biais_fixedpt.shape[0]):
            if (i < (biais_fixedpt.shape[0]-1)):
                f.write("%d, " %biais_fixedpt[i])
            else:
                f.write("%d]" % biais_fixedpt[i])
                f.write(":signed<32>array[%d];" %biais_fixedpt.shape[0])

        f.write("\n\n")

def GenerateFirstConvCst(filename,layer,scale_factor,caph_dataype):
	#Recuperer les valeurs des Kernels depuis le caffe_model + Application du facteur d'echelle
	filters_conv1 = np.array(np.round(scale_factor*layer[0].data),dtype=int)
	name_weights_conv1="weights_conv1"
	name_biais_conv1="biais_conv1"

	f= open(filename,'w');
	f.write("const %s = [\n" %name_weights_conv1)
	for nb in range(filters_conv1.shape[0]): # recuperer le nombre de kernels
		f.write("\t[")
		for i in range(filters_conv1.shape[2]):
			for j in range(filters_conv1.shape[3]):
				f.write("%d" % filters_conv1[nb][0][filters_conv1.shape[2]-i-1][filters_conv1.shape[3]-j-1])
				if ( i == (filters_conv1.shape[2]-1) and (j ==filters_conv1.shape[2]-1)):
					f.write(" ")
				else:
					f.write(",")

		if (nb < (filters_conv1.shape[0]-1)):
			f.write("],\n")
		else:
			f.write(" ] ] : " + caph_dataype +" array[%d][9];\n" %filters_conv1.shape[0] )

	filters_conv1_b= np.array(np.round(scale_factor*layer[1].data),dtype=int)
	#~ filters_conv1_b= deepcopy(layer[1].data);
	# Ecrit la premiere liste de biais
	f.write("const %s= [" %name_biais_conv1)
	for i in range(filters_conv1_b.shape[0]):
		if (i < (filters_conv1_b.shape[0]-1)):
			f.write("%d, " %filters_conv1_b[i])
		else:
			f.write("%d] : " % filters_conv1_b[i])
			f.write( caph_dataype+" array[%d];" %filters_conv1_b.shape[0])

	f.write("\n \n")

# ==========================================================================================================
