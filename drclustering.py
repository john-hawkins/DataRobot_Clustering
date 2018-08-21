# #################################################################
# Methods for clustering DataRobot model prediction explanations
# #################################################################
# Import the prediction explanation functions in drpredexplanations.py 
import drpredexplanations as pe 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functools import reduce
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
import datarobot as dr
import pandas as pd
import numpy as np

MAX_SAMPLES = 2000

# #############################################################
# THIS METHOD WILL BE USED BY SOME CUSTOM CLUSTERING APPROACHES
# THAT DO NOT REQUIRE TRANSFORMING THE RESULTS OF THE PREDICTION
# EXPLANATION SCORE INTO A NUMERIC VECTOR 
# #############################################################
def calculate_custom_distance(row1, row2, n_reasons=5):
    max_similarity = n_reasons * 3
    similarity = 0
    for i in range(n_reasons):
       r = 6*(i+1)
       if row1[r] == row2[r]:
           if (row1[r+1]==row2[r+1]):
              if (row1[r+3]==row2[j+3]):
                  similarity = similarity + 3
              elif ( abs(row1[r+4]+row2[j+4]) == ( abs(row1[r+4])+ abs(row2[j+4]) ) ):
                  similarity = similarity + 2
              else : # WHEN THE FEATURE AND VALUE ARE THE SAME BUT EXPLANATION DIRECTION IS DIFFERENT WE PENALISE
                  similarity = similarity - 2
           elif row1[j+3]==row2[j+3]:
              similarity = similarity + 1

    return max_similarity - similarity

#######################################################################
# K Means Cluster on Explanation Strength 
# #############################################################
def kmeans_cluster_by_strength(all_rows, kvalue, include_score=True, n_reasons=5):
    dfnew = pe.get_strength_per_feature_cols(all_rows, n_reasons)
    if include_score:
        dfnew['dr_score'] = all_rows['class_1_probability']
    kmeans = KMeans(n_clusters=kvalue)
    kmeans = kmeans.fit(dfnew)
    labels = kmeans.predict(dfnew)
    return (dfnew, kmeans, labels)

#######################################################################
# SAMPLE THE DATA SET DOWN BEFORE RUNNING ANY PROCESSES
#########################################################################
def sample_down(pdata):
    dfsample = pdata
    if len(dfsample) > MAX_SAMPLES:
        dfsample = pdata.sample(MAX_SAMPLES)
    return dfsample

#######################################################################
# RUN KMEANS CLUSTER AND PLOT
#########################################################################
def generate_kmeans_cluster_plot(proj, mod, pdata, kvalue, colone, coltwo):
    rdata = sample_down(pdata)
    expl_rows = pe.retrieve_prediction_explanations(proj, mod, rdata)
    dfsample, kmeans, labels = kmeans_cluster_by_strength(expl_rows, kvalue)
    print("Clustering finished: ")
    TARGET=proj.target
    dim1 = rdata[colone]
    dim2 = rdata[coltwo]
    dim3 = expl_rows['class_1_probability']
    strlabs = [('C'+str(elem)) for elem in labels]
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dim1, dim2, dim3, c=strlabs)
    ax.set_xlabel(colone)
    ax.set_ylabel(coltwo)
    ax.set_zlabel(TARGET)
    return plt

#######################################################################
# CREATE CLUSTER PLOT AND SAVE IT IN A FILE
#########################################################################
def create_and_save_kmeans_cluster_plot(proj, mod, pdata, kvalue, colone, coltwo, filename):
    plt = generate_kmeans_cluster_plot(proj, mod, pdata, kvalue, colone, coltwo)
    plt.savefig(filename, format='png')
