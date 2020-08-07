# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:57:22 2020

@author: pulkit
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = dataset['fea']
    return x - np.mean(x, axis=0)

def get_covariance(dataset):
    n = len(dataset)
    covarience = np.dot(np.transpose(dataset), dataset) * 1/(n-1)
    return covarience

def get_eig(S, m):
    evalue, evector = eigh(S)
    #sort in descending order
    evalue[::-1].sort()
    evector = np.flip(evector,1)
    a = (m,m)
    b = (len(S),m)
    mat = np.zeros(a)
    evecMat = np.zeros(b)
    for i in range(len(S)):
        for j in range(m):
            evecMat[i][j] = evector[i][j].copy()
    for i in range(len(mat)):
        mat[i][i] = evalue[i]
    return mat,evecMat

def project_image(image, U):
    vec = np.dot(np.transpose(image), U)
    projection = np.dot(vec, np.transpose(U))
    return projection

def display_image(orig, proj):
    proj= np.transpose(np.reshape(proj, [32,32]))
    orig = np.transpose(np.reshape(orig, [32,32]))
    img, pic = plt.subplots(ncols = 2, nrows = 1)
    pic[0].set_title("Original")
    pic[1].set_title("Projection")
    first = pic[0].imshow(orig, aspect ='equal')
    img.colorbar(first, ax = pic[0])
    second = pic[1].imshow(proj, aspect = 'equal')
    img.colorbar(second, ax = pic[1])
    plt.show()
    return