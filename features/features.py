""" 
This Module helps in extracting features from given matrix i.e ROI (Region of Interest) / Window

""" 
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math


# calculate glcm function
def calc_glcm(image_matrix):

    """This function calculates GLCM of a given matrix

    Args:
        image_matrix : matrix Argument (2D Array)

    Returns:
        2D array : returns gray scale image in form of matrix (2D array)
    """
    y = np.zeros([256, 256])
    p, l = image_matrix.shape
    # y = np.zeros([p,l])

    # p,l = image_matrix.shape
    for i in range(p):  # row
        for j in range(l):  # col
            if j + 1 < l:
                y[image_matrix[i, j], image_matrix[i, j + 1]] += 1

    # Get the Transpose of the matrix
    glcm_step1 = y[1:256, 1:256]
    # glcm_step1 = y[1:p,1:l]
    rows = len(glcm_step1)
    cols = len(glcm_step1[0])

    Transpose_M = []
    for k in range(cols):
        row = []
        for l in range(rows):
            row.append(glcm_step1[l][k])
        Transpose_M.append(row)

    # Calculate Final GLCM = Transpose_M + glcm_step1
    final_glcm = glcm_step1 + Transpose_M

    # Normalize the final GLCM
    final_glcm = final_glcm.astype("float")
    Sum_Total = 0.0
    p, l = final_glcm.shape
    for i in range(p):  # row
        for j in range(l):  # col
            Sum_Total += final_glcm[i, j]

    # Divide each cell by sum of whole matrix
    for i in range(p):  # row
        for j in range(l):  # col
            final_glcm[i, j] = final_glcm[i, j] / Sum_Total

    return final_glcm


# calculate contrast
def contrast(glcm_calculated_matrix):
    """_summary_

    Args:
        glcm_calculated_matrix (int): matrix / 2D array of ints

    Returns:
        float: contrast value of the given matrix / 2D array (also known as ROI / window)
    """
    contr = 0.0
    # contrast_glcm = resultant_glcm.copy()
    p, l = glcm_calculated_matrix.shape
    for i in range(p):  # row
        for j in range(l):  # col
            abs_val = abs(i - j) ** 2
            glcm_calculated_matrix[i, j] += abs_val * glcm_calculated_matrix[i, j]
            contr += glcm_calculated_matrix[i, j]

    return contr


# calculate energy
def energy(glcm_calculated_matrix):
    """_summary_

    Args:
        glcm_calculated_matrix1 (int): Matrix / 2D array of ints

    Returns:
        _float_: _returns energy of the ROI / Window_
    """
    ener = 0.0
    # energy_glcm = resultant_glcm.copy()
    p, l = glcm_calculated_matrix.shape
    for i in range(p):  # row
        for j in range(l):  # col
            ener += glcm_calculated_matrix[i][j] ** 2

    return ener


# calculate entropy
def entropy(glcm_calculated_matrix):
    """

    Args:
        glcm_calculated_matrix (int): matrix / 2D array of integers

    Returns:
        float: returns float  entropy value
    """
    entr = 0.0
    # entropy_glcm = resultant_glcm.copy()
    p, l = glcm_calculated_matrix.shape
    for i in range(p):  # row
        for j in range(l):  # col
            if glcm_calculated_matrix[i, j] > 0:
                entr -= glcm_calculated_matrix[i][j] * math.log10(
                    glcm_calculated_matrix[i, j]
                )
    return entr
