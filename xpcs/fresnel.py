#! /usr/bin/env python
import numpy as np

def basic_reflectivity(alpha, reflectivity_index):

    dns2 = 2 * reflectivity_index
    kz = np.sin(alpha)
    kt = np.sqrt(np.sin(alpha)**2 - dns2)
    Rf = (kz - kt)/(kz + kt)
    Tf = 2 * kz / (kz + kt)
    return  Rf, Tf

def propagation_coeffs(alphai, alpha, reflectivity_index):

    Ri, Ti  = basic_reflectivity(alphai, reflectivity_index)
    Rf, Tf  = basic_reflectivity(alpha, reflectivity_index)
    return [Ti*Tf, Ri*Tf, Ti*Rf, Ri*Rf] 
