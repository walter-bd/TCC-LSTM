#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:25:34 2018

@author: walter
"""
import pandas
import numpy as np
import datetime
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import openpyxl

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

'''Read Data from csv'''
def read_data(filename, cols, drop_na=False):
    dataset_x_y = pandas.read_csv(filename, usecols=cols, engine='python')
    if drop_na:
        list_dataset = dataset_x_y.dropna(axis=0, how='any').values.tolist()
    y = []
    x = []
    dic = {"a.m.":"am", "p.m.":"pm"}
    for index, i in enumerate(list_dataset):
        y.append(float(i[1].replace(",",".")))
        x_aux = datetime.datetime.strptime(replace_all(i[0],dic), "%d/%m/%Y %I:%M:%S %p").date()
        x.append(x_aux)
    del x_aux, i, index
    return x, y

'''Read Data from XLSX file and convert to CSV'''
def convert_xlsx(filename, sheetname, csv_name):
    data_xls = pandas.read_excel(filename, sheetname, convert_float=False, index_col=None)
    data_xls.to_csv(csv_name, encoding='utf-8')
    
    

def feature_scaling(y, axis_sel=0, keep_dims=True, method='standarization'):
    if isinstance(y, (list,)):
        y = np.array(y)
    if isinstance(y, (np.ndarray,)):
        y.reshape(len(y),1)
        if (method=='standarization'):
            mean_y = np.mean(y, axis=axis_sel, keepdims=keep_dims)
            std_y = np.std(y, axis=axis_sel, keepdims=keep_dims)
            y = np.divide(y-mean_y, std_y)
            return y, std_y, mean_y
        if (method=='min-max'):
            max_y = np.max(y, axis=axis_sel, keepdims=keep_dims)
            min_y = np.min(y, axis=axis_sel, keepdims=keep_dims)
            y = np.divide(y-min_y, max_y-min_y)
            return y, max_y, min_y
        if (method=='minimax'):
            max_y = np.abs(np.max(y, axis=axis_sel, keepdims=keep_dims))
            y = np.divide(y, max_y)
            min_y = 0
            return y, max_y, min_y
        return None

def scaling_test(y, max_y, min_y):
    if isinstance(y, (list,)):
        y = np.array(y)
    if isinstance(y, (np.ndarray,)):
        y.reshape(len(y),1)
        y = np.divide(y-min_y, max_y-min_y)
        return y, max_y, min_y

def wavelet_decomposition(y, wavelet_mother="db8", level_wt=3):
    w = pywt.Wavelet(wavelet_mother)
    coeffs = {}
    wd = {}
    coeffs['cA' + str(level_wt)] = pywt.downcoef('a', y, w, mode='sym', level=level_wt)
    wd['cA' + str(level_wt)] = pywt.upcoef('a', coeffs['cA' + str(level_wt)], w, level=level_wt, take=len(y))
    for i in range(level_wt, 0, -1):
        coeffs['cD' + str(i)] = pywt.downcoef('d', y, w, mode='sym', level=i)
        wd['cD' + str(i)] = pywt.upcoef('d', coeffs['cD' + str(i)], w, level=i, take=len(y)) 
    return (w ,wd, coeffs)

def plot_wavelets(y, wd, level_wt, data_label, wavelet_label,  f_size=15, l_size=10, pdf=False):
    fig, ax = plt.subplots(level_wt + 2, 1, figsize=(40, 30))
    for i in range(1,level_wt + 2):
        if i == level_wt + 1:
            coeffs_label = 'cA' + str(level_wt)
            ax[i-1].plot(list(range(0,len(wd[coeffs_label]))), wd[coeffs_label], label=coeffs_label)
        else:
            coeffs_label = 'cD' + str(i)
            ax[i-1].plot(list(range(0,len(wd[coeffs_label]))), wd[coeffs_label], label=coeffs_label)
    
        ax[i-1].set_title(wavelet_label + " " + data_label + " " + coeffs_label,fontsize=f_size)
        ax[i-1].set_xlabel('Days', fontsize=f_size)
        ax[i-1].set_ylabel('Displacement(mm)', fontsize=f_size)
        ax[i-1].xaxis.set_tick_params(labelsize=l_size)
        ax[i-1].yaxis.set_tick_params(labelsize=l_size)
        handles, labels = ax[i-1].get_legend_handles_labels()
        ax[i-1].legend(handles, labels, fontsize=f_size)
        ax[i-1].grid()
    coeffs_label = 'cA' + str(level_wt)
    y_x = wd[coeffs_label]
    for i in range(1, level_wt+1):
        coeffs_label = 'cD' + str(i)
        y_x += wd[coeffs_label]

    ax[level_wt+1].plot(list(range(0,len(y))),y, label="original data")
    ax[level_wt+1].plot(list(range(0,len(y_x))),y_x, label="reconstructed data")
    ax[level_wt+1].set_title(wavelet_label + " " + data_label + " original data ",fontsize=f_size)
    ax[level_wt+1].set_xlabel('Days', fontsize=f_size)
    ax[level_wt+1].set_ylabel('Displacement(mm)', fontsize=f_size)
    ax[level_wt+1].xaxis.set_tick_params(labelsize=l_size)
    ax[level_wt+1].yaxis.set_tick_params(labelsize=l_size)
    handles, labels = ax[level_wt+1].get_legend_handles_labels()
    ax[level_wt+1].legend(handles, labels, fontsize=f_size)
    ax[level_wt+1].grid()
    if pdf:
        pp = PdfPages(wavelet_label + '_decom.pdf')
        fig.savefig(pp, format='pdf')
        pp.close()