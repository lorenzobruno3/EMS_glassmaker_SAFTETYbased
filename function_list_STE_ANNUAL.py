# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:06:19 2025

@author: Lorenzo Bruno
"""

import cvxpy as cp
import numpy as np
import pandas as pa
import numpy as np

#from realtime_main import time_end_annual


#def function_dataPV_STE(time_end):
#    dataPV = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="PV_generation")
#    def dict_func(xx):
#        dict_build = {}
#        dict_build = {ii: xx.iloc[ii+1, 0] for ii in range(0,time_end)}          #[kW]
#        return dict_build
#    PV_dict = dict_func(dataPV)
#    PV = []
#    for ii in range(time_end):
#        PV.append(PV_dict[ii])    
#    return PV
def function_dataPV_STE(time_end_annual):
    dataPV = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func(xx):
        dict_build = {}
        dict_build = {ii: xx.iloc[ii+1, 8] for ii in range(0,time_end_annual)}          #[kW]
        return dict_build
    PV_dict = dict_func(dataPV)
    PV = []
    for ii in range(time_end_annual):
        PV.append(PV_dict[ii])    
    return PV
def function_PV_cost_STE(time_end_annual):
    dataPV_cost = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func_PV_cost(xx):
        dict_build_PV_cost = {}
        dict_build_PV_cost = {ii: xx.iloc[ii+1, 8] for ii in range(0,time_end_annual)}          #[€/kW]
        return dict_build_PV_cost
    PV_cost_dict = dict_func_PV_cost(dataPV_cost)
    PV_cost = []
    for ii in range(time_end_annual):
        PV_cost.append(PV_cost_dict[ii])    
    return PV_cost
# =============================================================================
# ELECTRICAL LOAD
# =============================================================================
def function_loadSET_ele_STE(time_end_annual):
    loadSET_ele = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")        # THESE are the synthetic DATA
    def dict_func_ele(xx):
        dict_build_ele = {}
        dict_build_ele = {ii: xx.iloc[ii+1, 14] for ii in range(0,time_end_annual)}          #[kW]
        return dict_build_ele
    load_dict_ele = dict_func_ele(loadSET_ele)
    load_ele = []
    for ii in range(time_end_annual):
       load_ele.append(load_dict_ele[ii])
    return load_ele
def function_loadSET_ele_FUR_STE(time_end_annual):
    loadSET_ele = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")        # THESE are the synthetic DATA
    def dict_func_ele(xx):
        dict_build_ele = {}
        dict_build_ele = {ii: xx.iloc[ii+1, 2] for ii in range(0,time_end_annual)}          #[kW]
        return dict_build_ele
    load_dict_ele = dict_func_ele(loadSET_ele)
    load_ele = []
    for ii in range(time_end_annual):
       load_ele.append(load_dict_ele[ii])
    return load_ele
# =============================================================================
# GRID PURCHASE
# =============================================================================
def function_grid_price_purchase_STE(time_end_annual):
    dataGRIDCOST = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func(xx):
        dict_build_gridcost_pur = {}
        xx.iloc[:, 9] = pa.to_numeric(
            xx.iloc[:, 9]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .str.replace('€', '', regex=False),
            errors='coerce'
        )
        #xx.iloc[:, 9] = pa.to_numeric(xx.iloc[:, 9], errors='coerce')
        dict_build_gridcost_pur = {ii: xx.iloc[ii + 1, 9] /1000 for ii in range(0, time_end_annual)}
        #dict_build_gridcost_pur = {ii: xx.iloc[ii+1, 9]/1000 for ii in range(0,time_end_annual)}          #[€/kWhe]
        return dict_build_gridcost_pur
    GRIDCOST_PUR_dict = dict_func(dataGRIDCOST)
    GRIDCOST_PUR = []
    for ii in range(time_end_annual):
        GRIDCOST_PUR.append(GRIDCOST_PUR_dict[ii])    
    return GRIDCOST_PUR
# =============================================================================
# NATURAL GAS
# =============================================================================
def function_NG_cost_STE(time_end_annual):
    dataNG_cost = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func_NG_cost(xx):
        dict_build_NG_cost = {}
        dict_build_NG_cost = {ii: xx.iloc[ii+1, 10]/1000 for ii in range(0,time_end_annual)}          #[€/kWh*h]
        return dict_build_NG_cost
    NG_cost_dict = dict_func_NG_cost(dataNG_cost)
    NG_cost = []
    for ii in range(time_end_annual):
        NG_cost.append(NG_cost_dict[ii])    
    return NG_cost
# =============================================================================
# THERMAL LOAD
# =============================================================================
def function_loadSET_th_STE(time_end_annual):
    loadSET_th = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func_th(xx):
        dict_build_th = {}
        dict_build_th = {ii: abs(xx.iloc[ii+1, 5]) for ii in range(0,time_end_annual)}          #[kWh/h]
        return dict_build_th
    load_dict_th = dict_func_th(loadSET_th)
    load_th = []
    for ii in range(time_end_annual):
       load_th.append(load_dict_th[ii])
    return load_th
time_end_annual = 35423 #GIOCO DEL PORCELLINO
def function_burner_th_STE(time_end):
    burner_set_th = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func_burner_th(xx):
        dict_build_burner_th = {}
        dict_build_burner_th = {ii: xx.iloc[ii+1, 15] for ii in range(0,time_end_annual)}          #[kW]
        return dict_build_burner_th
    load_dict_burner_th = dict_func_burner_th(burner_set_th)
    burner_th_eff = []
    for ii in range(time_end_annual):
       burner_th_eff.append(load_dict_burner_th[ii])
    return burner_th_eff
# =============================================================================
# EU CARBON PERMITS
# =============================================================================
def function_CARBON_PERMITS(time_end_annual):
    dataCARBON_PERMITS_cost = pa.read_excel("profiles_STEKLARNA_ANNUAL.xlsx", sheet_name="data")
    def dict_func_CARBON_PERMITS(xx):
        dict_build_CARBON_PERMITS = {}
        dict_build_CARBON_PERMITS = {ii: xx.iloc[ii+1, 20]/1000 for ii in range(0,time_end_annual)}          #[€/kgCO2]
        return dict_build_CARBON_PERMITS

    CARBON_PERMITS_dict = dict_func_CARBON_PERMITS(dataCARBON_PERMITS_cost)
    CARBON_PERMITS = []
    for ii in range(time_end_annual):
        CARBON_PERMITS.append(CARBON_PERMITS_dict[ii])
    return CARBON_PERMITS


