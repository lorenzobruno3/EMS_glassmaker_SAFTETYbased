# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:06:19 2025

@author: Lorenzo Bruno
"""

import cvxpy as cp
import numpy as np
import pandas as pa
import numpy as np

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
def function_dataPV_STE(time_end):
    dataPV = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="PV_generation_FINAL")
    def dict_func(xx):
        dict_build = {}
        dict_build = {ii: xx.iloc[ii+1, 0] for ii in range(0,time_end)}          #[kW]
        return dict_build
    PV_dict = dict_func(dataPV)
    PV = []
    for ii in range(time_end):
        PV.append(PV_dict[ii])    
    return PV
def function_PV_cost_STE(time_end):
    dataPV_cost = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="PV_OPEX")
    def dict_func_PV_cost(xx):
        dict_build_PV_cost = {}
        dict_build_PV_cost = {ii: xx.iloc[ii+1, 0] for ii in range(0,time_end)}          #[€/kW] 
        return dict_build_PV_cost
    PV_cost_dict = dict_func_PV_cost(dataPV_cost)
    PV_cost = []
    for ii in range(time_end):
        PV_cost.append(PV_cost_dict[ii])    
    return PV_cost
# =============================================================================
# ELECTRICAL LOAD
# =============================================================================
def function_loadSET_ele_STE(time_end):
    loadSET_ele = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="ELE_consumption_PLANTA")        # THESE are the synthetic DATA
    def dict_func_ele(xx):
        dict_build_ele = {}
        #dict_build_ele = {ii: abs(xx.iloc[ii+1, 0]) for ii in range(0,time_end)}          #[kW]
        dict_build_ele = {ii: abs(xx.iloc[ii+1+143, 0]) for ii in range(0,time_end)}          #[kW]
        return dict_build_ele
    load_dict_ele = dict_func_ele(loadSET_ele)
    load_ele = []
    for ii in range(time_end):
       load_ele.append(load_dict_ele[ii])
    return load_ele
def function_loadSET_ele_FUR_STE(time_end):
    loadSET_ele = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="ELE_consumption_FURN")        # THESE are the synthetic DATA
    def dict_func_ele(xx):
        dict_build_ele = {}
        #dict_build_ele = {ii: abs(xx.iloc[ii+1, 0]) for ii in range(0,time_end)}          #[kW]
        dict_build_ele = {ii: abs(xx.iloc[ii+1+143, 0]) for ii in range(0,time_end)}          #[kW]
        return dict_build_ele
    load_dict_ele = dict_func_ele(loadSET_ele)
    load_ele = []
    for ii in range(time_end):
       load_ele.append(load_dict_ele[ii])
    return load_ele
# =============================================================================
# GRID PURCHASE
# =============================================================================
def function_grid_price_purchase_STE(time_end):
    dataGRIDCOST = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="PCC (€_MWh)")
    def dict_func(xx):
        dict_build_gridcost_pur = {}
        #dict_build_gridcost_pur = {ii: xx.iloc[ii+1, 0]/1000 for ii in range(0,time_end)}          #[€/kWhe]
        dict_build_gridcost_pur = {ii: xx.iloc[ii+1+143, 0]/1000 for ii in range(0,time_end)}          #[€/kWhe]
        return dict_build_gridcost_pur
    GRIDCOST_PUR_dict = dict_func(dataGRIDCOST)
    GRIDCOST_PUR = []
    for ii in range(time_end):
        GRIDCOST_PUR.append(GRIDCOST_PUR_dict[ii])    
    return GRIDCOST_PUR
# =============================================================================
# NATURAL GAS
# =============================================================================
def function_NG_cost_STE(time_end):
    dataNG_cost = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="price_NG")
    def dict_func_NG_cost(xx):
        dict_build_NG_cost = {}
        #dict_build_NG_cost = {ii: xx.iloc[ii+1, 0] for ii in range(0,time_end)}          #[€/kWh*h]
        dict_build_NG_cost = {ii: xx.iloc[ii+1+143, 0] for ii in range(0,time_end)}          #[€/kWh*h]
        return dict_build_NG_cost
    NG_cost_dict = dict_func_NG_cost(dataNG_cost)
    NG_cost = []
    for ii in range(time_end):
        NG_cost.append(NG_cost_dict[ii])    
    return NG_cost
# =============================================================================
# THERMAL LOAD
# =============================================================================
def function_loadSET_th_STE(time_end):
    loadSET_th = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="NG_meltingend")
    def dict_func_th(xx):
        dict_build_th = {}
        #dict_build_th = {ii: abs(xx.iloc[ii+1, 1]) for ii in range(0,time_end)}          #[kWh/h]
        dict_build_th = {ii: abs(xx.iloc[ii+1+143, 1]) for ii in range(0,time_end)}          #[kWh/h]
        return dict_build_th
    load_dict_th = dict_func_th(loadSET_th)
    load_th = []
    for ii in range(time_end):
       load_th.append(load_dict_th[ii])
    return load_th
def function_burner_th_STE(time_end):
    burner_set_th = pa.read_excel("profiles_STEKLARNA.xlsx", sheet_name="burner_efficiency")
    def dict_func_burner_th(xx):
        dict_build_burner_th = {}
        #dict_build_burner_th = {ii: xx.iloc[ii+1, 0] for ii in range(0,time_end)}          #[kW]
        dict_build_burner_th = {ii: xx.iloc[ii+1+143, 0] for ii in range(0,time_end)}          #[kW]
        return dict_build_burner_th
    load_dict_burner_th = dict_func_burner_th(burner_set_th)
    burner_th_eff = []
    for ii in range(time_end):
       burner_th_eff.append(load_dict_burner_th[ii])
    return burner_th_eff


