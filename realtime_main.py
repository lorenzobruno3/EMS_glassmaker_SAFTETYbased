#Using license file /path/to/license/file/gurobi.lic # -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:33:18 2024
CVXPY application for convex optimisation
@author: bruno
"""
# ==================================================================================================================================================
## MOTIVAZIONE
# ==================================================================================================================================================
#raga ce la faremo
#sempre raga, sempre
#chiudiamo questa mmerda
# =========================================================================
## LIBRARIES
# =========================================================================
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import pandas as pa
import function_list as ff
import function_list_SCENARIOS as ffss
import function_list_STE as ste
import function_list_STE_ANNUAL as steann
import seaborn as ss
from tqdm import tqdm
#import debug_nan as de
#import excel_export as ee
#import function_loadSET
# =========================================================================
## REMEMBER
# =========================================================================
"""
SCENARIO0: 
    solo NG
SCENARIO1: 
    ely con eff=0
    stato initial tank=0
SCENARIO2: 
    perc produzione 5%
    ely con eff
    stato initial tank 50%   
SCENARIO3:
    perc produzione 65%https://climate.ec.europa.eu/document/download/92ec0ab3-24cf-4814-ad59-81c15e310bea_en?filename=2024_carbon_market_report_en.pdf
"""
# ==================================================================================================================================================
## GENERAL DATA
# ==================================================================================================================================================
time_end = 14*(24)#2*7*24+24               #2*(24*4)
time_end_annual = 35423
index_time = list(range(time_end))
planning_horizon = 24#int((24/2)*4) # it must be INTEGER NUMBER
LHV_H2 = 33.33                                                                      # [kWh/kgH2]
LHV_NG = 10.4                                                                       # [kWh/Nm3]
MM_CO2 = 44                                                                         # [g/mol]
MM_H2 = 2                                                                         # [g/mol]
MM_H2O = 18                                                                         # [g/mol]
MM_CH4 = 16                                                                         # [g/mol]
MM_O2 = 32                                                                         # [g/mol]
CO2_prod_rate = MM_CO2/MM_CH4                                                       # [-]
density_NG = 0.712                                                                  # [kg/Nm3]
density_H2O = 1000                                                                  # [kg/m3]
density_O2 = 1.429                                                                  # [kg/m3]
price_water = 2                                                                     # [euro/m3]
price_O2 = 0.07                                                                     # [euro/m3]
price_O2_kWh = 0.13                                                                 # [euro/kWh]
# =========================================================================
## PV DATA UPLOAD FROM EXCEL 
# =========================================================================
#P_pv_l_ima = ff.function_dataPV(time_end)
#P_pv_l_sce = ffss.function_dataPV_SCENARIOS(time_end)    
P_pv_l = ste.function_dataPV_STE(time_end) #P_pv_l_sce                                                #area is supposed 1000 m2
P_pv_l_ber = steann.function_dataPV_STE(time_end_annual) #P_pv_l_sce                                                #area is supposed 1000 m2
PV_cost_l = ste.function_PV_cost_STE(time_end)
PV_cost_l_ber = steann.function_PV_cost_STE(time_end_annual)
# =========================================================================
## ELECTRICAL LOAD 
# =========================================================================
#P_load_ele_ima = ff.function_loadSET_ele(time_end)
#P_load_ele_sce = ffss.function_loadSET_ele_SCENARIOS(time_end)
P_load_ele_l = ste.function_loadSET_ele_STE(time_end) #P_load_ele_sce
P_load_ele_l_ber = steann.function_loadSET_ele_STE(time_end_annual) #P_load_ele_sce
P_load_ele_FUR_l = ste.function_loadSET_ele_FUR_STE(time_end)   #
P_load_ele_FUR_l_ber = steann.function_loadSET_ele_FUR_STE(time_end_annual)   #
# =========================================================================
## GRID MANAGEMENT
# =========================================================================
#grid_price_purchase_ima = ff.function_grid_price_purchase(time_end)
#grid_price_purchase_sce = ffss.function_grid_price_purchase_SCENARIOS(time_end)
grid_price_purchase_l =  ste.function_grid_price_purchase_STE(time_end) #grid_price_purchase_sce
grid_price_purchase_l_ber =  steann.function_grid_price_purchase_STE(time_end_annual) #grid_price_purchase_sce
# =========================================================================
## COMPRESSOR
# =========================================================================
compressor = ff.function_compressorclass("COMP2")
# =========================================================================
## ELECTROLYSER
# =========================================================================
electro1 = ff.function_electro1class(planning_horizon, "PEM")
#electro1 = ff.function_electro1class(planning_horizon, "PEM_alt")
#electro1 = ff.function_electro1class(planning_horizon, "PEM_0")
#electro1 = ff.function_electro1class(planning_horizon, "PEM_A")
#electro1 = ff.function_electro1class(planning_horizon, "PEM_10")
#electro1 = ff.function_electro1class(planning_horizon, "PEM_100")
# =========================================================================
## TANK
# =========================================================================
#tank = ff.function_tankclass(planning_horizon, "TYPE2")
#tank = ff.function_tankclass(planning_horizon, "TYPE100")
#tank = ff.function_tankclass(planning_horizon, "TYPE20_saf")
tank = ff.function_tankclass(planning_horizon, "TYPE16_saf")
#tank = ff.function_tankclass(planning_horizon, "TYPE01")
# =========================================================================
## NG DATA UPLOAD FROM EXCEL 
# =========================================================================
#NG_price_ima = ff.function_NG_cost(time_end)
#NG_price_sce = ffss.function_NG_cost(time_end)
NG_price_l = ste.function_NG_cost_STE(time_end) #NG_price_sce
NG_price_l_ber = steann.function_NG_cost_STE(time_end_annual) #NG_price_sce
# =========================================================================
## CARBON_PERMITS
# =========================================================================
CARBON_PERMITS_price_l = ff.function_CARBON_PERMITS(time_end)
CARBON_PERMITS_price_l_ber = steann.function_CARBON_PERMITS(time_end_annual)
# =========================================================================
## BURNER  
# =========================================================================
burner_th_eff_l = ste.function_burner_th_STE(time_end)
burner_th_eff_l_ber = steann.function_burner_th_STE(time_end_annual)
# =========================================================================
## THERMAL LOAD 
# =========================================================================
#P_load_th_ima = ff.function_loadSET_th(time_end)
#P_load_th_sce = ffss.function_loadSET_th_SCENARIOS(time_end)
P_load_th_l = ste.function_loadSET_th_STE(time_end) #P_load_th_sce
P_load_th_l_ber = steann.function_loadSET_th_STE(time_end_annual) #P_load_th_sce
# =========================================================================
## ENVIRONMENTAL
# =========================================================================
environmental = ff.function_environmentalclass("environmental")
# ==================================================================================================================================================
## DEFINITION OF MOMO MODEL
# ==================================================================================================================================================
P_0 = electro1.P_ely_0
P_STB = electro1.P_ely_STB
P_ON = electro1.P_ely_ON_min
P_max = electro1.rated_power
M = 1e9    
d_0_OFFely = 0
d_0_STBely = 0
d_0_ONely = 1
# ==================================================================================================================================================
## DEFINITION OF VARIABLES
# ==================================================================================================================================================
# all the variables have the length of planning_horizon 
P_grid = cp.Variable(planning_horizon, nonneg=True)
P_pv_var = cp.Variable(planning_horizon, nonneg=True)
P_ele_furnace = cp.Variable(planning_horizon, nonneg=True)
P_ely_in = electro1.P_ely_in
P_ely_out = electro1.P_ely_out
P_ely_out_bur = cp.Variable(planning_horizon, nonneg=True)
P_comp_in = cp.Variable(planning_horizon, nonneg=True)
P_comp_grid = cp.Variable(planning_horizon, nonneg=True)
P_tank_in = tank.P_tank_in
P_tank_out = tank.P_tank_out
P_node_C = cp.Variable(planning_horizon, nonneg=True)
P_node_D = cp.Variable(planning_horizon, nonneg=True)
P_NG = cp.Variable(planning_horizon, nonneg=True)
CO2_NG = cp.Variable(planning_horizon, nonneg=True)
mass_O2_tot = cp.Variable(planning_horizon, nonneg=True)
P_bur_in = cp.Variable(planning_horizon, nonneg=True)
P_load_th_TOT = cp.Variable(planning_horizon, nonneg=True)
# =========================================================================
## PLOT ARRAYS  
# =========================================================================
P_grid_plot = []
P_pv_var_plot =[]
P_ele_furnace_plot = []
P_ely_in_plot = []
P_ely_out_plot = []
P_ely_out_previous = []
P_ely_out_bur_plot = []
P_standby_plot = []
P_comp_in_plot = []
P_comp_grid_plot = []
P_tank_in_plot = []
P_tank_out_plot = []
soc_tank_plot = []
P_node_C_plot = []
P_bur_in_plot = []
P_pv_plot = []
P_NG_plot = []
P_load_th_TOT_plot = []
CO2_NG_plot = []
C_pv_cost_plot = []
C_grid_cost_plot = []
ELY_cost_overall_plot = []
C_comp_cost_plot = []
C_tank_cost_plot = []
C_NG_cost_plot = []
C_H2O_cost_plot = []
C_O2_cost_plot = []
cost_emission_grid_plot = []
cost_emission_NG_plot = []
cost_total_opt_plot = []
tot_cost = []
DeltaON_PLOT = []
DeltaOFF_PLOT = []
DeltaSTB_PLOT = []
SigmaONOFF_PLOT = []
SigmaOFFON_PLOT = []
SigmaONSTB_PLOT = []
SigmaSTBON_PLOT = []
SigmaSTBOFF_PLOT = []
SigmaOFFSTB_PLOT = []
state_costs_plot = [] 
state_costs_STB_plot = []
state_costs_ON_plot = [] 
transition_costs_plot = []
mass_H2_PLOT= []
mass_H2O_PLOT = []
mass_O2_burn_PLOT = []
mass_O2_fromH2_PLOT = []
State1_tank_PLOT = []
State2_tank_PLOT = []
State3_tank_PLOT = []
State4_tank_PLOT = []
State5_tank_PLOT = []
UHI_tank_PLOT = []
State1_ely_PLOT = []
State2_ely_PLOT = []
State3_ely_PLOT = []
State4_ely_PLOT = []
State5_ely_PLOT = []
UHI_ely_PLOT = []
C_tank_UHI_PLOT = []
# ==================================================================================================================================================
## EMS DEFINITION PLOT
# ==================================================================================================================================================
P_ely_in_EMS_plot = np.zeros((time_end-planning_horizon,planning_horizon))
P_ely_out_EMS_plot = np.zeros((time_end-planning_horizon,planning_horizon))
DeltaON_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
DeltaOFF_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
DeltaSTB_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaONOFF_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaOFFON_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaONSTB_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaSTBON_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaSTBOFF_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
SigmaOFFSTB_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
state_costs_EMS_plot = np.zeros((time_end-planning_horizon,planning_horizon)) 
state_costs_EMS_STB_plot = np.zeros((time_end-planning_horizon,planning_horizon))
state_costs_EMS_ON_plot = np.zeros((time_end-planning_horizon,planning_horizon)) 
transition_costs_EMS_plot = np.zeros((time_end-planning_horizon,planning_horizon))
mass_H2_EMS = np.zeros((time_end-planning_horizon,planning_horizon))
mass_H2O_EMS = np.zeros((time_end-planning_horizon,planning_horizon))
mass_O2_burn_EMS = np.zeros((time_end-planning_horizon,planning_horizon))
mass_O2_fromH2_EMS = np.zeros((time_end-planning_horizon,planning_horizon))
State1_tank_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
State2_tank_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
State3_tank_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
State4_tank_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
State5_tank_EMS_PLOT = np.zeros((time_end-planning_horizon,planning_horizon))
# ==================================================================================================================================================
## EMS DEFINITION
# ==================================================================================================================================================
P_pv = np.array(P_pv_l)
PV_cost = np.array(PV_cost_l)
#P_hydro = np.array(P_hydro_l)
#HYDRO_cost = np.array(HYDRO_cost_l)
P_load_ele = np.array(P_load_ele_l)
P_load_ele_FUR = np.array(P_load_ele_FUR_l)
grid_price_purchase = np.array(grid_price_purchase_l)
NG_price = np.array(NG_price_l)
CARBON_PERMITS_price = np.array(CARBON_PERMITS_price_l)
burner_th_eff = np.array(burner_th_eff_l)
P_ely_rated = np.ones(P_ely_in.size)*electro1.rated_power
P_load_th = np.array(P_load_th_l)
# time_trans = np.eye(planning_horizon, k = -1)       # matrice da moltiplicare con il vettore x
# aux = np.array(P_ely_out_plot) - np.eye(3*7*24-24, k = -1)@np.array(P_ely_out_plot)
# max(aux)
aa = np.array([[ -1],[ 1]])
A_temporal = np.kron(np.eye(planning_horizon-1), aa)
AA = np.column_stack((A_temporal, np.zeros(A_temporal.shape[0]))) - np.column_stack((np.zeros(A_temporal.shape[0]), A_temporal))
bb = np.kron(np.ones(planning_horizon-1), np.array([electro1.RAMP, electro1.RAMP]))

P_ely_inicial = electro1.rated_power*electro1.efficiency 

for index_cont in tqdm(range(time_end-planning_horizon)):

    CO2_NG = P_NG / LHV_NG * density_NG * CO2_prod_rate # [kW]*[Nm3/kWh]*[kg/Nm3]
    mass_H2 = P_ely_out / LHV_H2
    mass_O2_burn = P_NG / LHV_NG * (1/2) * (MM_O2 / MM_CH4)  # [kW]*[Nm3/kWh]*[kg/Nm3]
    mass_O2_fromH2 = mass_H2 * (1/ 2) * (MM_O2 / MM_H2)  # [kg_H2O]##########AAAAAA##########
    mass_H2O = mass_H2 * (MM_H2O / MM_H2)

    C_pv_cost = cp.multiply(P_pv_var,0*PV_cost[index_cont:index_cont + planning_horizon])
    #P_hydro_cost = np.multiply(P_hydro[index_cont:index_cont + planning_horizon],HYDRO_cost[index_cont:index_cont + planning_horizon])
    C_grid_cost = cp.multiply(P_grid, grid_price_purchase[index_cont:index_cont + planning_horizon])
    ####################C_ely_cost = P_ely_in*electro1.OPEX_var 
    C_comp_cost = P_comp_in*compressor.OPEX ##########AAAAAA##########
    C_tank_cost = P_tank_in/LHV_H2*tank.OPEX ##########AAAAAA##########
    C_tank_UHI = electro1.UHI_ely/6*tank.CAPEX*tank.rated_capacity_kg ##########AAAAAA##########
    C_NG_cost = cp.multiply(P_NG,NG_price[index_cont:index_cont + planning_horizon])
    #cost_emission_grid = cp.multiply(P_grid, environmental.carbon_cost_indirect)
    #cost_emission_NG = cp.multiply(CO2_NG, environmental.carbon_cost) #### ENVIRONMENTAL FIXED COST
    cost_emission_grid = cp.multiply(P_grid, CARBON_PERMITS_price[index_cont:index_cont + planning_horizon]*environmental.ECI)
    cost_emission_NG = cp.multiply(CO2_NG, CARBON_PERMITS_price[index_cont:index_cont + planning_horizon])
    C_H2O_cost = price_water/density_H2O*mass_H2O
    C_O2_cost = price_O2 / density_O2 * mass_O2_burn
    P_load_th_TOT = P_load_th[index_cont:index_cont + planning_horizon] + P_load_ele_FUR[index_cont:index_cont + planning_horizon]
    # =========================================================================
    ## RAMP UP
    # ========================================================================= 
    #   if index_cont == 0:  ##########AAAAAA##########
    #       P_ely_out_previous == 0 ##########AAAAAA##########
    #   else:   ##########AAAAAA##########
    #       P_ely_out_previous == P_ely_out_plot[index_cont-1] ##########AAAAAA##########
        
    #variables_to_check = {
    #    "P_pv": P_pv,
    #    "PV_cost": PV_cost,
    #    # "P_hydro": P_hydro,
    #    # "HYDRO_cost": HYDRO_cost,
    #    "P_load_ele": P_load_ele,
    #    "grid_price_purchase": grid_price_purchase,
    #    "NG_price": NG_price,
    #    "burner_th_eff": burner_th_eff,
    #    "P_ely_rated": P_ely_rated,
    #    "P_load_th": P_load_th,
    #    "A_temporal": A_temporal,
    #    "AA": AA,
    #    "bb": bb,
    #    "P_ely_inicial": P_ely_inicial,
    #    "C_pv_cost": C_pv_cost,
    #    "C_ely_cost": C_ely_cost,
    #    "C_comp_cost": C_comp_cost,
    #    "C_tank_cost": C_tank_cost,
    #    "C_NG_cost": C_NG_cost,
    #    "cost_emission_grid": cost_emission_grid,
    #    "cost_emission_NG": cost_emission_NG,
    #}
    # === Run the checks ===
    #print("\n🔎 Checking variables for NaN values:")
    #for var_name, var_value in variables_to_check.items():
    #    de.check_nan_in_array(var_name, var_value)
    #print("✅ NaN check complete.\n")
    # =========================================================================
    ## OBJECTIVE FUNCTION
    # =========================================================================   
    state_costs = electro1.power_standby*electro1.c_STB + electro1.P_ely_in*electro1.c_ON  ##########AAAAAA##########
    transition_costs = electro1.SigmaOFFSTB*electro1.c_OFF_STB + electro1.SigmaSTBOFF*electro1.c_STB_OFF + electro1.SigmaSTBON*electro1.c_STB_ON + electro1.SigmaONSTB*electro1.c_ON_STB + electro1.SigmaONOFF*electro1.c_ON_OFF + electro1.SigmaOFFON*electro1.c_OFF_ON  ##########AAAAAA##########
    cost_total_opt = C_grid_cost + C_pv_cost + C_NG_cost + cost_emission_grid + cost_emission_NG + state_costs + transition_costs + C_comp_cost + C_tank_cost + C_tank_UHI + C_H2O_cost + C_O2_cost  ##########AAAAAA##########     ####################           + P_hydro_cost + C_ely_cost
    objective = cp.sum(cost_total_opt)     #  
    # =========================================================================
    ## CONSTRAINTS
    # =========================================================================
    tank_constraints = tank.get_constraints()  ##########AAAAAA##########
    #####FOR#####electro1_constraints_MOMO = electro1.get_constraints_MOMO(P_0, P_STB, P_ON, P_max, M, d_0_OFFely, d_0_STBely, d_0_ONely)  
    electro1_constraints_MOMO = electro1.get_constraints_MOMO(P_0, P_STB, P_ON, P_max, M, d_0_OFFely, d_0_STBely, d_0_ONely, P_ely_inicial) ##########AAAAAA##########
    other_constraints = [P_pv_var <= P_pv[index_cont:index_cont + planning_horizon],
                         P_grid + P_pv_var == P_load_ele[index_cont:index_cont + planning_horizon] + P_ely_in + electro1.power_standby + P_comp_grid + P_ele_furnace , ##########AAAAAA##########                                  #    + P_hydro[index_cont:index_cont + planning_horizon]
                         P_ele_furnace <= 0.12*P_load_th_TOT,#P_load_th[index_cont:index_cont + planning_horizon],
                         P_ele_furnace >= 0.08*P_load_th_TOT,#P_load_th[index_cont:index_cont + planning_horizon],
                         P_ely_out == electro1.efficiency*P_ely_in, ##########AAAAAA##########
                         P_ely_in <= P_ely_rated,       # electro1.high_range*P_ely_rated, ##########AAAAAA##########
                         #####P_ely_out - time_trans@P_ely_out <= electro1.RAMP*np.ones_like(P_ely_out),
                         #####P_ely_out - time_trans@P_ely_out >= - electro1.RAMP*np.ones_like(P_ely_out),
                         AA@P_ely_out <= bb,  ##########AAAAAA##########
                         #P_ely_out[0]-P_ely_inicial<= electro1.RAMP,  ##########PPPP##########
                         #-P_ely_out[0]+P_ely_inicial<= electro1.RAMP,  ##########PPPP##########
                         #####P_ely_in >= electro1.low_range*P_ely_rated,   ES INFEASIBLEEE
                         #####P_ely_out[0] - P_ely_out_previous <= electro1.RAMP,
                         #####-P_ely_out[0] + P_ely_out_previous <= electro1.RAMP,
                         P_node_C >= 0.05*P_load_th_TOT,    ##########AAAAAA##########       
                         P_ely_out == P_ely_out_bur + P_comp_in,      ##########AAAAAA##########       
                         P_comp_grid == P_comp_in*compressor.compressor_work/LHV_H2,  ##########AAAAAA##########
                         mass_O2_tot == mass_O2_fromH2 + mass_O2_burn,
                         P_comp_in == P_tank_in,  ##########AAAAAA##########
                         P_ely_out_bur + P_tank_out == P_node_C,##########AAAAAA##########
                         P_NG + P_node_C == P_node_D, ##########AAAAAA##########  
                         P_node_D == P_bur_in,    
                         P_load_th_TOT == cp.multiply(P_bur_in, burner_th_eff[index_cont:index_cont + planning_horizon]) + P_ele_furnace
    ]

    constraints = other_constraints + tank_constraints + electro1_constraints_MOMO    ##########AAAAAA########## 
    # =========================================================================
    ## PROBLEM SOLVING
    # =========================================================================
    problem = cp.Problem(cp.Minimize(objective), constraints)  #poner decimales?
    #print(problem)
    problem.solve(verbose=False,solver=cp.GUROBI)     # True to print the solving/ False to not print it #cp.MOSEK
  

    ## STATUS CHECK
    #for i, constraint in enumerate(constraints):
    #    if hasattr(constraint, "expr") and constraint.expr is not None:
    #        print("Constraint:", constraint)
    #        print("Expression:", constraint.expr)  # Should not be None

    #        residual = constraint.violation()
    #        print(f"Constraint {i}: Residual = {residual}")
    #    else:
    #        print(f"Constraint {i}: Skipped (no expression)")

    # else:
    #     print("Problem solved successfully.")
    #     print(f"Optimal value: {problem.value}")
    #     print(f"x: {x.value}, y: {y.value}")

    P_ely_inicial = P_ely_in[1].value  ##########AAAAAA########## METTO ! INVECE DI ZERO  P_ely_inicial = P_ely_out[0].value
    d_0_OFFely = electro1.DeltaOFF[1].value  ##########AAAAAA##########  METTO ! INVECE DI ZERO  d_0_OFFely = electro1.DeltaOFF[0].value
    d_0_STBely = electro1.DeltaSTB[1].value  ##########AAAAAA########## METTO ! INVECE DI ZERO d_0_STBely = electro1.DeltaSTB[0].value 
    d_0_ONely = electro1.DeltaON[1].value  ##########AAAAAA########## METTO ! INVECE DI ZERO  d_0_ONely = electro1.DeltaON[0].value
    #print(d_0_OFFely)
    #print(d_0_STBely)
    #print(d_0_ONely)
    
    tank.update()##########AAAAAA########## 
    tot_cost.append(objective.value)
    P_grid_plot.append(P_grid[0].value)
    P_ele_furnace_plot.append(P_ele_furnace[0].value)  ##########AAAAAA##########
    P_ely_in_plot.append(P_ely_in[0].value)  ##########AAAAAA##########
    P_ely_out_plot.append(P_ely_out[0].value)  ##########AAAAAA##########
    P_ely_out_bur_plot.append(P_ely_out_bur[0].value)  ##########AAAAAA##########
    P_standby_plot.append(electro1.power_standby[0].value)  ##########AAAAAA##########
    C_pv_cost_plot.append(C_pv_cost[0].value)
    P_pv_var_plot.append(P_pv_var[0].value)
    C_H2O_cost_plot.append(C_H2O_cost[0].value)#######################################################################################
    C_O2_cost_plot.append(C_O2_cost[0].value)
    C_grid_cost_plot.append(C_grid_cost[0].value)
    P_comp_in_plot.append(P_comp_in[0].value)   ##########AAAAAA##########
    P_comp_grid_plot.append(P_comp_grid[0].value)   ##########AAAAAA##########
    P_tank_in_plot.append(P_tank_in[0].value)   ##########AAAAAA##########
    P_tank_out_plot.append(P_tank_out[0].value)  ##########AAAAAA########## 
    soc_tank_plot.append(tank.soc[0].value)  ##########AAAAAA##########
    P_node_C_plot.append(P_node_C[0].value)  ##########AAAAAA##########
    P_bur_in_plot.append(P_bur_in[0].value)
    ######ELY_cost_overall_plot.append(C_ely_cost[0].value)
    C_comp_cost_plot.append(C_comp_cost[0].value)  ##########AAAAAA##########
    C_tank_cost_plot.append(C_tank_cost[0].value)  ##########AAAAAA##########
    cost_emission_grid_plot.append(cost_emission_grid[0].value)
    cost_emission_NG_plot.append(cost_emission_NG[0].value)
    cost_total_opt_plot.append(cost_total_opt[0].value)   
    P_NG_plot.append(P_NG[0].value)
    C_NG_cost_plot.append(C_NG_cost[0].value)
    CO2_NG_plot.append(CO2_NG[0].value)
    P_load_th_TOT_plot.append(P_load_th_TOT[0])
    DeltaON_PLOT.append(electro1.DeltaON[0].value)  ##########AAAAAA##########
    DeltaOFF_PLOT.append(electro1.DeltaOFF[0].value)  ##########AAAAAA##########
    DeltaSTB_PLOT.append(electro1.DeltaSTB[0].value)  ##########AAAAAA##########
    SigmaONOFF_PLOT.append(electro1.SigmaONOFF[0].value)
    SigmaOFFON_PLOT.append(electro1.SigmaOFFON[0].value)
    SigmaONSTB_PLOT.append(electro1.SigmaONSTB[0].value)
    SigmaSTBON_PLOT.append(electro1.SigmaSTBON[0].value)
    SigmaSTBOFF_PLOT.append(electro1.SigmaSTBOFF[0].value)
    SigmaOFFSTB_PLOT.append(electro1.SigmaOFFSTB[0].value)
    state_costs_plot.append(state_costs[0].value)  ##########AAAAAA##########
    state_costs_STB_plot.append(electro1.power_standby[0].value*electro1.c_STB)  ##########AAAAAA########## 
    state_costs_ON_plot.append(electro1.P_ely_in[0].value*electro1.c_ON)  ##########AAAAAA##########
    transition_costs_plot.append(transition_costs[0].value)  ##########AAAAAA##########
    mass_H2_PLOT.append(mass_H2[0].value)
    mass_H2O_PLOT.append(mass_H2O[0].value)
    mass_O2_burn_PLOT.append(mass_O2_burn[0].value)
    mass_O2_fromH2_PLOT.append(mass_O2_fromH2[0].value)
    State1_tank_PLOT.append(tank.State1[0].value)
    State2_tank_PLOT.append(tank.State2[0].value)
    State3_tank_PLOT.append(tank.State3[0].value)
    State4_tank_PLOT.append(tank.State4[0].value)
    State5_tank_PLOT.append(tank.State5[0].value)
    UHI_tank_PLOT.append(tank.UHI_tank[0].value)
    State1_ely_PLOT.append(electro1.State1[0].value)
    State2_ely_PLOT.append(electro1.State2[0].value)
    State3_ely_PLOT.append(electro1.State3[0].value)
    State4_ely_PLOT.append(electro1.State4[0].value)
    State5_ely_PLOT.append(electro1.State5[0].value)
    UHI_ely_PLOT.append(electro1.UHI_ely[0].value)
    C_tank_UHI_PLOT.append(C_tank_UHI[0].value)
    #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO0000000000000000000000000000000000000000000000000000
    DeltaON_EMS_PLOT[index_cont] = electro1.DeltaON.value
    DeltaOFF_EMS_PLOT[index_cont] = electro1.DeltaOFF.value
    DeltaSTB_EMS_PLOT[index_cont] = electro1.DeltaSTB.value
    SigmaONOFF_EMS_PLOT[index_cont] = electro1.SigmaONOFF.value
    SigmaOFFON_EMS_PLOT[index_cont] = electro1.SigmaOFFON.value
    SigmaONSTB_EMS_PLOT[index_cont] = electro1.SigmaONSTB.value
    SigmaSTBON_EMS_PLOT[index_cont] = electro1.SigmaSTBON.value
    SigmaSTBOFF_EMS_PLOT[index_cont] = electro1.SigmaSTBOFF.value
    SigmaOFFSTB_EMS_PLOT[index_cont] = electro1.SigmaOFFSTB.value
    state_costs_EMS_plot[index_cont] = state_costs.value 
    state_costs_EMS_STB_plot[index_cont] = electro1.power_standby.value*electro1.c_STB
    state_costs_EMS_ON_plot[index_cont] = electro1.P_ely_in.value*electro1.c_ON
    transition_costs_EMS_plot[index_cont] = transition_costs.value
    P_ely_in_EMS_plot[index_cont] = P_ely_in.value
    P_ely_out_EMS_plot[index_cont] = P_ely_out.value
    mass_H2_EMS[index_cont] = mass_H2.value  # [kg_H2]
    mass_H2O_EMS[index_cont] = mass_H2O.value  # [kg_H20]
    mass_O2_burn_EMS[index_cont] = mass_O2_burn.value
    mass_O2_fromH2_EMS[index_cont] = mass_O2_fromH2.value
    State1_tank_EMS_PLOT[index_cont] = tank.State1.value
    State2_tank_EMS_PLOT[index_cont] = tank.State2.value
    State3_tank_EMS_PLOT[index_cont] = tank.State3.value
    State4_tank_EMS_PLOT[index_cont] = tank.State4.value
    State5_tank_EMS_PLOT[index_cont] = tank.State5.value
# =========================================================================
## POST PROCESSING
# ==========================================================================
E_PV = []
E_HYDRO = []
E_grid = []
E_load_electric = []
E_load_thermal = []
mass_H2 = []
mass_H2O_theo = []
# =========================================================================
## ENERGY EVALUATION: calcolo di energia  rinnovabile
# ==========================================================================
E_PV = sum(P_pv_var_plot)
#E_HYDRO = sum(P_hydro[0:time_end-planning_horizon])
E_grid = sum(P_grid_plot[0:time_end-planning_horizon])
E_ele_furnace_plot = sum(P_ele_furnace_plot)
##########AAAAAA##########E_tank_out = sum(P_tank_out_plot)
E_load_th_FUR = sum(P_load_th_l[0:time_end-planning_horizon])
E_load_electric_PLANT = sum(P_load_ele_l[0:time_end-planning_horizon])
E_load_ele_FUR = sum(P_load_ele_FUR_l[0:time_end-planning_horizon])
E_load_thermal_TOT = sum(P_load_th_TOT_plot[0:time_end-planning_horizon])
E_load_th_FUR_week = E_load_th_FUR/2
E_load_electric_PLANT_week = E_load_electric_PLANT/2
E_load_ele_FUR_week = E_load_ele_FUR/2
E_load_thermal_TOT_week = E_load_thermal_TOT/2
print("LOAD thFUR, elePLANT, eleFUR, thTOT in 3 WEEKS")
print(E_load_th_FUR)
print(E_load_electric_PLANT)
print(E_load_ele_FUR)
print(E_load_thermal_TOT)
print()
print("per weeks")
print(E_load_th_FUR_week)
print(E_load_electric_PLANT_week)
print(E_load_ele_FUR_week)
print(E_load_thermal_TOT_week)
print()
# =========================================================================
## HYDROGEN PRODUCTION: calcolo di produzione di idrogeno
# ==========================================================================
mass_H2 = [P_ely_out_plot[ii]*(1/LHV_H2) for ii in range(time_end-planning_horizon)]  ##########AAAAAA##########                                                # [kgH2] = [kW]*[h]/[kWh/hg]
mass_H2_tot = sum(mass_H2)    ##########AAAAAA##########
mass_H2_week = mass_H2_tot/2  ##########AAAAAA##########
prod_ratio_H2_ely_out = sum(P_ely_out_plot)/(electro1.rated_power*electro1.efficiency*index_cont)  ##########AAAAAA##########
prod_ratio_H2_tank_out = sum(P_tank_out_plot)/(electro1.rated_power*electro1.efficiency*index_cont)   ##########AAAAAA##########                                                                    # [kg_h2]
prod_ratio_H2_ely_out_bur = sum(P_ely_out_bur_plot)/(electro1.rated_power*electro1.efficiency*index_cont)  ##########AAAAAA##########
print("RATIO H2 elyout, takn out e elyoutBUR")  ##########AAAAAA##########
print(prod_ratio_H2_ely_out)  ##########AAAAAA##########
print(prod_ratio_H2_tank_out)  ##########AAAAAA##########
print(prod_ratio_H2_ely_out_bur)  ##########AAAAAA##########
# =========================================================================
## WATER CONSUMPTION: calcolo di consumo d'acqua
# ==========================================================================
mass_H2O_theo = [mass_H2[ii]* (MM_H2O/MM_H2) for ii in range(time_end-planning_horizon)]                            # [kg_H2O]##########AAAAAA##########
mass_H2O_theo_tot = sum(mass_H2O_theo)                                                                        # [kg_H2O]  ##########AAAAAA##########
# =========================================================================
## OXYGEN PRODUCTION: calcolo di produzione di ossigeno
# ==========================================================================
mass_O2_theo_prod = [mass_H2[ii]*(1/2)*(MM_O2/MM_H2) for ii in range(time_end-planning_horizon)]                            # [kg_H2O]##########AAAAAA##########
mass_O2_theo_prod_tot = sum(mass_O2_theo_prod)                                                                        # [kg_H2O]  ##########AAAAAA##########
# =========================================================================
## ENERGY PER COMPONENTS: calcolo di consumo per componenti
# ==========================================================================
E_ely_out = sum(P_ely_out_plot)  ##########AAAAAA##########
E_ely_out_bur = sum(P_ely_out_bur_plot)  ##########AAAAAA##########
E_comp_in = sum(P_comp_in_plot)  ##########AAAAAA##########
E_tank_out = sum(P_tank_out_plot)  ##########AAAAAA##########
E_node_C = sum(P_node_C_plot)  ##########AAAAAA##########
E_NG = sum(P_NG_plot)
# =========================================================================
## EMISSIONS EVALUATION: calcolo di emissioni
# ==========================================================================
emission_grid = [P_grid_plot[ii]*1*environmental.ECI for ii in range(time_end-planning_horizon)] # [gCO2] = [kW]*[h]*[gCO2/kWh] 
emission_grid_total = sum(emission_grid)/1000   # [kg]
emission_NG_total = sum(CO2_NG_plot)            # [kg]
print("EMISSIONI TOTALI GRID e NG")
print(emission_grid_total)
print(emission_NG_total)
print()
# =========================================================================
## export: EXCELL FILE GENERATION
# ==========================================================================
#ee.xlwriter_function()
# =========================================================================
## plot: TOTAL COST
# ==========================================================================
plt.plot(tot_cost, label="total cost")
plt.title('TOTAL OPTIMIZED COST')
plt.grid()
plt.legend()
plt.show()
# =========================================================================
## plot: COMPARISON values of HYDROGEN and NATURAL GAS SUPPLY
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(P_ely_in_plot, label="power IN ely")
ax.plot(P_ely_out_plot, label="power OUT ely")
ax.plot(P_node_C_plot, label="power HYDROGEN OUTPUT")
plt.ylabel('power [kW]')
plt.legend()

#[0:time_end-planning_horizon]
ax2 = ax.twinx()
ax2.plot(P_NG_plot, label="P_NG", color="red")
plt.ylabel('power [kWh]')
plt.xlabel('hour [h]')         
plt.title('COMPARISON values of HYDROGEN and NATURAL GAS SUPPLY')
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: EMISSION COMPARISON
# ==========================================================================
plt.plot(cost_emission_NG_plot, label="emissions NG cost")
plt.plot(cost_emission_grid_plot, label="emissions GRID cost")
plt.xlabel('hour [h]')         #
plt.ylabel('CO2 emissions cost [euro]')
plt.title('COMPARISON values of EMISSIONS COSTS')
plt.legend()
plt.show()
# =========================================================================
## plot: COMPARISON WITH LOADS
# ==========================================================================
plt.plot(P_load_th_TOT_plot, label="power TOTAL FURNACE")
plt.plot(P_NG_plot, label="P_NG")
plt.plot(P_ely_out_plot, label="P_ely_out")
plt.plot(P_tank_out_plot, label="P_tank OUT")
plt.xlabel('hour [h]')         #
plt.ylabel('power [kW]')
plt.title('COMPARISON values of THERMAL LOAD and HYDROGEN PRODUCTION')
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: GRID PRICE + STATE + GAS PRICE
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(DeltaON_PLOT, label="Delta ON")
ax.plot(DeltaOFF_PLOT, label="Delta OFF")
ax.plot(DeltaSTB_PLOT, label="Delta STB")
plt.ylabel('state [-]')
plt.legend()

ax2 = ax.twinx()
ax2.plot(grid_price_purchase[0:time_end-planning_horizon], label="grid price purchase", color="red")
ax2.plot(NG_price[0:time_end-planning_horizon], label="NG price", color="black")
plt.ylabel('price [euro/kWh]')
plt.xlabel('hour [h]')         
plt.title('COMPARISON ELY OPERATION WITH GRID AND NG PRICES')
plt.legend()
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: STATE BEHAVIOUR FOR THE ELECTROLYSER
# ==========================================================================
plt.plot(DeltaON_PLOT, label="Delta ON")
plt.plot(DeltaOFF_PLOT, label="Delta OFF")
plt.plot(DeltaSTB_PLOT, label="Delta STB")
plt.xlabel('hour [h]')         #
plt.ylabel('state [-]')
plt.title('STATE BEHAVIOUR FOR THE ELECTROLYSER')
plt.legend()
plt.show()
# =========================================================================
## plot: LOAD COMPARISON
# ==========================================================================
plt.plot(P_load_th_l[0:time_end-planning_horizon], label="load THERMAL")
plt.plot(P_load_ele_l[0:time_end-planning_horizon], label="load electric PLANT")
plt.plot(P_load_ele_FUR_l[0:time_end-planning_horizon], label="load electric FURNACE")
plt.plot(P_load_th_TOT_plot, label="power TOTAL FURNACE")
#plt.plot(P_ely_out_plot, label="power OUT ely")
plt.xlabel('hour [h]')         #
plt.ylabel('power [kW]')
plt.title('COMPARISON values of ELECTRIC and THERMAL LOAD')
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: PRICES COMPARISON ANNUAL
# ==========================================================================
plt.plot(grid_price_purchase_l[0:time_end-planning_horizon], label="grid_price_purchase_l_ann")
plt.plot(NG_price_l[0:time_end-planning_horizon], label="NG_price_l_ann")
plt.plot(CARBON_PERMITS_price_l[0:time_end-planning_horizon], label="CARBON_PERMITS_price_l")
plt.xlabel('time [15min]')         #
plt.ylabel('power [€/kWh]')
plt.title('COMPARISON PRICES VALUES on ANNUAL RANGE')
plt.show()
# =========================================================================
## plot: TANK POWER 
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(P_tank_in_plot, label="P_tank IN")
ax.plot(P_tank_out_plot, label="P_tank OUT")
plt.ylabel('power [kW]')
plt.legend()
ax2 = ax.twinx()
ax2.plot(soc_tank_plot, label="SOC", color='red')
plt.ylabel('soc [kWh]')
plt.xlabel('hour [h]')         #
plt.title('TANK POWER')
plt.legend()
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: TANK BEHAVIOUR 
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(UHI_tank_PLOT, label="UHI_tank_PLOT")
plt.ylabel('UHI [m2/year]')
plt.legend()
ax2 = ax.twinx()
ax2.plot(soc_tank_plot, label="SOC", color='red')
plt.ylabel('soc [kWh]')
plt.xlabel('hour [h]')         #
plt.title('TANK POWER')
plt.legend()
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: TANK BEHAVIOUR 
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(UHI_tank_PLOT, label="UHI_tank_PLOT")
plt.ylabel('UHI [m2/year]')
plt.legend()
ax2 = ax.twinx()
ax2.plot(C_tank_UHI_PLOT, label="C_tank_UHI_PLOT", color='red')
plt.ylabel('cost UHI [€]')
plt.xlabel('hour [h]')         #
plt.title('TANK UHI analysis')
plt.legend()
plt.legend(loc=5)
plt.show()
"""
# =========================================================================
## plot: COMPARISON ANNUAL
# ==========================================================================
plt.plot(P_pv_l_ber, label="P_pv_l_ber")
plt.plot(P_pv_l[0:time_end-planning_horizon], label="P_pv_l")
plt.xlabel('time [1h - 15min]')         #
plt.ylabel('power [kW]')
plt.title('COMPARISON values on different scale')
plt.legend(loc=5)
plt.show()
# =========================================================================
## plot: ELY COMPARISONS
# ==========================================================================
#plt.plot(P_ely_in_plot, label="power IN ely")
#plt.plot(P_ely_out_plot, label="power OUT ely")
#plt.plot(P_comp_in_plot, label="P_comp_in")
#plt.plot(P_ely_out_bur_plot, label="P_ely_out_bur")
# plt.plot(P_tank_in_plot, label="P_tank_in")
# plt.plot(P_tank_out_plot, label="P_tank_out")
# plt.plot(P_bur_in_plot, label="P_bur_in")
#plt.xlabel('hour [h]')         #
#plt.ylabel('power [kW]')
#plt.title('COMPARISON values')
#plt.legend()
#plt.show()
# =========================================================================
## plot: TANK POWER 
# ==========================================================================
#fig, ax = plt.subplots()
#ax.plot(P_tank_in_plot, label="P_tank IN")
#ax.plot(P_tank_out_plot, label="P_tank OUT")
#plt.ylabel('power [kW]')
#plt.legend()
#ax2 = ax.twinx()
#ax2.plot(soc_tank_plot, label="SOC", color='red')
#plt.ylabel('soc [kWh]')
#plt.xlabel('hour [h]')         #
#plt.title('TANK POWER')
#plt.legend()
#plt.legend(loc=5)
#plt.show()
# =========================================================================
## plot: COST
# ==========================================================================
#####################plt.plot(ELY_cost_overall_plot, label="ELY_cost")
#plt.plot(C_NG_cost_plot, label="C_NG_cost")
#plt.plot(C_pv_cost_plot, label="C_PV_cost")
#plt.plot(C_grid_cost_plot, label="C_GRID_cost")
#plt.legend()
#plt.xlabel('hours [h]')         #
#plt.ylabel('costs [€]')
#plt.title('COSTS COMPARISONS')
#plt.grid()
#plt.show()
# =========================================================================
## plot: WATER COST
# ==========================================================================
#C_H2O_cost_plot_MICRO = [C_H2O_cost_plot[ii] for ii in range(np.size(C_H2O_cost_plot))]
plt.plot(C_H2O_cost_plot, label="cost_water")
plt.plot(C_O2_cost_plot, label="cost_O2")
plt.legend()
plt.xlabel('hours [h]')         #
plt.ylabel('costs [€]')
plt.title('WATER COST of the system')
plt.grid()
plt.show()
# =========================================================================
## plot: ETS EMISSION COSTS - CARBON_PERMITS
# ==========================================================================
#C_H2O_cost_plot_MICRO = [C_H2O_cost_plot[ii] for ii in range(np.size(C_H2O_cost_plot))]
plt.plot(CARBON_PERMITS_price_l, label="CARBON_PERMITS_price_l")
plt.legend()
plt.xlabel('hours [h]')         #
plt.ylabel('price [€/kgCO2]')
plt.title('CARBON PERMITS price')
plt.grid()
plt.show()
# =========================================================================
## plot: ELY OPERATION COMPARISON 
# ==========================================================================
##########AAAAAA##########fig, ax = plt.subplots()
##########AAAAAA##########ax.plot(DeltaON_PLOT, label="Delta ON")
##########AAAAAA##########ax.plot(DeltaOFF_PLOT, label="Delta OFF")
##########AAAAAA##########ax.plot(DeltaSTB_PLOT, label="Delta STB")
##########AAAAAA##########plt.ylabel('state [-]')
##########AAAAAA##########plt.legend()

##########AAAAAA##########ax2 = ax.twinx()
##########AAAAAA##########ax2.plot(P_ely_out_plot, label="power OUT ely", color='red')
##########AAAAAA##########plt.ylabel('power [kWh]')
##########AAAAAA##########plt.xlabel('hour [h]')         
##########AAAAAA##########plt.title('COMPARISON ELY OPERATION')
##########AAAAAA##########plt.legend()
##########AAAAAA##########plt.legend(loc=5)
##########AAAAAA##########plt.show()
# =========================================================================
## plot: NODE A COMPARISON
# ==========================================================================
##########AAAAAA##########plt.plot(P_ele_furnace_plot, label="P electricity to FURNACE")
##########AAAAAA##########plt.plot(P_pv_l, label="P PV")
##########AAAAAA##########plt.plot(P_ely_in_plot, label="power IN ely")
##########AAAAAA##########plt.plot(P_grid_plot, label="PURCHASE power")
##########AAAAAA##########plt.xlabel('hour [h]')         #
##########AAAAAA##########plt.ylabel('power [kW]')
##########AAAAAA##########plt.title('COMPARISON values at NODE A')
##########AAAAAA##########plt.legend()
##########AAAAAA##########plt.show()  
# =========================================================================
## plot: COSTS COMPARISON
# ==========================================================================
#plt.plot(C_pv_cost_plot, label="C_pv_cost plot")
#plt.xlabel('hour [h]')         #
#plt.ylabel('cost [€]')
#plt.title('COMPARISON values of COSTS')
#plt.legend()
#plt.show()
# =========================================================================
## plot: PRICES
# ==========================================================================
plt.plot(PV_cost, label="PV_cost price")
## plt.plot(HYDRO_cost, label="HYDRO_cost price")
plt.plot(grid_price_purchase, label="grid_price_purchase price")
plt.plot(NG_price_l, label="NG_cost_l")
plt.plot(price_water*np.ones(planning_horizon), label="price_water")
plt.xlabel('hours [h]')         #
plt.ylabel('price [€/kW]')
plt.title('PRICES')
plt.grid()
plt.legend()
plt.show()
# =========================================================================
## plot: POWER STANDBY
# ==========================================================================
#plt.plot(P_standby_plot, label="standby power")
#plt.xlabel('hour [h]')         #
#plt.ylabel('power [kW]')
#plt.title('STANDBY POWER')
#plt.legend()
#plt.show()
# =========================================================================
## plot: STATE BEHAVIOUR COSTS
# ==========================================================================
fig, ax = plt.subplots()
ax.plot(state_costs_plot, label="state_costs TOT")
ax.plot(state_costs_STB_plot, label="state_costs STB")
ax.plot(state_costs_ON_plot, label="state_costs ON")
plt.ylabel('costs [€]')
plt.legend()

ax2 = ax.twinx()
ax2.plot(transition_costs_plot, label="transition costs", linestyle='--', color='red', )
plt.ylabel('costs [€]')
plt.xlabel('hour [h]')         
plt.title('STATE BEHAVIOUR COSTS')
plt.legend()
plt.show()
# =========================================================================
## plot: INPUT PV DATA
# ==========================================================================
# plt.plot(grid_price_purchase/1000, label="P_grid")
# plt.title('TOTAL GRID PURCHASE COST')
# plt.grid()
# plt.legend()
# plt.show()
# =========================================================================
## plot: FORECAST COST
# ==========================================================================
# plt.plot(P_pv_cost, label="PV_cost_cost")
# plt.plot(P_hydro_cost, label="P_hydro_cost")
# plt.plot(C_grid_cost.value, label="C_grid_cost")
#plt.plot(C_ely_cost.value, label="C_ely_cost")
#plt.legend()
#plt.xlabel('hours [h]')         #
#plt.ylabel('costs [€]')
#plt.title('ELY FORECAST COSTS')
#plt.grid()
#plt.show()
# =========================================================================
## plot: TANK SOC
# ==========================================================================
# plt.plot(soc_tank_plot, label="soc_tank_out")
# plt.xlabel('hour [h]')         #
# plt.ylabel('soc [kWh]')
# plt.title('TANK soc')
# plt.legend()
# plt.show()
# =========================================================================
## plot: OVERALL COST BEFORE
# ==========================================================================
#plt.plot(tot_cost, label="tot_cost wrong")
#plt.legend()
#plt.xlabel('hours [h]')         #
##plt.ylabel('costs [€]')
#plt.title('TOTAL OVERALL COST of the system - WRONG CALCULATION')
#plt.grid()
#plt.show()
# =========================================================================
## plot: COSTS COMPARISON
# ==========================================================================
plt.plot( P_comp_grid_plot, label=" P_comp_grid plot")
plt.xlabel('hour [h]')         #
plt.ylabel('power [kW]')
plt.title('COMPRESOR POWER')
plt.legend()
plt.show()
# =========================================================================
## plot: GRID
# ==========================================================================
plt.plot(P_grid_plot, label="PURCHASE power")
#plt.plot(grid.sell_price, label="SELL price")
plt.xlabel('hour [h]')         #
plt.ylabel('power kW')
plt.title('GRID PURCHASE')
plt.legend()
plt.show()
# =========================================================================
## plot: NG COST 
# ==========================================================================
plt.plot(P_NG_plot, label="P_NG_plot")
plt.xlabel('hour [h]')         #
plt.ylabel('power [kWh]')
plt.title('POWER NATURAL GAS')
plt.legend()
plt.show()
# =========================================================================
## plot: GRID PRICE PURCHASE COMPARISON
# ==========================================================================
plt.plot(grid_price_purchase_ima, label="grid_price_purchase IMA")
plt.plot(grid_price_purchase_l, label="grid_price_purchase SCENARIOS")
plt.xlabel('hour [h]')         #
plt.ylabel('price [€/kW]')
plt.title('COMPARISON values of GRID IMA and SCENARIOS of GRID PRICE')
plt.legend()
plt.show()
# =========================================================================
## plot: GAS PRICE PURCHASE COMPARISON
# ==========================================================================
plt.plot(NG_price_ima, label="NG_price IMA")
plt.plot(NG_price_l, label="NG_price_l SCENARIOS")
plt.xlabel('hour [h]')         #
plt.ylabel('price [€/kW]')
plt.title('COMPARISON values of GRID IMA and SCENARIOS of GRID PRICE')
plt.legend()
plt.show()
# =========================================================================
## plot: LOAD COMPARISON
# ==========================================================================
plt.plot(P_load_th_ima, label="P_load_th SYNTHETIC")
plt.plot(P_load_th_l, label="P_load_th SCENARIOS")
plt.plot(P_load_ele_l, label="P_load_electric ")
plt.xlabel('hour [h]')         #
plt.ylabel('power [kW]')
plt.title('COMPARISON values of LOADS')
plt.legend()
plt.show()
"""
######################################################################
######################################################################
######################################################################
# EXCEL EXPORT 

DeltaON_PLOT_stamp=sum(DeltaON_PLOT)
DeltaOFF_PLOT_stamp=sum(DeltaOFF_PLOT)
DeltaSTB_PLOT_stamp=sum(DeltaSTB_PLOT)
state_costs_plot_stamp=sum(state_costs_plot)
state_costs_STB_plot_stamp=sum(state_costs_STB_plot)
state_costs_ON_plot_stamp=sum(state_costs_ON_plot)
transition_costs_plot_stamp=sum(transition_costs_plot)

# Creazione DataFrame
df1 = pa.DataFrame({
    "DeltaON_PLOT": DeltaON_PLOT,  ##########AAAAAA##########
    "DeltaOFF_PLOT": DeltaOFF_PLOT,  ##########AAAAAA##########
    "DeltaSTB_PLOT": DeltaSTB_PLOT,  ##########AAAAAA##########
    "SigmaONOFF_PLOT": SigmaONOFF_PLOT,  ##########AAAAAA##########
    "SigmaOFFON_PLOT": SigmaOFFON_PLOT,  ##########AAAAAA##########
    "SigmaONSTB_PLOT": SigmaONSTB_PLOT,  ##########AAAAAA##########
    "SigmaSTBON_PLOT": SigmaSTBON_PLOT,  ##########AAAAAA##########
    "SigmaSTBOFF_PLOT": SigmaSTBOFF_PLOT,  ##########AAAAAA##########
    "SigmaOFFSTB_PLOT": SigmaOFFSTB_PLOT,  ##########AAAAAA##########
    "P_grid_plot": P_grid_plot,
    "P_ely_in_plot": P_ely_in_plot,  ##########AAAAAA##########
    "P_ely_out_plot": P_ely_out_plot,  ##########AAAAAA##########
    "P_node_C_plot": P_node_C_plot,  ##########AAAAAA##########
    "mass_H2": mass_H2,  ##########AAAAAA##########
    "mass_H2O_theo": mass_H2O_theo,  ##########AAAAAA##########
    "mass_O2_theo_prod": mass_O2_theo_prod,  ##########AAAAAA##########
    "State1_ely_PLOT": State1_ely_PLOT,  ##########AAAAAA##########
    "State2_ely_PLOT": State2_ely_PLOT,  ##########AAAAAA##########
    "State3_ely_PLOT": State3_ely_PLOT,  ##########AAAAAA##########
    "State4_ely_PLOT": State4_ely_PLOT,  ##########AAAAAA##########
    "State5_ely_PLOT": State5_ely_PLOT,  ##########AAAAAA##########
    "UHI_ely_PLOT": UHI_ely_PLOT  ##########AAAAAA##########
})

df2 = pa.DataFrame({
    "prod_ratio_H2_ELY": [prod_ratio_H2_ely_out],  ##########AAAAAA##########
    "prod_ratio_H2_TANK": [prod_ratio_H2_tank_out],  ##########AAAAAA##########
    "prod_ratio_H2_BUR OUT": [prod_ratio_H2_ely_out_bur], ##########AAAAAA##########
    "E_ely_out": [E_ely_out],  ##########AAAAAA##########
    "E_ely_out_bur": [E_ely_out_bur], ##########AAAAAA##########
    "E_comp_in": [E_comp_in],  ##########AAAAAA########## 
    "E_tank_out": [E_tank_out],  ##########AAAAAA##########
    "E_node_C": [E_node_C],  ##########AAAAAA##########
    "mass_H2_tot": [mass_H2_tot],   ##########AAAAAA##########
    "mass_H2_week": [mass_H2_week],  ##########AAAAAA##########
    "mass_H2O_theo_tot": [mass_H2O_theo_tot],   ##########AAAAAA##########
    "emission_grid_total_kg": [emission_grid_total],
    "emission_NG_total_kg": [emission_NG_total],
    "E_NG": [E_NG],
    "E_PV": [E_PV],
    "E_ele_furnace_plot": [E_ele_furnace_plot] ##########AAAAAA##########
})

df3 = pa.DataFrame({
    "P_load_th_l": P_load_th_l[0:time_end-planning_horizon],
    "P_load_ele_l": P_load_ele_l[0:time_end-planning_horizon],
    "P_load_ele_FUR_l": P_load_ele_FUR_l[0:time_end-planning_horizon],
    "P_load_th_TOT_plot": P_load_th_TOT_plot[0:time_end-planning_horizon],
    "E_load_th_FUR": E_load_th_FUR,
    "E_load_electric_PLANT": E_load_electric_PLANT,
    "E_load_ele_FUR": E_load_ele_FUR,
    "E_load_thermal_TOT": E_load_thermal_TOT,
    "E_load_th_FUR_week": E_load_th_FUR_week,
    "E_load_electric_PLANT_week": E_load_electric_PLANT_week,
    "E_load_ele_FUR_week": E_load_ele_FUR_week,
    "E_load_thermal_TOT_week": E_load_thermal_TOT_week,
    "NG_price_l": NG_price_l[0:time_end-planning_horizon],
    "P_NG_plot": P_NG_plot,
    "grid_price_purchase_l": grid_price_purchase_l[0:time_end-planning_horizon]
})

df4 = pa.DataFrame({
    "tot_cost": tot_cost,
    "C_grid_cost_plot": C_grid_cost_plot,
    "C_pv_cost_plot": C_pv_cost_plot,
    "C_NG_cost_plot": C_NG_cost_plot,
    "cost_emission_NG_plot": cost_emission_NG_plot,
    "cost_emission_grid_plot": cost_emission_grid_plot,
    "state_costs_plot": state_costs_plot,  ##########AAAAAA##########
    "state_costs_STB_plot": state_costs_STB_plot, ##########AAAAAA##########
    "state_costs_ON_plot": state_costs_ON_plot,  ##########AAAAAA##########
    "cost_emission_grid_plot": cost_emission_grid_plot,
    "transition_costs_plot": transition_costs_plot,  ##########AAAAAA##########
    "C_comp_cost_plot": C_comp_cost_plot,  ##########AAAAAA##########
    "C_tank_cost_plot": C_tank_cost_plot,  ##########AAAAAA##########
    "C_H2O_cost_plot": C_H2O_cost_plot,  ##########AAAAAA##########
    "C_O2_cost_plot": C_O2_cost_plot,  ##########AAAAAA##########
})

df5 = pa.DataFrame({
    "P_tank_in_plot": P_tank_in_plot,  ##########AAAAAA##########
    "P_tank_out_plot": P_tank_out_plot,  ##########AAAAAA##########
    "soc_tank_plot": soc_tank_plot,  ##########AAAAAA##########
    "State1_tank_PLOT": State1_tank_PLOT,  ##########AAAAAA##########
    "State2_tank_PLOT": State2_tank_PLOT,  ##########AAAAAA##########
    "State3_tank_PLOT": State3_tank_PLOT,  ##########AAAAAA##########
    "State4_tank_PLOT": State4_tank_PLOT,  ##########AAAAAA##########
    "State5_tank_PLOT": State5_tank_PLOT,  ##########AAAAAA##########
    "UHI_tank_PLOT": UHI_tank_PLOT,  ##########AAAAAA##########
    "C_tank_UHI_PLOT": C_tank_UHI_PLOT  ##########AAAAAA##########
})

df6 = pa.DataFrame(SigmaONOFF_EMS_PLOT)  ##########AAAAAA##########
df7 = pa.DataFrame(SigmaOFFON_EMS_PLOT)  ##########AAAAAA##########
df8 = pa.DataFrame(SigmaONSTB_EMS_PLOT)  ##########AAAAAA##########
df9 = pa.DataFrame(SigmaSTBON_EMS_PLOT)  ##########AAAAAA##########
df10 = pa.DataFrame(SigmaSTBOFF_EMS_PLOT)  ##########AAAAAA##########
df11 = pa.DataFrame(SigmaOFFSTB_EMS_PLOT)  ##########AAAAAA##########

df12 = pa.DataFrame(state_costs_EMS_plot)  ##########AAAAAA##########
df13 = pa.DataFrame(state_costs_EMS_STB_plot)  ##########AAAAAA##########
df14 = pa.DataFrame(state_costs_EMS_ON_plot)  ##########AAAAAA##########
df15 = pa.DataFrame(transition_costs_EMS_plot)  ##########AAAAAA##########

df16 = pa.DataFrame(DeltaON_EMS_PLOT)  ##########AAAAAA##########
df17 = pa.DataFrame(DeltaOFF_EMS_PLOT)  ##########AAAAAA##########
df18 = pa.DataFrame(DeltaSTB_EMS_PLOT)  ##########AAAAAA##########

df19 = pa.DataFrame(State1_tank_EMS_PLOT)  ##########AAAAAA##########
df20 = pa.DataFrame(State2_tank_EMS_PLOT)  ##########AAAAAA##########
df21 = pa.DataFrame(State3_tank_EMS_PLOT)  ##########AAAAAA##########
df22 = pa.DataFrame(State4_tank_EMS_PLOT)  ##########AAAAAA##########
df23 = pa.DataFrame(State5_tank_EMS_PLOT)  ##########AAAAAA##########

#with pa.ExcelWriter('output_STEKLERNA_TOT_SCE0.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLERNA_TOT_SCE1.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLERNA_TOT_SCE2.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLERNA_TOT_SCE3.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLERNA_TOT_SCE2_5_valid2.xlsx') as writer:

#with pa.ExcelWriter('output_STEKLERNA_TOT_FINAL_SCE2_5.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLERNA_TOT_FINAL_SCE2_68.xlsx') as writer:  ##########AAAAAA##########
#with pa.ExcelWriter('output_STEKLERNA_TOT_FINAL_SCE0.xlsx') as writer:  ##########AAAAAA##########
#with pa.ExcelWriter('output_STEKLERNA_TOT_FINAL_SCE1.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_testingputput_FOREWARDconstraint.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_BERLIN_CARBON_PERMITS.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_CARBON_PERMITS_ANNUAL_12horizon_tank30.xlsx') as writer:    
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_BERLIN_CARBON_PERMITS_SAFETY_tank20.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_BERLIN_CARBON_PERMITS_SAFETY_tank16.xlsx') as writer:
#with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_BERLIN_CARBPERMITS_tank16_14days_UHI.xlsx') as writer:
with pa.ExcelWriter('output_STEKLARNA_TOT_FINAL_SCE2_05_FOREWARDconstraint_BERLIN_CARBPERMITS_tank16_14days_UHIcostOBJ.xlsx') as writer:
    df1.to_excel(writer, sheet_name='operation')
    df2.to_excel(writer, sheet_name='scenario', index=False)
    df3.to_excel(writer, sheet_name='input')
    df4.to_excel(writer, sheet_name='costs')
    df5.to_excel(writer, sheet_name='tank')  ##########AAAAAA##########
    df6.to_excel(writer, sheet_name='SigmaONOFF_EMS_PLOT')  
    df7.to_excel(writer, sheet_name='SigmaOFFON_EMS_PLOT') 
    df8.to_excel(writer, sheet_name='SigmaONSTB_EMS_PLOT') 
    df9.to_excel(writer, sheet_name='SigmaSTBON_EMS_PLOT') 
    df10.to_excel(writer, sheet_name='SigmaSTBOFF_EMS_PLOT') 
    df11.to_excel(writer, sheet_name='SigmaOFFSTB_EMS_PLOT') 
    df12.to_excel(writer, sheet_name='state_costs_EMS_plot') 
    df13.to_excel(writer, sheet_name='state_costs_EMS_STB_plot') 
    df14.to_excel(writer, sheet_name='state_costs_EMS_ON_plot') 
    df15.to_excel(writer, sheet_name='transition_costs_EMS_plot') 
    df16.to_excel(writer, sheet_name='DeltaON_EMS_PLOT') 
    df17.to_excel(writer, sheet_name='DeltaOFF_EMS_PLOT') 
    df18.to_excel(writer, sheet_name='DeltaSTB_EMS_PLOT') 
    df19.to_excel(writer, sheet_name='State1_tank_EMS_PLOT')
    df20.to_excel(writer, sheet_name='State2_tank_EMS_PLOT')
    df21.to_excel(writer, sheet_name='State3_tank_EMS_PLOT')
    df22.to_excel(writer, sheet_name='State4_tank_EMS_PLOT')
    df23.to_excel(writer, sheet_name='State5_tank_EMS_PLOT')

# Esportazione in Excel
#df.to_excel("output_STEKLERNA.xlsx", index=False)
#df.to_excel("output_STEKLERNA_0.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_0.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_10.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_100.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_alt.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_10perc.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_20perc.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_50perc.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_60perc.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_65perc.xlsx", index=False)
#df.to_excel("output_STEKLERNA_TOT_SCE3.xlsx", index=False)

print()
print()









