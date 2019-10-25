# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:25:31 2019

@author: Philip
"""
import pandas as pd
import numpy as np
import julian
import calendar
import os 

def HamonETo_mm(wthdata, lat = None, dz_m = 0):
    wthdata = wthdata.copy()
    def cal_daylight_hours(lat):
        J = [julian.to_jd(item, fmt='jd') for item in wthdata["Date"] ]
        si = [0.409*np.sin(0.0172*j-1.39) for j in J]
        phi = lat*np.pi/180
        w = [np.arccos(-np.tan(s)*np.tan(phi)) for s in si]
        wthdata["Daylight_Hr"] = [24/np.pi*wi for wi in w]
        print("Doned daylight hours at lat="+str(lat)+" !")
    cal_daylight_hours(lat)
    tlaps = 0.6 # Assume TX01 decrease 0.6 degree when dz_m = 100 m
    temp = wthdata["TX01"] - tlaps*dz_m/100
    daylight_hour = wthdata["Daylight_Hr"] 
    H = np.array(daylight_hour)
    T = np.array(temp)
    es = 0.6108*np.exp(17.27*T/(T+237.3))
    wthdata["ETo"] = list(29.8*H*es/(T+273.2)) # mm
    print("Done ETo(mm)!")
    return wthdata
                
def GWLF(Data,Par, lat = 23.5, dz_m = 0):
    # Input time series data
    PP01 = Data.loc[:,"PP01"]/10                # (mm) -> (cm)
    # 5 day in advance acc rainfall amount (cm)
    AccP5 = [0,sum(PP01[0:1]),sum(PP01[0:2]),sum(PP01[0:3]),sum(PP01[0:4])]+[sum(PP01[i:i+5]) for i in range(len(PP01)-5)]
    tm = Data.loc[:,"TX01"]
    tm.index = Data["Date"]
    tm = tm.resample("M").mean()
    Tm = [] # Monthly avg temperature
    for i in range(len(tm)):
        t = [tm[i]]*calendar.monthrange(tm.index.year[i],tm.index.month[i])[1]
        Tm = Tm + t
    if "ETo" not in list(Data):
        Data = HamonETo_mm(Data, lat, dz_m)     # (mm)
    ETo = Data["ETo"]/10                        # (cm)
    # Calculated data
    Runoff = []                                 # (cm)
    Infil = []                                  # (cm)
    GW = []                                     # (cm)
    SWC_sat = []                                # (cm)
    SWC_unsat = []                              # (cm)
    Discharge = []                              # (cms)
    Ks = []                                     # water stress
    ET = []                                     # (cm)
    Percol = []                                 # (cm)
    
    # Initialization
    SWC_sat.append(Par.loc[0,"ini_SWC_sat"])
    SWC_unsat.append(Par.loc[0,"ini_SWC_unsat"])
    CN2 = Par.loc[0,"CN"]
    CN1 = 4.2*CN2/(10-0.058*CN2)
    CN3 = 23*CN2/(10+0.13*CN2)
    def CalCN(AM,Tmavg):
        if Tmavg > 10:
            AM1 = 3.6 # (cm)
            AM2 = 5.3 # (cm)
        else:
            AM1 = 1.3 # (cm)
            AM2 = 2.8 # (cm)
        if AM > AM2:
            CN = CN3
        elif AM > AM1 and AM <=AM2:
            CN = CN2+(CN3-CN2)*((AM-AM1)/(AM2-AM1))
        elif AM <= AM1:
            CN = CN1+(CN2-CN1)*((AM-0)/(AM1-0))
        return CN 

    for i in range(0,len(Data["PP01"])):
        # Calculate CN
        CN = CalCN(AccP5[i],Tm[i])
        w = 2540/CN-25.4
        
        if((PP01[i]) < (0.2*w)):
            Runoff.append(0)
        else:
            # Convert rainfall unit from mm to cm
            Runoff.append((PP01[i]-0.2*w)**2/(PP01[i]+0.8*w)) 
        
        Infil.append(PP01[i] - Runoff[i])
        GW.append( Par.loc[0,"r"]*SWC_sat[i])
        Discharge.append( Runoff[i] + GW[i] )
        
        if(SWC_unsat[i] >= Par.loc[0,"AWC"]*0.5):
            Ks.append(1)
        else:
            Ks.append( SWC_unsat[i]/(Par.loc[0,"AWC"]*0.5))
            
        ET.append( np.minimum(Ks[i]*Par.loc[0,"Kc"]*ETo[i],SWC_unsat[i]+Infil[i]))
        Percol.append( np.maximum(0, SWC_unsat[i]+Infil[i]-ET[i]-Par.loc[0,"AWC"]))
        SWC_unsat.append(SWC_unsat[i] + Infil[i] - ET[i] - Percol[i])
        SWC_sat.append(SWC_sat[i] + Percol[i] - GW[i])
    
    #Discharge cm -> CMS
    Discharge = list(np.array(Discharge)*Par.loc[0,"Area"]*10000/86400/100)
    
    Output = {"Date":Data["Date"],
                "Discharge":Discharge}
    Output_D = pd.DataFrame(Output,columns = Output.keys())
    Output_D = Output_D.set_index('Date')

    # If want to output other variables.
    Calculated_data = {"Ks":Ks,
                        "ET":ET,
                        "Runoff":Runoff,
                        "Infil":Infil,
                        "GW":GW, 
                        "SWC_sat": SWC_sat,
                        "SWC_unsat": SWC_unsat}  
    Calculated_data = pd.DataFrame.from_dict(Calculated_data, orient='index')
    Calculated_data = Calculated_data.transpose()
    #Calculated_data = pd.DataFrame(Calculated_data,columns=Calculated_data.keys())
    print("GWLF done!")
    return Output_D, Calculated_data

def Performance(Sim,Obv,RemoveNA = True):
    if RemoveNA:
        Sim = list(Sim); Obv = list(Obv)
        NaIndex = list(np.argwhere(np.isnan(Obv)).reshape((-1,)))
        Sim = [Sim[i] for i in range(len(Sim)) if i not in NaIndex]
        Obv = [Obv[i] for i in range(len(Obv)) if i not in NaIndex]
    Obv = np.array(Obv)
    Sim = np.array(Sim)
    rms = (np.nanmean((Obv-Sim)**2))**0.5   #mean_squared_error(Obv, Sim)**0.5
    r = np.nansum((Obv-np.nanmean(Obv))*((Sim-np.nanmean(Sim)))) / ( ((np.nansum((Obv-np.nanmean(Obv))**2))**0.5)*((np.nansum((Sim-np.nanmean(Sim))**2))**0.5))
            #r2_score(Sim,Obv)
    CE = 1 - np.nansum((Obv-Sim)**2)/np.nansum((Obv-np.nanmean(Obv))**2) # Nash
    CP = 1 - np.nansum((Obv[1:]-Sim[1:])**2)/np.nansum((Obv[1:]-Obv[:-1])**2)
    data = {"RMSE": rms,
            "r": r,
            "CE": CE,
            "CP": CP}
    performance = pd.DataFrame(data,columns = data.keys(),index = [0])
    print(performance)
    return performance

def D2tenday(df, DateinIndex = False):
    df = df.copy()
    if "stno" in list(df):
        df = df.drop("stno",axis = 1)
    if DateinIndex is False:
        df = df.set_index("Date") 
    L = len(df.resample("M").mean())
    y=df.index[0].year
    m=df.index[0].month
    rng = pd.date_range(pd.datetime(y,m,5),pd.datetime(df.index[-1].year,df.index[-1].month,25),freq = "D")
    rng1 = rng[rng.day == 5];rng2 = rng[rng.day == 15];rng3 = rng[rng.day == 25]
    Tenday = []; Date = []
    for i in range(L):
        df1 = df[df.index.year == y+int((m-1)/12)]
        df2 = df1[df1.index.month == (m-1)%12+1]
        df10_1 = np.nanmean(df2[df2.index.day <= 10],axis=0)
        df10_2 = df2[df2.index.day > 10];df10_2 = np.nanmean(df10_2[df10_2.index.day <= 20],axis=0)
        df10_3 = np.nanmean(df2[df2.index.day > 20],axis=0)
        Tenday.append(df10_1);Tenday.append(df10_2);Tenday.append(df10_3)
        Date.append(rng1[i]);Date.append(rng2[i]);Date.append(rng3[i])
        m += 1
    d_ini = df.index[0].day; d_last = df.index[-1].day
    h = int(d_ini/10); t = int(d_last/10) - 2; 
    if t>=0: t=len(Date)
    Tenday = Tenday[h:t]; Date = Date[h:t]
    Tenday = np.array(Tenday)
    df_tenday = pd.DataFrame(Tenday,columns = list(df))
    df_tenday.index = Date   
    return df_tenday
#%%
if __name__ == '__main__':
    print("Please enter the path of working folder.")
    path = os.path.normpath(input())
    print("\nInput time series csv file name:\nColumns should include \"Date\",	\"PP01\", \"TX01\", (\"ETo\", \"Discharge\") in units [mm], [C], ([mm], [cms])")
    Tfilename = os.path.join(path, os.path.normpath(input()))
    Input_t = pd.read_csv(Tfilename,parse_dates=["Date"],engine = "python")
    print(Input_t.head(5))
    print("\nInput parameter csv file name:\nMake sure to follow the template format.")
    Pfilename = os.path.join(path, os.path.normpath(input()))
    Par = pd.read_csv(Pfilename,skiprows = 2)
    print(Par.head(4))
    if "ETo" not in list(Input_t):
        print("\nPlease enter the latitude of the location.")
        latitude = float(input())
        print("\nPlease enter the elevation correction [m] of the temperature or enter 0.")
        dz = float(input())
    else:
        latitude = 23.5
        dz = 0
    print("\nSimulation start!\n")
    Output_D, Cal_D = GWLF(Input_t,Par,lat = latitude, dz_m = dz)
    Output_M = Output_D.resample("M").mean()
    Output_10 = D2tenday(Output_D, DateinIndex = True)
    if "Discharge" in list(Input_t):
        print("\n=========================================================\n\nCalibration result:")
        Obv_M = Input_t.set_index('Date').resample("M").mean().loc[:,"Discharge"]
        Obv_10 = D2tenday(Input_t, DateinIndex = False).loc[:,"Discharge"]
        print("\nDaily:\n")
        PerformanceD = Performance(Output_D.loc[:,"Discharge"],Input_t.loc[:,"Discharge"])
        print("\nTenday:\n")
        PerformanceM = Performance(Output_M.loc[:,"Discharge"],Obv_M)
        print("\nMonthly:\n")
        Performance10 = Performance(Output_10.loc[:,"Discharge"],Obv_10)     
    else: 
        print("\nCalbration result can be shown if \"Discharge\" is provided.")
    print("=========================================================")
    print("Output the simulation result? [y/n]")
    if input() == "y":
        Output_D.to_csv(os.path.join(path, "OUTPUT_D.csv"))
        Output_10.to_csv(os.path.join(path, "OUTPUT_10.csv"))
        Output_M.to_csv(os.path.join(path, "OUTPUT_M.csv"))
    print("Done.")
    input()