import os
import re
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
from scipy import signal,optimize,integrate,constants as const
import matplotlib.lines as lines
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys

def parseMCD(path,data_type,verbose=False,version='2021'):
    d={}
    for _, _, files in os.walk(path): #walk along all files in directory given a path
        for name in files: #for each file in list of files...
            if "test" not in str.lower(name): #remove any test files performed during data acqusition
                if version == '2021': #format example: "2-0T_2", is 2T, 3rd scan
                    name_parse = re.search('(.*)(?:-0T_)(.*)', name) #parse filename to get field and scan numbers
                    try:
                        field = name_parse.group(1) 
                    except AttributeError:
                        print("Possible input file formatting error?")
                    scan_number = name_parse.group(2)
                elif version == '2020': #format example: "0T_Mid_0" is 0T, 1st scan
                    name_parse = re.split('_', name)
                    try:
                        field = re.search('(.*)(?=T)', name_parse[0]).group(1)
                    except:
                        print("Possible input file formatting error?")
                    scan_number = name_parse[2]
                if verbose == True:
                    print("Adding", field + "T, Scan # " + scan_number + "...") #uncomment to check files are being added/named correctly
                df=pd.read_table(path+name, sep='\t',names=['wavelength','pemx','pemy','chpx','chpy','deltaA']) #create dataframe for each file
                df.insert(loc=1, column='energy', value=1240/df['wavelength']) #calculate energy from wavelength
                df['mdeg']=df['deltaA']*32982 # calculate mdeg from deltaA
                d[field + " " + scan_number + " " + data_type] = df #send dataframe to dictionary
    return d

def parseABS(path):
    d_abs = {}
    for _, _, files in os.walk(path): #walk along all files in directory given a path
        for name in files: #for each file in list of files...
            if "blank" in str.lower(name): #select for blank and ems file
                dic_name="Blank"
                # print("Adding", dic_name + "...") #uncomment to check files are being added/named correctly
                df=pd.read_table(path+name, sep='\t',names=['wavelength','pemx','pemy','chpx','chpy','deltaA']) #create dataframe for each file
                df['energy']=1240/df['wavelength'] #calculate energy from wavelength
                d_abs[dic_name] = df #send dataframe to dictionary
            if "ems" or "sample" in str.lower(name):
                dic_name="Ems"
                # print("Adding", dic_name + "...") #uncomment to check files are being added/named correctly
                df=pd.read_table(path+name, sep='\t',names=['wavelength','pemx','pemy','chpx','chpy','deltaA']) #create dataframe for each file
                df['energy']=1240/df['wavelength'] #calculate energy from wavelength
                d_abs[dic_name] = df #send dataframe to dictionary
    df_abs=pd.DataFrame(data=(d_abs['Ems']['wavelength'], d_abs['Ems']['energy'], d_abs['Ems']['chpx'], d_abs['Blank']['chpx'])).transpose() #make dataframe from ems/blank dictionary
    df_abs.columns=['wavelength','energy','ems','blank'] #setup columns
    df_abs=df_abs[(df_abs != 0).all(1)] #remove rows with 0's.
    df_abs['absorbance']=(2-np.log10(100 * df_abs['ems'] / df_abs['blank'])) #calculate absorbance from emission and blank data
    df_abs['smoothed_absorbance']=signal.savgol_filter(df_abs['absorbance'],43,2) #smooth absorbance plot using Savitzky-Golay
    df_abs = df_abs[df_abs.wavelength < 1700] #remove collection greater than 1700 nm (used for InGaAs errors mainly)
    return df_abs

def plotABS(df,material,max_ev,min_ev,op='smooth'):
    fig,ax=plt.subplots(figsize=(4,2))
    plt.xlabel('Energy (eV)')
    if op=='raw':
        plt.title("Raw Absorbance")
        sns.lineplot(data=df,x='energy',y='absorbance',color='Black')
    if op=='smooth':
        plt.title("Smoothed Absorbance")
        sns.lineplot(data=df,x='energy',y='smoothed_absorbance',color='Black')
    plt.ylabel('Absorbance (a.u.)')
    plt.xlim(max_ev,min_ev)
    plt.style.use('seaborn-paper')
    plt.savefig(material + '_abs',dpi=200,transparent=False,bbox_inches='tight')
    plt.show()

def calcAverageMCD(dic): #need to define this before finding the mcd difference
    MCDAveragesDic = {}
    for name, df in dic.items():
        name_parse = re.split('\s+',name)
        field, data_type = name_parse[0], name_parse[2]
        new_key = field + " " + data_type
        if new_key not in MCDAveragesDic:
            MCDAveragesDic[new_key] = pd.DataFrame() #if field is not in new dictionary, create an empty dataframe
        if new_key in MCDAveragesDic:
            df_concat=pd.concat([MCDAveragesDic[new_key], df]).groupby(['wavelength'], as_index=False) #concatenate field entry with new df entry, so long as field is matching
            MCDAveragesDic[new_key] = df_concat #update dictionary entry with newly concatenated one
        MCDAveragesDic[new_key] = MCDAveragesDic[new_key].mean() #take the average of the concatenated df
    return MCDAveragesDic

def calcDifference(signal, blank, spectra):
    new_dic = {}
    for name, signal_df in signal.items():
        field = re.split('\s+',name)[0]
        blank_df = blank[field + ' blank']
        if spectra == "MCD":
            dic_label = field + ' difference'
            new_dic[dic_label] = signal_df.copy()
            cols = signal_df.columns.difference(['wavelength','energy'])
            new_dic[dic_label][cols] = signal_df[cols] - blank_df[cols]
        elif spectra == "ABS":
            dic_label = field + ' absorption'
            new_dic[dic_label] = signal_df.filter(['wavelength','energy'], axis=1)
            new_dic[dic_label].insert(loc=len(new_dic[dic_label].columns), column='signal', value=signal_df['chpx']) 
            new_dic[dic_label].insert(loc=len(new_dic[dic_label].columns), column='blank', value=blank_df['chpx']) 
            new_dic[dic_label]['absorption'] = 2 - np.log10(100 * signal_df['chpx'] / blank_df['chpx'])
    return new_dic
    
def zeroSubtract(worked_up_data):
    for col in worked_up_data:
        try:
            field = re.split('_',col)[0]
        except:
            pass
        if 'deltaA' in col:
            worked_up_data[field + '_deltaA_diff'] = worked_up_data[col] - worked_up_data['0_deltaA']
        elif 'absorption' in col:
            worked_up_data[field + '_abs_diff'] = worked_up_data[col] - worked_up_data['0_absorption']
    return worked_up_data

def simulateMCDSpectra(data,max_ev,min_ev,ev=0.04,version='2021'): #function to visually show separation of LCP and RCP from base abs
    x = data['energy']
    if version == '2020':
        y = data['absorbance']
    else:
        y = data['2_absorption']
    coeff_L=poly.polyfit([x+ev for x in x],y,9) #LCP poly fit coeffs
    coeff_R=poly.polyfit([x-ev for x in x],y,9) #RCP poly fit coeffs
    fit_L=poly.polyval(x,coeff_L) #LCP line fit
    fit_R=poly.polyval(x,coeff_R) #RCP line fit

    # y_list = y.tolist()
    # y_max = np.max(y_list)
    # ev_peak_index = y_list.index(y_max)
    # ev_peak = x.iloc[ev_peak_index]
    # print(ev_peak)

    fit_diff=(fit_L-fit_R)/(np.max(x)) #calculate LCP-RCP normalized to absorbance max. Will incorporate based on field later. --Was originally multiplied by 1000 to match publication units.--
    # x = x.values.tolist()fit_R
    plt.figure(figsize=(6,6),dpi=80)

    ax1 = plt.subplot(2,1,1)
    plt.ylabel('Absorbance (a.u.)')
    plt.xlim(max_ev,min_ev)
    plt.ylim(0.9,1.5)
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    ax1.axes.yaxis.set_ticklabels([])
    plt.scatter(x,y,s=1.3,c='Black')
    plt.plot(x,fit_L,c='Red')
    plt.plot(x,fit_R,c='Blue')
    plt.plot([1.115,1.115],[0,10],c='Black',ls='--')
    # plt.plot(x,fit_diff,c='Purple')
    plt.legend(('Raw','LCP','RCP'))
    

    ax2 = plt.subplot(2,1,2)
    plt.ylabel(r'$\Delta$A')
    plt.xlabel('Energy (eV)')
    plt.xlim(max_ev,min_ev)
    plt.ylim(-0.065,0.04)
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    ax2.axes.yaxis.set_ticklabels([])
    plt.plot(x,fit_diff,c='Purple')
    plt.plot([0,10],[0,0],c='Black',ls='--')
    plt.plot([1.115,1.115],[-1,1],c='Black',ls='--')
    plt.savefig('Simulated MCD.png',dpi=200,transparent=False,bbox_inches='tight')
    plt.show()

    return fit_diff

def calc_effective_mass_and_plot(mcd_data,abs_data,max_ev,min_ev,material,correction_factor=1,version='2021'):
    ev_list=[]
    std_dev_fit_list=[]
    m_list=[]
    B_list=[]
    for col in mcd_data:
        if 'deltaA_diff' in col:
            field = col.split("_")[0]
            if field is not '0':
                xdata=mcd_data.loc[mcd_data['energy'].between(min_ev, max_ev, inclusive='both'),'energy'].values
                ydata=mcd_data.loc[mcd_data['energy'].between(min_ev, max_ev, inclusive='both'),col].values
                B_fit=int(field) #used for meV plotting later in zdf
                B=np.absolute(B_fit) #magnetic field (T)
                B_list.append(B_fit)
                if version == '2020':
                    ydata=ydata/(np.max(abs_data['absorbance'])*correction_factor) #Uncomment to normalize by abs max
                    def func(x,ev,y): #define simulated mcd function from absorbance spectrum
                        coeffL=poly.polyfit(abs_data['energy']+ev,abs_data['smoothed_absorbance'],9) #find polynomial coeffs from original absorption spectra
                        coeffR=poly.polyfit(abs_data['energy']-ev,abs_data['smoothed_absorbance'],9) #find polynomial coeffs from original absorption spectra
                        LCP=poly.polyval(x,coeffL) #find y from +ev shifted LCP spectrum
                        RCP=poly.polyval(x,coeffR) #find y from -ev shifted RCP spectrum

                        # return LCP-RCP #return y from LCP-RCP, Note: may need to flip this depending on spectrum
                        return LCP-RCP-y #switch to this if doing y adjustment
                else:
                    ydata=ydata/(np.max(abs_data[field + '_absorption'])*correction_factor) #Uncomment to normalize by abs max
                    def func(x,ev,y): #define simulated mcd function from absorbance spectrum
                        coeffL=poly.polyfit(abs_data['energy']+ev,abs_data[field + '_absorption'],9) #find polynomial coeffs from original absorption spectra
                        coeffR=poly.polyfit(abs_data['energy']-ev,abs_data[field + '_absorption'],9) #find polynomial coeffs from original absorption spectra
                        LCP=poly.polyval(x,coeffL) #find y from +ev shifted LCP spectrum
                        RCP=poly.polyval(x,coeffR) #find y from -ev shifted RCP spectrum

                        # return LCP-RCP #return y from LCP-RCP, Note: may need to flip this depending on spectrum
                        return RCP-LCP-y #switch to this if doing y adjustment

                # ydata_normalized=np.nan_to_num(ydata_normalized, nan=0.0)
                # if B_fit < 0:
                #     popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=0.00001,method='trf',bounds=(0.000005,0.001)) #lsf optimization to spit out zeeman split mev, guess is 10^-3 eV
                # if B_fit > 0:
                #     popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=-0.00001,method='trf',bounds=(-0.001,-0.000005)) #lsf optimization to spit out zeeman split mev.

                if B_fit < 0:
                    popt,pcov = optimize.curve_fit(func,xdata,ydata,p0=(0.0003,0.0001),method='trf',bounds=([0.00001,-0.01],[0.00070,0.01])) #multiple variables: bounds are: ([lower bounds],[upper bounds])
                elif B_fit > 0:
                    popt,pcov = optimize.curve_fit(func,xdata,ydata,p0=(-0.0003,-0.0001),method='trf',bounds=([-0.00070,-0.01],[-0.00001,0.01])) #lsf optimization to spit out zeeman split mev, guessing ~0.05 meV

                # print(popt)
                # print(pcov) #list of residuals
                ev=popt[0] #return minimzed ev to variable
                ev_list.append(ev*1000) #add ev to list as meV
                std_dev_of_fit=(np.sqrt(np.diag(pcov))*1000)[0] #return std dev of fitting
                std_dev_fit_list.append(std_dev_of_fit) #add std dev fit
                # c=const.c #speed of light (m/s)
                e=const.e #charge of electron (C)
                m_e=const.m_e #mass of electron (kg)
                w_c=ev/const.physical_constants['Planck constant over 2 pi in eV s'][0] #cyclotron resonance frequency from Planck constant in eV/Hz
                effective_mass=e*B/w_c/2/m_e/const.pi #effective mass (m*/m_e)
                m_list.append(np.absolute(effective_mass)) #add m* to list

                fig,ax=plt.subplots(figsize=(2,4))
                plt.title(str(field) + 'T Fit')
                plt.ylabel(r'MCD ($\Delta$A/A$_{max}$)')
                plt.xlabel('Energy (eV)')
                plt.xlim(max_ev+0.1,min_ev-0.1)
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.xaxis.set_major_locator(MultipleLocator(0.2))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                plt.plot(xdata,ydata,label='experiment_data',c='Black')
                plt.plot(xdata,[func(x,*popt) for x in xdata],label='simulated_fit',c='Red')
                baseline = lines.Line2D(range(10),np.zeros(1),c='black',ls='--',lw=0.6) #draw baseline at 0T
                ax.add_line(baseline) #add baseline to plot
                # plt.legend(loc=0)
                # plt.text(3,0,'%.3f meV\n%.3f m*' % (ev*1000,effective_mass),fontweight='bold',bbox={'facecolor':'white','alpha':0.5,'pad':0.1}) #places text according to graph x,y coords
                plt.savefig(material + '_' + str(field) + "T_fit",dpi=100,transparent=False,bbox_inches='tight')
    average_ev = np.mean(ev_list)
    std_dev_ev = np.std(ev_list)
    average_m = np.mean(m_list)
    std_dev_m = np.std(m_list)
    zdf = pd.DataFrame(list(zip(B_list,ev_list,std_dev_fit_list,m_list)),columns=['B','E_Z','E_Z_std_dev','m*'])
    return average_ev, std_dev_ev, average_m, std_dev_m, zdf

def findPeakMax(data,max_ev,min_ev,peak_number,spectra):
    columns = ['Field',spectra + '_' + peak_number + '_amp_max',spectra + '_' + peak_number + '_amp_min']
    amplitudePair = []
    for col in data:
        if 'deltaA_diff' in col or 'absorption' in col or 'absorbance' in col:
            field = col.split("_")[0]
            if field is not '0':
                xdata = data.loc[data['energy'].between(min_ev, max_ev, inclusive='both'),'energy'].values
                ydata = data.loc[data['energy'].between(min_ev, max_ev, inclusive='both'),col].values
                try:
                    if np.mode(ydata) > 0:
                        ydataAmplitude = np.max(ydata)
                    elif np.mode(ydata) < 0 :
                        ydataAmplitude = np.min(ydata)
                except ValueError:
                    print('Error! Zero-size array possible. Perhaps peak limits are set incorrectly?')
                amplitudePair.append([int(field),ydataAmplitude])
    amplitudeData = pd.DataFrame(amplitudePair, columns=columns)
    return amplitudeData

def findAllPeaks(data,ev_list,spectra):
    allAmplitudeData = pd.DataFrame()
    for peak_num, ev_list in enumerate(ev_list):
        max_ev = ev_list[0]
        min_ev = ev_list[1]
        peak_num = str(peak_num + 1)
        amplitudeData = findPeakMax(data, max_ev, min_ev, peak_num, spectra=spectra)
        allAmplitudeData = pd.concat([allAmplitudeData, amplitudeData], axis=1).T.drop_duplicates().T
    return allAmplitudeData

def convertToCSV(data, spectra):
    csv_df = pd.DataFrame()
    for name, df in data.items():
        # print(name, df)
        field = int(re.split('\s+',name)[0])
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(how='any', inplace=True)
        if spectra == 'MCD':
            df.drop(['chpx','chpy','pemx','pemy', 'energy'], axis=1, inplace=True)
            rename_list = {'deltaA':'{}_deltaA'.format(field),'mdeg':'{}_mdeg'.format(field)}
            df.rename(columns=rename_list,inplace=True)
        elif spectra == 'ABS':
            df.drop(['signal','blank','energy'], axis=1, inplace=True) 
            rename_list = {'absorption':'{}_absorption'.format(field)}
            df.rename(columns=rename_list,inplace=True)
        try:
            csv_df = csv_df.merge(df, how='inner', on='wavelength')
        except KeyError:
            csv_df = df
    csv_df = csv_df.reindex(sorted(list(csv_df), key=lambda x: x.split('_')[-1]), axis=1)
    try:
        csv_df.insert(0,'energy', [1240/x for x in csv_df['wavelength']])
    except KeyError:
        print("Did you type in the correct folder?")
    csv_df.set_index('wavelength',inplace=True)
    return csv_df

if __name__ == '__main__':
    print("This is a function file! Hah, gottem'")