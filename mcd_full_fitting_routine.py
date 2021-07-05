import os
import re
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as lines
import seaborn as sns
from scipy import signal,optimize,constants as const
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams.update({'figure.max_open_warning': 0}) #Remove figure creation RuntimeWarning.

def getWavelength(eV):
    return 1240/eV

def getEnergy(nm):
    return 1240/nm

def parse_mcd(path):
    d={}
    for root, dirs, files in os.walk(path): #walk along all files in directory given a path
        for num, name in enumerate(files): #for each file in list of files...
            if "test" not in str.lower(name): #remove any test files performed during data acqusition
                field_name = re.search('(.*)(?=T_)..',name).group(0) #search for beginning of file name "#T_" and set as name for df. Man this was difficult to figure out syntactically. 
                f=field_name + str(num%3) #differentiate repeated scans at same field
                # print("Adding", f + "...") #uncomment to check files are being added/named correctly
                df=pd.read_table(path+name, sep='\t',names=['wavelength','pemx','pemy','chpx','chpy','deltaA']) #create dataframe for each file
                df['field']=int(re.search('(.*)(?=T)',name).group(0)) #add column for field
                df['energy']=1240/df['wavelength'] #calculate energy from wavelength
                df['mdeg']=df['deltaA']*32982 # calculate mdeg from deltaA
                d[f] = df #send dataframe to dictionary
    return d

def parse_mcd_tests(path):
    d={}
    for root, dirs, files in os.walk(path): #walk along all files in directory given a path
        for num, name in enumerate(files): #for each file in list of files...
            if "test" in str.lower(name): #remove any test files performed during data acqusition
                field_name = re.search('(.*)(?=T_)..',name).group(0) #search for beginning of file name "#T_" and set as name for df. Man this was difficult to figure out syntactically. 
                f=field_name + str(num%3) #differentiate repeated scans at same field
                # print("Adding", f + "...") #uncomment to check files are being added/named correctly
                df=pd.read_table(path+name, sep='\t',names=['wavelength','pemx','pemy','chpx','chpy','deltaA']) #create dataframe for each file
                df['field']=int(re.search('(.*)(?=T)',name).group(0)) #add column for field
                df['energy']=1240/df['wavelength'] #calculate energy from wavelength
                df['mdeg']=df['deltaA']*32982 # calculate mdeg from deltaA
                d[f] = df #send dataframe to dictionary
    return d

def plot_mcd(dic,op='avg',x_axis='Energy (eV)',title='[PH]',xdata='energy',ydata='mdeg'):
    plt.clf()
    fig,ax=plt.subplots(figsize=(4,2))
    # norm=plt.Normalize(-10,10) #optional to remove discrete H bar divisions
    norm=colors.BoundaryNorm(np.linspace(-10,10,11),ncolors=256)
    sm=plt.cm.ScalarMappable(cmap='coolwarm_r',norm=norm) 
    fig.colorbar(sm,ticks=range(-10,11,2),label='H (T)') #make color bar based on H (T) for plot
    for df in dic.values():
        #Dr. Seaborn or: How I Learned to Stop Worrying and Love sns.lineplot. Such efficiency. Much wow.
        sns.lineplot(data=df,x=xdata,y=ydata, linewidth=0.6,
                    hue='field',hue_norm=(-10,10),
                    palette=sns.color_palette('coolwarm_r',as_cmap=True),
                    legend=None)
    if x_axis=='Energy (eV)':
        ax.set_xlabel(x_axis)
    if op=='raw':
        plt.title("Raw MCD " + title)
    if op=='avg':
        plt.title("Averaged MCD " + title)
    
    ax.plot([-10,10],[0,0],color='black',linestyle='-',linewidth='1') #add 0T baseline

    ### Change the axis settings as needed for data ###
    ax.set_ylabel('MCD (mdeg)')
    ax.set_xlim(2.7,1.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator()) # auto set minor ticks
    ax.set_ylim(-1.5,1.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator()) # auto set minor ticks
    
    ax2 = ax.twiny() # creates a new axis with invisible y and independent x on opposite side of first x-axis
    ax2.set_xlabel(r'Wavelength (nm)')
    ax2.set_xscale('function',functions=(getWavelength,getEnergy)) # set twin scale (convert degree eV to nm)
    xmin, xmax = ax.get_xlim() # get left axis limits
    ax2.set_xlim((getWavelength(xmax),getWavelength(xmin))) # apply function and set transformed values to right axis limits
    ax2.xaxis.set_minor_locator(AutoMinorLocator()) # auto set minor ticks
    
    ax2.plot([],[]) # set an invisible artist to twin axes to prevent falling back to initial values on rescale events

    #Set tick parameters    
    ax.xaxis.set_tick_params(which='major', size=5, width=0.8, direction='in') #axes.linewidth by default for matplotlib is 0.8, so that value is used here for aesthetic matching.
    ax.xaxis.set_tick_params(which='minor', size=2, width=0.8, direction='in')
    ax.yaxis.set_tick_params(which='major', size=5, width=0.8, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=2, width=0.8, direction='in', right='on')

    ax2.xaxis.set_tick_params(which='major', size=5, width=0.8, direction='in')
    ax2.xaxis.set_tick_params(which='minor', size=2, width=0.8, direction='in')

    plt.tight_layout()

    plt.style.use('seaborn-paper')
    plt.savefig(op + '_mcd_' + title,dpi=300,transparent=False,bbox_inches='tight')
    plt.show()

def calc_raw_avg_mcd(dic): #need to define this before finding the mcd difference
    df_avgs={}
    for name, df in dic.items():
        field = re.search('(.*)(?=T)',name).group(0) #set variable 'field' equal to int(field) from original dictionary
        if field not in df_avgs: 
            df_avgs[field] = pd.DataFrame() #if field is not in new dictionary, create an empty dataframe
        if field in df_avgs:
            df_concat=pd.concat([df_avgs[field], df]).groupby(['wavelength'], as_index=False) #concatenate field entry with new df entry, so long as field is matching
            df_avgs[field] = df_concat #update dictionary entry with newly concatenated one
        df_avgs[field]=df_avgs[field].mean() #take the average of the concatenated df
    return df_avgs

def calc_diff_mcd(dic,op='sub'):
    df_diff={}
    for name, df in dic.items():
        df_diff[name] = pd.DataFrame()
        if name == '0': 
            del df_diff[name] #placeholder for now. In future would like to plot 0T field difference, but previous function looks like it deletes a version of 0 so no pair to subtract.
            pass
        elif '-' not in name: #loop only positive half of dictionary
            if op=='add':
                df_diff[name] = df + dic['-' + name] #add positive and negative dictionary entries
                df_diff[name]['energy'] = df['energy'] #fix energy back to original values
                df_diff[name]['field'] = df['field'] #fix field back to original values
            elif op=='sub':
                df_diff[name] = df - dic['-' + name] #subtract positive and negative dictionary entries
                df_diff[name]['energy'] = df['energy'] #fix field back to original values
                df_diff[name]['field'] = df['field']
        else:
            del df_diff[name]
            continue
    return df_diff

def mcd_blank_subtraction(dic_mcd,dic_blank):
    df_blank_subtracted={}
    for name, df in dic_mcd.items():
        df_blank_subtracted[name] = df - dic_blank[name]
        df_blank_subtracted[name]['energy'] = df['energy'] #fix field back to original values
        df_blank_subtracted[name]['field'] = df['field']

def plot_diff_mcd(dic,op='avg',x_axis='Energy (eV)'):
    fig,ax=plt.subplots(figsize=(4,2))
    # norm=plt.Normalize(-10,10) #optional to remove discrete H bar divisions
    norm=colors.BoundaryNorm(np.linspace(0,10,6),ncolors=256)
    sm=plt.cm.ScalarMappable(cmap='Greys',norm=norm) 
    fig.colorbar(sm,ticks=range(0,11,2),label='H (T)') #make color bar based on H (T) for plot
    for df in dic.values():
        #Dr. Seaborn or: How I Learned to Stop Worrying and Love sns.lineplot. Such efficiency. Much wow.
        sns.lineplot(data=df,x='energy',y='mdeg', linewidth=0.6,
                    hue='field',hue_norm=(0,10),
                    palette=sns.color_palette('Greys',as_cmap=True),
                    legend=None)
    if x_axis=='Energy (eV)':
        plt.xlabel(x_axis)
    if op=='raw':
        plt.title("Raw MCD")
    if op=='avg':
        plt.title("Difference MCD")
    plt.ylabel('MCD (mdeg)')
    plt.xlim(3.2,.55)
    baseline = lines.Line2D(range(6),np.zeros(1),c='black',ls='--',lw=0.6) #draw baseline at 0T
    ax.add_line(baseline) #add baseline to plot
    plt.style.use('seaborn-paper')
    plt.savefig('diff_mcd',dpi=100,transparent=True,bbox_inches='tight')
    plt.show()

def parse_abs(path):
    d_abs = {}
    for root, dirs, files in os.walk(path): #walk along all files in directory given a path
        for num, name in enumerate(files): #for each file in list of files...
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

def parse_lambda_950_abs(path):
    d_abs = {}
    df_abs=pd.read_csv(path, header=0, names=['wavelength','absorbance'])
    df_abs['energy']=1240/df_abs['wavelength'] #calculate energy from wavelength
    df_abs['smoothed_absorbance']=signal.savgol_filter(df_abs['absorbance'],59,3) #smooth absorbance plot using Savitzky-Golay
    return df_abs

def plot_abs(df,op='smooth',x_axis='energy'):
    fig,ax=plt.subplots(figsize=(4,2))
    if x_axis=='energy':
        plt.xlabel('Energy (eV)')
    if op=='raw':
        plt.title("Raw Absorbance")
        sns.lineplot(data=df,x='energy',y='absorbance',color='Black')
    if op=='smooth':
        plt.title("Smoothed Absorbance")
        sns.lineplot(data=df,x='energy',y='smoothed_absorbance',color='Black')
    plt.ylabel('Absorbance (a.u.)')
    plt.xlim(3.2,.55)
    plt.style.use('seaborn-paper')
    plt.savefig(op + '_abs',dpi=200,transparent=False,bbox_inches='tight')
    plt.show()

def plot_CP_diff(x,y,ev=0.04): #function to visually show separation of LCP and RCP from base abs
    coeff_L=poly.polyfit([x+ev for x in x],y,9) #LCP poly fit coeffs
    coeff_R=poly.polyfit([x-ev for x in x],y,9) #RCP poly fit coeffs
    fit_L=poly.polyval(x,coeff_L) #LCP line fit
    fit_R=poly.polyval(x,coeff_R) #RCP line fit

    fit_diff=(fit_L-fit_R)/(np.max(x)) #calculate LCP-RCP normalized to absorbance max.
    plt.figure(figsize=(6,6),dpi=80)

    plt.subplot(2,1,1)
    plt.ylabel('Absorbance (a.u.)')
    plt.xlim(3.2,.55)
    plt.scatter(x,y,s=1.3,c='Black')
    plt.plot(x,fit_L,c='Blue')
    plt.plot(x,fit_R,c='Red')
    plt.legend(('LCP','RCP','Raw'))

    plt.subplot(2,1,2)
    plt.ylabel('Absorbance (a.u.)')
    plt.xlabel('Energy (eV)')
    plt.xlim(3.0,.55)
    plt.plot(x,fit_diff,c='Purple')
    plt.legend(('Simulated MCD'))
    plt.savefig('Simulated MCD.png',dpi=200,transparent=False,bbox_inches='tight')
    plt.show()

    return fit_diff

def func(x,ev,y): #define simulated mcd function from absorbance spectrum
    coeffL=poly.polyfit(df_abs['energy']+ev,df_abs['absorbance'],9) #find polynomial coeffs from original absorption spectra
    coeffR=poly.polyfit(df_abs['energy']-ev,df_abs['absorbance'],9) #find polynomial coeffs from original absorption spectra
    LCP=poly.polyval(x,coeffL) #find y from +ev shifted LCP spectrum
    RCP=poly.polyval(x,coeffR) #find y from -ev shifted RCP spectrum

    # return LCP-RCP #return y from LCP-RCP
    return LCP-RCP-y #switch to this if doing y adjustment

def calc_effective_mass_and_plot(abs_fit,diff_dic,correction_factor=1):
    ev_list=[]
    std_dev_fit_list=[]
    m_list=[]
    B_list=[]
    for field in diff_dic.keys():
        if field is not '0':
            xdata=diff_dic[field].loc[diff_dic[field]['energy'].between(0.75, 2.0, inclusive=True),'energy']
            ydata=diff_dic[field].loc[diff_dic[field]['energy'].between(0.75, 2.0, inclusive=True),'avg-0T']   
            
            B_fit=int(field) #used for meV plotting later in zdf
            B=np.absolute(B_fit) #magnetic field (T)
            B_list.append(B_fit)

            ydata_normalized=ydata/(np.max(df_abs['absorbance'])*correction_factor) / 32982 # divided by mdeg conversion to obtain deltaA/A_Max
            # ydata_normalized=np.nan_to_num(ydata_normalized, nan=0.0)

            # if B_fit < 0:
            #     popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=0.00001,method='trf',bounds=(0.000005,0.001)) #lsf optimization to spit out zeeman split mev, guess is 10^-3 eV
            # if B_fit > 0:
            #     popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=-0.00001,method='trf',bounds=(-0.001,-0.000005)) #lsf optimization to spit out zeeman split mev.

            if B_fit < 0:
                popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=(0.0003,0.0001),method='trf',bounds=([0.00001,-0.01],[0.00070,0.01])) #multiple variables: bounds are: ([lower bounds],[upper bounds])
            if B_fit > 0:
                popt,pcov = optimize.curve_fit(func,xdata,ydata_normalized,p0=(-0.0003,-0.0001),method='trf',bounds=([-0.00070,-0.01],[-0.00001,0.01])) #lsf optimization to spit out zeeman split mev, guessing ~0.05 meV

            print(pcov) #list of residuals
            ev=popt[0] #return absolute val of minimzed ev to variable
            ev_list.append(ev*1000) #add ev to list as meV
            std_dev_of_fit=(np.sqrt(np.diag(pcov))*1000)[0] #return std dev of fitting
            std_dev_fit_list.append(std_dev_of_fit) #add std dev fit
            e=const.e #charge of electron (C)
            m_e=const.m_e #mass of electron (kg)
            w_c=ev/const.physical_constants['Planck constant in eV/Hz'][0] #cyclotron resonance frequency from Planck constant in eV/Hz
            effective_mass=e*B/w_c/2/m_e/const.pi #effective mass (m*/m_e)
            m_list.append(np.absolute(effective_mass)) #add m* to list

            fig,ax=plt.subplots(figsize=(2,4))
            plt.title(str(field) + 'T Fit')
            plt.ylabel(r'MCD ($\Delta$A/A$_{max}$)')
            plt.xlabel('Energy (eV)')
            plt.xlim(3.2,.55)
            plt.plot(xdata,ydata_normalized,label='experiment_data',c='Black')
            plt.plot(xdata,[func(x,*popt) for x in xdata],label='simulated_fit',c='Red')
            baseline = lines.Line2D(range(6),np.zeros(1),c='black',ls='--',lw=0.6) #draw baseline at 0T
            ax.add_line(baseline) #add baseline to plot
            plt.legend(loc=0)
            plt.text(3,0,'%.3f meV\n%.3f m*' % (ev*1000,effective_mass),fontweight='bold',bbox={'facecolor':'white','alpha':0.5,'pad':0.1}) #places text according to graph x,y coords
            plt.savefig(str(field) + "T_fit",dpi=100,transparent=False,bbox_inches='tight')
    average_ev = np.mean(ev_list)
    std_dev_ev = np.std(ev_list)
    average_m = np.mean(m_list)
    std_dev_m = np.std(m_list)
    zdf = pd.DataFrame(list(zip(B_list,ev_list,std_dev_fit_list,m_list)),columns=['B','E_Z','E_Z_std_dev','m*'])

    return average_ev, std_dev_ev, average_m, std_dev_m, zdf

def openHTML(f,title):
    f.write("<!DOCTYPE html>\n")
    f.write("<html lang='en'>\n")
    f.write("<head>\n")
    f.write('<base target="_blank"/>\n')
    f.write("<title>%s</title>\n" % title)
    f.write("</head>\n")
    f.write("<body>\n")
    f.write("<h1>%s</h1>\n" % title)

def writeHTMLimage(f,title,imgpath):
	f.write('<img src="%s" />\n' % imgpath)

def writeHTMLspacer(f,spacer):
    f.write('%s' % spacer)

def closeHTML(f):
	f.write("</body>\n")
	f.write("</html>\n")
	f.close()	

def writeHTMLfile_difference(file_name,report_date):
    f=open(file_name,'w')
    openHTML(f,'MCD ' + report_date + ' Report')
    writeHTMLspacer(f,'<div>\n')
    f.write('<p><b>Raw MCD Spectra</b></p>')
    writeHTMLimage(f,'raw_mcd','raw_mcd_sample.png')
    writeHTMLimage(f,'raw_mcd','raw_mcd_blank.png')
    writeHTMLspacer(f,'</div>\n<div>')
    f.write('<p><b>Average MCD Spectra</b></p>')
    writeHTMLimage(f,'avg_mcd','avg_mcd_sample.png')
    writeHTMLimage(f,'avg_mcd','avg_mcd_blank.png')
    writeHTMLspacer(f,'</div>\n<div>')
    f.write('<p><b>Diff MCD Spectra & Absorbance Spectra</b></p>')
    writeHTMLimage(f,'S-B_diff_mcd','avg_mcd_diff.png')
    writeHTMLimage(f,'S-0T_diff_mcd','avg_mcd_Diff_no_blank_0T_subbed.png')
    writeHTMLimage(f,'S-B-0T','avg_mcd_diff_0T_subbed.png') 
    writeHTMLimage(f,'raw_abs','raw_abs.png')
    writeHTMLimage(f,'smooth_abs','smooth_abs.png')
    writeHTMLspacer(f,'</div>\n<div>')
    f.write('<p><b>Diff Modulus MCD Spectra</b></p>')
    writeHTMLimage(f,'sample_mcd_modulus','avg_mcd_sample_modulus.png')
    writeHTMLimage(f,'blank_mcd_modulus','avg_mcd_blank_modulus.png')
    writeHTMLimage(f,'sample-0T','avg_mcd_sample-0T.png')  
    writeHTMLimage(f,'diff_mcd_modulus','avg_mcd_sub_modulus.png')
    writeHTMLimage(f,'diff_mcd_modulus_0T_subtracted','avg_mcd_sub_modulus_zero_subtracted.png')
    writeHTMLspacer(f,'</div>\n<div>')
    writeHTMLimage(f,'2T_fit','2T_fit.png')
    writeHTMLimage(f,'4T_fit','4T_fit.png')
    writeHTMLimage(f,'6T_fit','6T_fit.png')
    writeHTMLimage(f,'8T_fit','8T_fit.png')
    writeHTMLimage(f,'10T_fit','10T_fit.png')
    writeHTMLspacer(f,'</div>\n')
    f.write('<p><u>From the above data:</u></p>')
    f.write('<p>The average Zeeman splitting energy is <b>%.3f</b> \u00B1 %.4f meV.</p>' % (average_ev,std_dev_ev))
    f.write('<p>The average effective mass (<i>m*</i>) is  <b>%.3f</b> \u00B1 %.4f.</p>' % (average_m,std_dev_m))
    closeHTML(f)



'''-------------------------------FUNCTIONAL CODE BELOW-------------------------------'''


'''parse all data files'''
#Change these pathways as needed.
raw_mcd_dic = parse_mcd("/mnt/c/Users/roflc/Desktop/MCD DATA/5-1 CFS/MCD 05-18-21 NIR 5-1/") #raw mcd data in dictionary
df_abs =      parse_abs("/mnt/c/Users/roflc/Desktop/MCD DATA/5-1 CFS/ABS 05-17-21 5-1/NIR/Use/") #calculated abs data in dataframe

'''fit raw and avg mcd straight from datafile - no workup'''
plot_mcd(raw_mcd_dic,'raw',title='sample') #plot raw experimental mcd data
df_avgs = calc_raw_avg_mcd(raw_mcd_dic) #
plot_mcd(df_avgs,'avg',title='sample')

'''mcd difference (no blank)'''
for name, df in df_avgs.items():
    df_avgs[name]['avg-0T'] = df_avgs[name]['mdeg'] - df_avgs['0']['mdeg']
plot_mcd(df_avgs,'avg',title='Diff_no_blank_0T_subbed',ydata='avg-0T')

# '''perform absorbance simulation data fitting operations'''
plot_abs(df_abs,op='raw')
plot_abs(df_abs)

fit_diff=plot_CP_diff(df_abs['energy'],df_abs['absorbance'])
average_ev, std_dev_ev, average_m, std_dev_m, zdf = calc_effective_mass_and_plot(fit_diff,df_avgs,1)
print(zdf)
zdf.to_csv('zeeman_data.csv')

plt.clf() #Clear all previous plots
fig=plt.figure(figsize=(4,2))
ax1=fig.add_subplot(111) #need this to add separate series to same graph
ax1.scatter([x for x in zdf['B'] if x > 0], list(zdf.loc[zdf['B'] > 0,'E_Z']), label=r"$B(+)$",color="b")
ax1.scatter(np.absolute([x for x in zdf['B'] if x < 0]), list(zdf.loc[zdf['B'] < 0,'E_Z']), label=r"$B(-)$",color="r")
plt.legend(loc=0)
plt.xlabel('B (T)')
plt.xticks(np.arange(0,11,2))
ax1.set_xlim(1,11)
ax1.set_ylim(0,0.24)
plt.ylabel(r'$E_Z$ (meV)')
plt.savefig('mev_test_plot.png',dpi=200,bbox_inches='tight')
plt.show()

# '''write HTML file report'''
writeHTMLfile_difference('mcd_difference.html','03-30-2021, Both Max Signals')

xldf = pd.DataFrame()
for field, d in df_avgs.items():
    df = pd.DataFrame(d)
    df.dropna(axis=1,how='all',inplace=True)
    df.dropna(how='any',inplace=True)
    df.drop(['chpx','chpy','field','pemx','pemy','energy'],axis=1,inplace=True)
    df['avg-0T-deltaA']=df['avg-0T'] / 32982 * 1000 #give back deltaA x10^-3
    rename_list = {'deltaA':'{}_deltaA'.format(field),'mdeg':'{}_mdeg'.format(field),'avg-0T':'{}_avg-0T'.format(field),'avg-0T-deltaA':'{}_avg-0T-deltaA'.format(field)}
    df.rename(columns=rename_list,inplace=True)
    # print(df)
    try:
        xldf = xldf.merge(df, how='inner', on='wavelength')
    except KeyError:
        xldf = df

xldf = xldf.reindex(sorted(list(xldf), key=lambda x: x.split('_')[-1]), axis=1)
xldf.insert(0,'energy', [1240/x for x in xldf['wavelength']])
xldf.set_index('wavelength',inplace=True)
xldf.to_csv('CFS'+'_worked_up_diff_mcd.csv')

print("...\nDone!")
