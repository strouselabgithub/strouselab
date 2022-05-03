from functionsForMCD import *
import sys

if __name__ == '__main__':
    workingPath = '/home/jkusz/github/igloo/mcd/fitting'
    os.chdir(workingPath)
    
    material = 'Cd2SnO4-Solution'
    dataYear = '2021'
    max_ev = 6
    min_ev = 2

    # MCD Paths
    Cd2SnO4_UV_mcd = '/mnt/c/Users/roflc/Desktop/MCD DATA/Sn(II) MCD/Solution Data/Sig Cd2SnO4 Solution/'
    blank_mcd = '/mnt/c/Users/roflc/Desktop/MCD DATA/Sn(II) MCD/Solution Data/Blank Cd2SnO4 Solution/'

    CFS3_NIR_mcd = '/mnt/c/Users/roflc/Desktop/MCD DATA/3-1 CFS/NIR/MCD 04-13-21 NIR 3-1/'
    CFS5_NIR_mcd = '/mnt/c/Users/roflc/Desktop/MCD DATA/5-1 CFS/MCD 05-18-21 NIR 5-1/'
    CFS7_NIR_mcd = '/mnt/c/Users/roflc/Desktop/MCD DATA/7-1 CFS/NIR/MCD 04-02-21/'

    # ABS Paths
    CFS3_NIR_abs = "/mnt/c/Users/roflc/Desktop/MCD DATA/3-1 CFS/NIR/ABS 04-08-21 NIR/"
    # CFS3_VIS_abs = "/mnt/c/Users/roflc/Desktop/MCD DATA/3-1 CFS/VIS/ABS 04-07-21 VIS/"
    CFS5_NIR_abs = "/mnt/c/Users/roflc/Desktop/MCD DATA/5-1 CFS/ABS 05-17-21 5-1/NIR/Use/"
    # CFS5_VIS_abs = "/mnt/c/Users/roflc/Desktop/MCD DATA/5-1 CFS/ABS 05-17-21 5-1/VIS/"
    CFS7_NIR_abs = "/mnt/c/Users/roflc/Desktop/MCD DATA/7-1 CFS/ABS 03-29-21/"

    # 5-1 VT Paths
    # CFS5_VT_mcd = '/home/jkusz/github/strouse-data/raw_mcd/5-1 CFS/VT-MCD 05-21-21 NIR 5-1/'
    # CFS5_VT_abs = '/home/jkusz/github/strouse-data/raw_mcd/5-1 CFS/VT-Abs 05-22-21 NIR 5-1/'

    '''parse data from file'''
    signal_mcd = Cd2SnO4_UV_mcd
    ### Reminder to self: Add some sort of metadata file which tells which files output the series of csvs. Also have it so that new data makes its own folder based on material, and maybe a pop up window to change material variable name if necessary? ###
    signalData = parseMCD(signal_mcd,'signal',version=dataYear)
    blankData = parseMCD(blank_mcd,'blank')

    absData = parseABS(CFS5_NIR_abs) # absData only used for 2020 data

    averagedSignalData = calcAverageMCD(signalData)

    '''determine data processing path: if using old format or new'''
    if dataYear == '2020':
        finalMCDData = zeroSubtract(convertToCSV(averagedSignalData, spectra='MCD'))
        finalABSData = absData
    else:  
        averagedBlankData = calcAverageMCD(blankData)
        MCDSignal = convertToCSV(calcDifference(averagedSignalData, averagedBlankData, spectra='MCD'),spectra='MCD')
        absorptionData = convertToCSV(calcDifference(averagedSignalData, averagedBlankData, spectra='ABS'),spectra='ABS')
        # sys.exit()
        finalMCDData = zeroSubtract(MCDSignal)
        finalABSData = zeroSubtract(absorptionData)

    # print(finalMCDData, finalABSData)

    '''convert all data to csv for manual plotting'''
    finalMCDData.to_csv(material + '_worked_up_mcd.csv')
    finalABSData.to_csv(material + '_worked_up_absorption.csv')

    '''simulate MCD spectra and calculate m* parameters'''
    simulateMCDSpectra(finalABSData, max_ev, min_ev, version = dataYear)
    sys.exit()
    average_ev, std_dev_ev, average_m, std_dev_m, zdf = calc_effective_mass_and_plot(finalMCDData, finalABSData, max_ev, min_ev, material, correction_factor = 1, version = dataYear)
    plotABS(finalABSData, material, max_ev, min_ev)
    # zdf.to_csv(material + '_zeeman_data.csv')

    '''manually select peak positions to find maxima/minima for'''
    # peak1 = [2.5, 2.0]
    peak1 = [1.1, 0.8]
    peak2 = [1.4, 1.1]

    MCDPeaks = [peak1, peak2]
    # ,peak2,peak3]
    ABSPeaks = []

    '''find peaks in spectra and output into csv'''
    MCDAmplitudes = findAllPeaks(finalMCDData, MCDPeaks, spectra='MCD')
    MCDAmplitudes.to_csv(material + '_MCD_Amp_Peaks.csv', index=False)

    ABSAmplitudes = findAllPeaks(ABSCSVConversion, ABSPeaks, spectra='ABS')
    ABSAmplitudes.to_csv(material + '_ABS_Amp_Peaks.csv', index=False)

    print('Done!\n')