# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodVocoder_tb.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2023))::___
#
# XODMK Phase Vocoder
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
audioSrcDir = rootDir + "/data/src/wav"
audioOutDir = rootDir + "/data/res/wavout"

print("currentDir: " + currentDir)
print("rootDir: " + rootDir)
print("audioSrcDir: " + audioSrcDir)
print("audioOutDir: " + audioOutDir)

sys.path.insert(0, rootDir + '/xodma')

from xodmaAudioTools import load_wav, write_wav, peak_pick
from xodmaSpectralTools import amplitude_to_db, stft, istft, magphase
from xodmaVocoder import pvTimeStretch, pvPitchShift, pvRobotStretch
from xodmaSpectralPlot import specshow
from xodmaMiscUtil import valid_audio


sys.path.insert(1, rootDir + '/xodUtil')
import xodPlotUtil as xodplt

# sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir + '/xodDSP')
import xodClocks as clks
import xodWavGen as wavGen

# temp python debugger - use >>> pdb.set_trace() to set break
import pdb

# // *---------------------------------------------------------------------* //

plt.close('all')


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def arrayFromFile(fname):
    """ reads .dat data into Numpy array:
        fname is the name of existing file in dataInDir (defined above)
        example: newArray = arrayFromFile('mydata_in.dat') """

    fileSrcFull = audioSrcDir + '/' + fname

    datalist = []
    with open(fileSrcFull, mode='r') as infile:
        for line in infile.readlines():
            datalist.append(float(line))
    arrayNm = np.array(datalist)

    fileSrc = os.path.split(fileSrcFull)[1]
    # src_path = os.path.split(sinesrc)[0]

    print('\nLoaded file: ' + fileSrc)

    lgth1 = len(list(arrayNm))  # get length by iterating csvin obj (only way?)
    print('Length of data = ' + str(lgth1))

    return arrayNm


def rolling_rms(x, winSz):
    # prepend zeros so rolling average result aligns with original time series
    zeroPadX = np.pad(x, (winSz, 0), mode='constant', constant_values=0)
    xc = np.cumsum(zeroPadX ** 2)
    return np.sqrt((xc[winSz:] - xc[:-winSz]) / winSz)


def plot_rolling_rms(x, winSz, plotNum, xName):
    rrms = rolling_rms(x, winSz)
    # pdb.set_trace()

    pltTitle = 'Rolling RMS: ' + xName
    pltXlabel = 'time-domain wav'
    pltYlabel = 'RMS Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    rraxis = np.linspace(0, len(rrms), len(rrms))
    # xodplt.xodPlot1D(plotNum, rrms, rraxis, pltTitle, pltXlabel, pltYlabel)
    # xodplt.xodPlot1D(plotNum, x, rraxis, pltTitle, pltXlabel, pltYlabel)

    rrmsArray = np.transpose(np.column_stack((x, rrms)))
    xodplt.xodMultiPlot1D(plotNum, rrmsArray, rraxis, pltTitle, pltXlabel, pltYlabel)


print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::XODMK Spectral Tools Experiments::---*')
print('// *--------------------------------------------------------------* //')
print('// //////////////////////////////////////////////////////////////// //')

# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

# Algorithm Select:

runAudioSTFT = 0        # Run EXP STFT <-> iSTFT
runVocoderPV1 = 1       # Run EXP using PV1 phase vocoder algorithm
runRobotSmith = 0       # Run EXP using pvRobotSmith phase vocoder algorithm

runPlots = 1            # Run Plotting


# srcSel: 0 = wavSrc, 1 = amenBreak, 2 = sineWave48K, 
#         3 = multiSin test, 4 = text array input

srcSel = 0


# MONO source signal
# wavSrc = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

# STEREO source signal
wavSrc = 'jahniBoyQuestaVox11.wav'
# wavSrc = 'opium_house_1.wav'
# wavSrc = 'dsvco.wav'
# wavSrc = 'detectiveOctoSpace_one.wav'
# wavSrc = 'ebolaCallibriscian_uCCrhythm.wav'


# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0

NFFT = 1024
STFTHOP = int(NFFT / 4)
WIN = 'hann'

''' Valid Window Types: 

boxcar
triang
blackman
hamming
hann
bartlett
flattop
parzen
bohman
blackmanharris
nuttall
barthann
kaiser (needs beta)
gaussian (needs standard deviation)
general_gaussian (needs power, width)
slepian (needs width)
dpss (needs normalized half-bandwidth)
chebwin (needs attenuation)
exponential (needs decay scale)
tukey (needs taper fraction)

'''

# define plotting fnum -> increment for each plot
fnum = 0

# // *---------------------------------------------------------------------* //
# // *----- Load .wav file -----*
# // *---------------------------------------------------------------------* //

# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples

# Load Stereo/mono .wav file

if srcSel == 0:
    srcNm = wavSrc
elif srcSel == 1:
    srcNm = 'The_Amen_Break_48K.wav'
elif srcSel == 2:
    srcNm = 'MonoSinOut_48K_560Hz_5p6sec.wav'
elif srcSel == 3:
    srcNm = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

audioSrc = audioSrcDir + '/' + srcNm

[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrc, wavLength, True)

# Expected return variable properties:
# >> aSrc.ndim = 2
# >> aSrc.dtype = ('float64')
# >> aSrc.shape = (xLength, 2)
# >> isinstance(aSrc, np.ndarray) = True

if aNumChannels == 2:
    aSrc_ch1 = aSrc[:, 0]
    aSrc_ch2 = aSrc[:, 1]
else:
    aSrc_ch1 = aSrc
    aSrc_ch2 = aSrc

# aT = 1.0 / afs
# print('\nsample period: ------------------------- '+str(aT))
# print('wav file datatype: '+str(sf.info(audioSrcA).subtype))


# // *--- Plot - source signal ---*

if runPlots:
    fnum += 1
    pltTitle = 'Input Waveform: ' + srcNm
    pltXlabel = 'time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, len(aSrc_ch1), len(aSrc_ch1))
    xodplt.xodPlot1D(fnum, aSrc_ch1, xaxis, pltTitle, pltXlabel, pltYlabel)

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

if runAudioSTFT:
    print('\n')
    print('// *---:: Audio STFT <-> iSTFT test ::---*')

    n_fft = NFFT
    aSrcSTFT = stft(aSrc_ch1, n_fft=n_fft)
    aSrcInverse = istft(aSrcSTFT, dtype=aSrcSTFT.dtype)

    if runPlots:
        # fnum += 1
        # pltTitle = 'Inverse STFT: aSrcInverse'
        # pltXlabel = 'time-domain wav'
        # pltYlabel = 'Magnitude'
        #
        # # define a linear space from 0 to 1/2 Fs for x-axis:
        # xaxis = np.linspace(0, len(aSrcInverse), len(aSrcInverse))
        # xodplt.xodPlot1D(fnum, aSrcInverse, xaxis, pltTitle, pltXlabel, pltYlabel)

        # Plot composite wav forms
        shape1 = aSrc_ch1.shape[0]
        shape2 = aSrcInverse.shape[0]
        if shape1 > shape2:
            # Pad array2 with zeros
            array1 = aSrc_ch1
            array2 = np.pad(aSrcInverse, (0, shape1 - shape2), mode='constant', constant_values=0)
        elif shape2 > shape1:
            # Pad array1 with zeros
            array1 = np.pad(aSrc_ch1, (0, shape2 - shape1), mode='constant', constant_values=0)
            array2 = aSrcInverse
        errArray = array1 - array2
        sigArray = np.transpose(np.column_stack((array1, array2, errArray)))

        fnum += 1
        pltTitle = 'Wave Source vs. Inverse STFT'
        pltXlabel = 'time-domain wav'
        pltYlabel = 'Magnitude'
        # define a linear space from 0 to 1/2 Fs for x-axis:
        xaxis = np.linspace(0, len(aSrcInverse), len(aSrcInverse))
        xodplt.xodMultiPlot1D(fnum, sigArray, xaxis, pltTitle, pltXlabel, pltYlabel)

        fnum += 1
        rrmsWinSize = STFTHOP
        rmsName = 'wavSrc_ch1'
        plot_rolling_rms(aSrc_ch1, rrmsWinSize, fnum, rmsName)

# // *---------------------------------------------------------------------* //

if runVocoderPV1:
    print('\n')
    print('// *---:: Phase Vocoder Time-Stretch EFX test ::---*')

    n_fft = NFFT

    timeCompress = 1.33     # Time Compress rate
    timeExpand = 1.0       # Time Expand rate

    yPV1_Compress_ch1 = pvTimeStretch(aSrc_ch1, timeCompress, n_fft)
    yPV1_Compress_ch2 = pvTimeStretch(aSrc_ch2, timeCompress, n_fft)

    yPV1_Compress = np.transpose(np.column_stack((yPV1_Compress_ch1, yPV1_Compress_ch2)))

    print('\nPerformed Time-Compress by ' + str(1/timeCompress))

    yPV1_Expand_ch1 = pvTimeStretch(aSrc_ch1, timeExpand, n_fft)
    yPV1_Expand_ch2 = pvTimeStretch(aSrc_ch2, timeExpand, n_fft)

    yPV1_Expand = np.transpose(np.column_stack((yPV1_Expand_ch1, yPV1_Expand_ch2)))

    print('Performed Time-Expand by ' + str(1/timeExpand))

    print('\n// *---:: Write .wav files ::---*')

    outFilePath = audioOutDir + '/yPV1_Original.wav'
    write_wav(outFilePath, aSrc, afs)

    outFilePath = audioOutDir + '/yPV1_Compress.wav'
    write_wav(outFilePath, yPV1_Compress, afs)

    outFilePath = audioOutDir + '/yPV1_Expand.wav'
    write_wav(outFilePath, yPV1_Expand, afs)

    print('\nOutput directory: ' + audioOutDir)
    print('wrote .wav file yPV1_Original.wav')
    print('wrote .wav file yPV1_Compress.wav')
    print('wrote .wav file yPV1_Expand.wav')

    if runPlots:
        # fnum += 1
        # pltTitle = 'Inverse STFT: aSrcInverse'
        # pltXlabel = 'time-domain wav'
        # pltYlabel = 'Magnitude'
        #
        # # define a linear space from 0 to 1/2 Fs for x-axis:
        # xaxis = np.linspace(0, len(aSrcInverse), len(aSrcInverse))
        # xodplt.xodPlot1D(fnum, aSrcInverse, xaxis, pltTitle, pltXlabel, pltYlabel)

        # Plot composite wav forms
        shape1 = aSrc_ch1.shape[0]
        shape2 = yPV1_Expand_ch1.shape[0]
        if shape1 > shape2:
            # Pad array2 with zeros
            array1 = aSrc_ch1
            array2 = np.pad(yPV1_Expand_ch1, (0, shape1 - shape2), mode='constant', constant_values=0)
        elif shape2 > shape1:
            # Pad array1 with zeros
            array1 = np.pad(aSrc_ch1, (0, shape2 - shape1), mode='constant', constant_values=0)
            array2 = yPV1_Expand_ch1
        errArray = array1 - array2
        sigArray = np.transpose(np.column_stack((array1, array2)))
        # sigArray = np.transpose(np.column_stack((array1, array2, errArray)))

        fnum += 1
        pltTitle = 'Wave Source vs. Inverse STFT'
        pltXlabel = 'time-domain wav'
        pltYlabel = 'Magnitude'
        # define a linear space from 0 to 1/2 Fs for x-axis:
        xaxis = np.linspace(0, len(array1), len(array1))
        xodplt.xodMultiPlot1D(fnum, sigArray, xaxis, pltTitle, pltXlabel, pltYlabel)

        fnum += 1
        pltTitle = 'Error Array (sourceWav - vocodedWav)'
        pltXlabel = 'time-domain wav'
        pltYlabel = 'Magnitude'

        # define a linear space from 0 to 1/2 Fs for x-axis:
        xodplt.xodPlot1D(fnum, errArray, xaxis, pltTitle, pltXlabel, pltYlabel)

        fnum += 1
        rrmsWinSize = STFTHOP
        rmsName = 'wavSrc_ch1'
        plot_rolling_rms(aSrc_ch1, rrmsWinSize, fnum, rmsName)


# // *---------------------------------------------------------------------* //

if runRobotSmith:
    print('\n')
    print('// *---:: Phase Vocoder RobotSmith EFX test ::---*')

    n_fft = NFFT

    timeCompress = 1.33     # Time Compress rate
    timeExpand = 0.56       # Time Expand rate

    # Vox Modulation Depth - range[0.01, 1.0] :
    # vxmod = 0.0
    vxmod = 0.015
    # vxmod = 0.55
    # vxmod = 0.1
    # vxmod = 6.25

    # Vox Mod Stereo
    vxtilt = -1.0   # -1 = tilt left, 1.0 = tilt right
    vxmodL = (vxtilt * vxmod)
    vxmodR = -(vxtilt * vxmod)

    yRS_Compress_ch1 = pvRobotStretch(aSrc_ch1, timeCompress, vxmodL, n_fft)
    yRS_Compress_ch2 = pvRobotStretch(aSrc_ch2, timeCompress, vxmodR, n_fft)
    yRS_Compress = np.transpose(np.column_stack((yRS_Compress_ch1, yRS_Compress_ch2)))

    print('\nPerformed Time-Compress by ' + str(1/timeCompress))

    yRS_Expand_ch1 = pvRobotStretch(aSrc_ch1, timeExpand, vxmodL, n_fft)
    yRS_Expand_ch2 = pvRobotStretch(aSrc_ch2, timeExpand, vxmodR, n_fft)
    yRS_Expand = np.transpose(np.column_stack((yRS_Expand_ch1, yRS_Expand_ch2)))

    print('Performed Time-Expand by ' + str(1/timeExpand))

    print('\n// *---:: Write .wav files ::---*')

    outFilePath = audioOutDir + '/yRS_Original.wav'
    write_wav(outFilePath, aSrc, afs)

    outFilePath = audioOutDir + '/yRS_Compress.wav'
    write_wav(outFilePath, yRS_Compress, afs)

    outFilePath = audioOutDir + '/yRS_Expand.wav'
    write_wav(outFilePath, yRS_Expand, afs)

    print('\nOutput directory: ' + audioOutDir)
    print('wrote .wav file yRS_Original.wav')
    print('wrote .wav file yRS_Compress.wav')
    print('wrote .wav file yRS_Expand.wav')


# // *---------------------------------------------------------------------* //

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //


# reference C code


# int pva(float *input, float *window, float *output,
#        int input_size, int fftsize, int hopsize, float sr){
#
# int posin, posout, i, k, mod;
# float *sigframe, *specframe, *lastph;
# float fac, scal, phi, mag, delta, pi = (float)twopi/2;
#
# sigframe = new float[fftsize];
# specframe = new float[fftsize];
# lastph = new float[fftsize/2];
# memset(lastph, 0, sizeof(float)*fftsize/2);
#
# fac = (float) (sr/(hopsize*twopi));
# scal = (float) (twopi*hopsize/fftsize);
#
# for(posin=posout=0; posin < input_size; posin+=hopsize){
#      mod = posin%fftsize;
#	   # window & rotate a signal frame
#      for(i=0; i < fftsize; i++) 
#          if(posin+i < input_size)
#            sigframe[(i+mod)%fftsize]
#                     = input[posin+i]*window[i];
#           else sigframe[(i+mod)%fftsize] = 0;
#
#      // transform it
#      fft(sigframe, specframe, fftsize);
#
#      // convert to PV output
#      for(i=2,k=1; i < fftsize; i+=2, k++){
#
#      // rectangular to polar
#      mag = (float) sqrt(specframe[i]*specframe[i] + 
#                        specframe[i+1]*specframe[i+1]);  
#      phi = (float) atan2(specframe[i+1], specframe[i]);
#      // phase diffs
#      delta = phi - lastph[k];
#      lastph[k] = phi;
#         
#      // unwrap the difference, so it lies between -pi and pi
#      while(delta > pi) delta -= (float) twopi;
#      while(delta < -pi) delta += (float) twopi;
#
#      // construct the amplitude-frequency pairs
#      specframe[i] = mag;
#	  specframe[i+1] = (delta + k*scal) * fac;
#
#      }
#      // output it
#      for(i=0; i < fftsize; i++, posout++)
#			  output[posout] = specframe[i];
#		  
# }
# delete[] sigframe;
# delete[] specframe;
# delete[] lastph;
#
# return posout;
# }


# int pvs(float* input, float* window, float* output,
#          int input_size, int fftsize, int hopsize, float sr){
#
# int posin, posout, k, i, output_size, mod;
# float *sigframe, *specframe, *lastph;
# float fac, scal, phi, mag, delta;
#
# sigframe = new float[fftsize];
# specframe = new float[fftsize];
# lastph = new float[fftsize/2];
# memset(lastph, 0, sizeof(float)*fftsize/2);
#
# output_size = input_size*hopsize/fftsize;
#
# fac = (float) (hopsize*twopi/sr);
# scal = sr/fftsize;
#
# for(posout=posin=0; posout < output_size; posout+=hopsize){
#
#   // load in a spectral frame from input 
#   for(i=0; i < fftsize; i++, posin++)
#        specframe[i] = input[posin];
#	
# // convert from PV input to DFT coordinates
# for(i=2,k=1; i < fftsize; i+=2, k++){
#   delta = (specframe[i+1] - k*scal)*fac;
#   phi = lastph[k]+delta;
#   lastph[k] = phi;
#   mag = specframe[i];
#  
#  specframe[i] = (float) (mag*cos(phi));
#  specframe[i+1] = (float) (mag*sin(phi)); 
#  
# }
#   // inverse-transform it
#   ifft(specframe, sigframe, fftsize);
#
#   // unrotate and window it and overlap-add it
#   mod = posout%fftsize;
#   for(i=0; i < fftsize; i++)
#       if(posout+i < output_size)
#          output[posout+i] += sigframe[(i+mod)%fftsize]*window[i];
# }
# delete[] sigframe;
# delete[] specframe;
# delete[] lastph;
#
# return output_size;
# }
