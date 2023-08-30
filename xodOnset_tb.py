# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaOnset_tb.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2023))::___
#
#
# XODMK Onset Detection Testbench
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
import soundfile as sf
# import librosa
# import librosa.display
import matplotlib.pyplot as plt


currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
audioSrcDir = rootDir + "/data/src/wav"
audioOutDir = rootDir + "/data/res/wavout"

print("rootDir: " + rootDir)
print("currentDir: " + currentDir)
print("audioSrcDir: " + audioSrcDir)
print("audioOutDir: " + audioOutDir)

sys.path.insert(0, rootDir+'/xodma')

from xodmaAudioTools import load_wav
from xodmaOnset import detectOnset, onset_strength, onset_backtrack
from xodmaSpectralTools import magphase, amplitude_to_db, stft, istft, peak_pick
from xodmaSpectralUtil import frames_to_samples, frames_to_time
from xodmaSpectralPlot import specshow

sys.path.insert(1, rootDir+'/xodUtil')
import xodPlotUtil as xodplt

#sys.path.insert(3, rootDir+'DSP')
#import xodClocks as clks
#import odmkSigGen1 as sigGen


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# // *---------------------------------------------------------------------* //

plt.close('all')

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //


def arrayFromFile(fname):
    ''' reads .dat data into Numpy array:
        fname is the name of existing file in dataInDir (defined above)
        example: newArray = arrayFromFile('mydata_in.dat') '''
        
    fileSrcFull = audioSrcDir+fname
        
    datalist = []
    with open(fileSrcFull, mode='r') as infile:
        for line in infile.readlines():
            datalist.append(float(line))
    arrayNm = np.array(datalist)
    
    fileSrc = os.path.split(fileSrcFull)[1]
    # src_path = os.path.split(sinesrc)[0]
    
    print('\nLoaded file: '+fileSrc)
    
    lgth1 = len(list(arrayNm))    # get length by iterating csvin obj (only way?)
    print('Length of data = '+str(lgth1))
    
    return arrayNm

# // *---------------------------------------------------------------------* //


print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::XODMK Spectral Mutate test::---*')
print('// *--------------------------------------------------------------* //')
print('// //////////////////////////////////////////////////////////////// //')


# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

# srcSel: 0 = wavSrc, 1 = amenBreak, 2 = sineWave48K, 
#         3 = multiSin test, 4 = text array input

srcSel = 1

plots = 1

plotOnsetEnv = 1

# STEREO source signal
#wavSrc = 'dsvco.wav'
#wavSrc = 'detectiveOctoSpace_one.wav'
#wavSrc = 'ebolaCallibriscian_uCCrhythm.wav'
#wavSrc = 'zoroastorian_mdychaos1.wav'
#wavSrc = 'The_Amen_Break_odmk.wav'

# MONO source signal
#wavSrc = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

wavSrcA = 'fromTheVoid_x0x.wav'

# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0

NFFT = 2048
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

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples

# Load Stereo/mono .wav file

if srcSel == 0:
    srcANm = wavSrcA
elif srcSel == 1:
    srcANm = 'The_Amen_Break_48K.wav'
elif srcSel == 2:
    srcANm = 'MonoSinOut_48K_560Hz_5p6sec.wav'
elif srcSel == 3:
    srcANm = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

audioSrcA = audioSrcDir + "/" + srcANm

[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrcA, wavLength)
print('\n// Loaded .wav file [ '+audioSrcA+' ]\n')

if aNumChannels == 2:
    aSrc_ch1 = aSrc[:, 0]
    aSrc_ch2 = aSrc[:, 1]
else:
    aSrc_ch1 = aSrc
    aSrc_ch2 = 0


# length of input signal - '0' => length of input .wav file
print('Channel A Source Audio:') 
print('aSrc Channels = '+str(len(np.shape(aSrc))))
print('length of input signal in seconds: ----- '+str(aLength))
print('length of input signal in samples: ----- '+str(aSamples))
print('audio sample rate: --------------------- '+str(afs))
print('wav file datatype: '+str(sf.info(audioSrcA).subtype))


sr = afs
aT = 1.0 / sr
print('\nSystem sample rate: -------------------- '+str(sr))
print('System sample period: ------------------ '+str(aT))


# if sample rates are different, resample B to A's rate
# ???

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

if 1:

    print('\n// *---:: detectOnset test ::---*')
    
    # " Default from 'large-scale search' "
    # peakThresh = 0.07
    # peakWait = 0.33
    
    peakThresh = 0.07
    peakWait = 5

    hop_length = 512
    backtrack = False
    
    plots = 1

    # optional simple call:
    # yOnsetSamples, yOnsetTime = detectOnset(aSrc_ch1, peakThresh, peakWait)

    yOnsetSamples, yOnsetTime = detectOnset(aSrc_ch1, peakThresh, peakWait, hop_length, sr, backtrack)

    if plotOnsetEnv == 1:
        onset_env = onset_strength(y=aSrc_ch1, sr=sr, hop_length=hop_length, aggregate=np.median)

        # Match tb peak params with xodmaOnset.py default **kwargs
        pre_max = 0.03 * sr // hop_length  # 30ms
        post_max = 0.00 * sr // hop_length + 1  # 0ms
        pre_avg = 0.10 * sr // hop_length  # 100ms
        post_avg = 0.10 * sr // hop_length + 1  # 100ms
        wait = peakWait  # 30ms
        delta = peakThresh

        # kwargs = {'pre_max': 2.0, 'post_max': 1.0, 'pre_avg': 9.0, 'post_avg': 10.0, 'wait': 30.0, 'delta': 0.07}
        kwargs = {'pre_max': pre_max, 'post_max': post_max, 'pre_avg': pre_avg,
                  'post_avg': post_avg, 'wait': wait, 'delta': delta}

        # # Peak pick the onset envelope
        onsets = peak_pick(onset_env, **kwargs)
        # pdb.set_trace()
        # Optionally backtrack the events
        if backtrack:
            onsets = onset_backtrack(onsets, onset_env)

    print('\ndetectOnset complete')


# // *---------------------------------------------------------------------* //
# // *--- Plot - source signal ---*

if 1:

    fnum = 3
    pltTitle = 'Input Signals: aSrc_ch1'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, len(aSrc_ch1), len(aSrc_ch1))

    xodplt.xodPlot1D(fnum, aSrc_ch1, xaxis, pltTitle, pltXlabel, pltYlabel)


# // *-----------------------------------------------------------------* //
# // *--- Plot Peak-Picking results vs. Spectrogram ---*

if plots > 0:
    # // *-----------------------------------------------------------------* //
    # // *--- Perform the STFT ---*

    NFFT = 2048
    ySTFT = stft(aSrc_ch1, NFFT)
    # assert (ySTFT.shape[1] == len(onset_env)), "Number of STFT frames != len onset_env"

    # times = frames_to_time(np.arange(ySTFT.shape[1]), sr, NFFT / 4)
    times = frames_to_time(np.arange(ySTFT.shape[1]), sr, NFFT / 4)
    plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
    ax = plt.subplot(2, 1, 1)

    specshow(amplitude_to_db(magphase(ySTFT)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
    # librosa.display.specshow(amplitude_to_db(magphase(ySTFT)[0], ref=np.max), y_axis='log',
    #                          x_axis='time', cmap=plt.cm.viridis)
    plt.title('CH1: Spectrogram (STFT)')

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(times, onset_env, alpha=0.66, label='Onset strength')
    plt.vlines(times[onsets], 0, onset_env.max(), color='r', alpha=0.8,
               label='Selected peaks')

    # plt.plot(times, onset_env, alpha=0.66, label='Onset strength')
    # plt.vlines(times[onsets], 0, onset_env.max(), color='r', alpha=0.8,
    #            label='Selected peaks')

    plt.legend(frameon=True, framealpha=0.66)
    plt.axis('tight')
    plt.tight_layout()

    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Onset Strength detection & Peak Selection')

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //

