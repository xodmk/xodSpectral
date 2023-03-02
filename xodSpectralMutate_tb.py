# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodSpectralMutate_tb.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
#
# XODMK Phase Vocoder Spectral Mutator EFFX
#
# Temp: Using Librosa vocoder implementation for model
#
# Single (Stereo) Channel:
#    - input stereo signal (.wav, etc.), detect onset events, separate into regions,
#      permutate and shuffle onset regions
#      * XFade - crossfade between regions
#      * time-stretch - re-time regions to replace existing region with same duratioin
#
# Dual (Stereo) Channel:
#    - intput master & slave signals (.wav, etc.), detect onset events,
#      separate into regions, splice slave regions into master
#      * XFade - crossfade between regions
# 
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
# import scipy as sp
import soundfile as sf
# import librosa
# import librosa.display
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
from xodmaAudioTools import load_wav, write_wav, resample  # , valid_audio
from xodmaAudioTools import fix_length  # , samples_to_time, time_to_samples
from xodmaOnset import onset_strength
from xodmaSpectralTools import amplitude_to_db, stft, istft, peak_pick
from xodmaSpectralTools import phase_vocoder, magphase
from xodmaSpectralUtil import frames_to_time
from xodmaSpectralPlot import specshow

sys.path.insert(0, rootDir + '/xodUtil')
import xodPlotUtil as xodplt

# sys.path.insert(3, rootDir+'DSP')
# import xodClocks as clks
# import odmkSigGen1 as sigGen


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

    fileSrcFull = xdir.audioSrcDir + fname

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


def get_peak_regions(peaks, length):
    ''' returns an array of peak regions (number of samples between peaks '''

    peak_regions = np.zeros((len(peaks) + 1))
    for i in range(len(peaks) + 1):
        if i == 0:
            peak_regions[0] = peaks[0]
        elif i == len(peaks):
            peak_regions[i] = length - peaks[i - 1]
        else:
            peak_regions[i] = peaks[i] - peaks[i - 1]

    return peak_regions


def time_stretch(y, rate):
    ''' Time-stretch an audio series by a fixed rate.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    rate : float > 0 [scalar]
        Stretch factor
        If `rate > 1`, then the signal is sped up
        If `rate < 1`, then the signal is slowed down

    Returns
    -------
    y_stretch : np.ndarray [shape=(rate * n,)]
        audio time series stretched by the specified rate


    Examples
    --------
    Compress to be twice as fast
    (xodAudiotools.load_wav)
    >>> [y, numChan, sr, aLength, aSamples] = load_wav(audioSrc, wavLength)
    >>> y_fast = time_stretch(y, 2.0)

    Or half the original speed

    >>> y_slow = time_stretch(y, 0.5) '''

    if rate <= 0:
        print('\nrate must be a positive number')
        return

    ySTFT = stft(y)
    stftStretch = phase_vocoder(ySTFT, rate)
    yStretch = istft(stftStretch, dtype=y.dtype)

    return yStretch


def pitch_shift(y, sr, n_steps, bins_per_octave=12):
    ''' Pitch-shift the waveform by `n_steps` half-steps.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time-series

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    n_steps : float [scalar]
        how many (fractional) half-steps to shift `y`

    bins_per_octave : float > 0 [scalar]
        how many steps per octave


    Returns
    -------
    y_shift : np.ndarray [shape=(n,)]
        The pitch-shifted audio time-series


    Examples
    --------
    Shift up by a major third (four half-steps)

    (xodAudiotools.load_wav)
    >>> [y, numChan, sr, aLength, aSamples] = load_wav(audioSrc, wavLength)
    >>> y_third = pitch_shift(y, sr, n_steps=4)

    Shift down by a tritone (six half-steps)

    >>> y_tritone = pitch_shift(y, sr, n_steps=-6)

    Shift up by 3 quarter-tones

    >>> y_three_qt = pitch_shift(y, sr, n_steps=3, bins_per_octave=24) '''

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.int):
        sys.exit('ERROR: func pitch_shift - bins_per_octave must be a positive integer.')

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = resample(time_stretch(y, rate), float(sr) / rate, sr)

    # Crop to the same dimension as the input
    return fix_length(y_shift, len(y))


# // *---------------------------------------------------------------------* //


def spectraMutate(y, NFFT, fs, ySeed, peakCtrl=32, ULR=2):
    '''Time-stretch an audio series by a fixed rate.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] audio time series
    
    NFFT : number of FFT points
    
    fs : audio sample rate
    
    ySeed : permutation seed
    
    ULR : 0=uncoupled / 1=left / 2=right

    Returns
    -------
    yExp : np.ndarray [shape=(rate * n,)]
        audio time series spectrally mutated


    Examples:
        NFFT = 2048, 
        pmseed = 5
        ULR = 2
        yRxExp = spectrExp(ySrc, NFFT, fs, pmseed, ULR=ULR)

    '''

    # peak_pick

    # peaks = peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)

    #    pre_max   : int >= 0 [scalar]
    #        number of samples before `n` over which max is computed
    #
    #    post_max  : int >= 1 [scalar]
    #        number of samples after `n` over which max is computed
    #
    #    pre_avg   : int >= 0 [scalar]
    #        number of samples before `n` over which mean is computed
    #
    #    post_avg  : int >= 1 [scalar]
    #        number of samples after `n` over which mean is computed
    #
    #    delta     : float >= 0 [scalar]
    #        threshold offset for mean
    #
    #    wait      : int >= 0 [scalar]
    #        number of samples to wait after picking a peak
    #
    #    Returns
    #    -------
    #    peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
    #        indices of peaks in `x`

    y_ch1 = y[:, 0];
    y_ch2 = y[:, 1];

    # onset_env_ch1 = onset_strength(y_ch1, fs, hop_length=512, aggregate=np.median)
    # onset_env_ch2 = onset_strength(y_ch2, fs, hop_length=512, aggregate=np.median)
    # currently uses fixed hop_length
    onset_env_ch1 = onset_strength(y_ch1, fs, NFFT / 4, aggregate=np.median)
    onset_env_ch2 = onset_strength(y_ch2, fs, NFFT / 4, aggregate=np.median)

    pkctrl = peakCtrl

    peaks_ch1 = peak_pick(onset_env_ch1, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)
    peaks_ch2 = peak_pick(onset_env_ch2, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)

    # peak_onsets_ch1 = np.array(onset_env_ch1)[peaks_ch1]
    # peak_onsets_ch2 = np.array(onset_env_ch2)[peaks_ch2]

    # // *-----------------------------------------------------------------* //
    # // *--- Calculate Peak Regions (# frames of peak regions) ---*    

    peak_regions_ch1 = get_peak_regions(peaks_ch1, len(onset_env_ch1))
    peak_regions_ch2 = get_peak_regions(peaks_ch2, len(onset_env_ch2))

    # // *-----------------------------------------------------------------* //
    # // *--- Perform the STFT ---*

    y_ch1 = y[:, 0];
    y_ch2 = y[:, 1];

    ySTFT_ch1 = stft(y_ch1, NFFT)
    ySTFT_ch2 = stft(y_ch2, NFFT)

    assert (ySTFT_ch1.shape[1] == len(onset_env_ch1)), "Number of STFT frames != len onset_env"

    # // *-----------------------------------------------------------------* //
    # // *--- Plot Peak-Picking results vs. Spectrogram ---*

    # times_ch1 = frames_to_time(np.arange(len(onset_env_ch1)), fs, hop_length=512)
    # currently uses fixed hop_length
    times_ch1 = frames_to_time(np.arange(len(onset_env_ch1)), fs, NFFT / 4)

    plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
    ax = plt.subplot(2, 1, 1)
    specshow(amplitude_to_db(magphase(ySTFT_ch1)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
    plt.title('CH1: Spectrogram (STFT)')

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(times_ch1, onset_env_ch1, alpha=0.66, label='Onset strength')
    plt.vlines(times_ch1[peaks_ch1], 0, onset_env_ch1.max(), color='r', alpha=0.8,
               label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.66)
    plt.axis('tight')
    plt.tight_layout()

    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Onset Strength detection & Peak Selection')

    times_ch2 = frames_to_time(np.arange(len(onset_env_ch2)), fs, hop_length=512)

    plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
    ax = plt.subplot(2, 1, 1)
    specshow(amplitude_to_db(magphase(ySTFT_ch2)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
    plt.title('CH2: Spectrogram (STFT)')

    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(times_ch2, onset_env_ch2, alpha=0.66, label='Onset strength')
    plt.vlines(times_ch2[peaks_ch2], 0, onset_env_ch2.max(), color='r', alpha=0.8,
               label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.66)
    plt.axis('tight')
    plt.tight_layout()

    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Onset Strength detection & Peak Selection')

    # // *-----------------------------------------------------------------* //
    # // *--- Spectral region mutator v2 ---*    

    np.random.seed(ySeed)

    permuteIdx_ch1 = np.random.permutation(len(peaks_ch1))
    permuteIdx_ch2 = np.random.permutation(len(peaks_ch2))
    # permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))

    peaks_permute_ch1 = np.array(peaks_ch1)[permuteIdx_ch1]
    peaks_permute_ch2 = np.array(peaks_ch2)[permuteIdx_ch2]

    # temp inspect
    #    for i in range(len(peaks_permute_ch1)):
    #        print('permuteIdx_ch1['+str(i)+'] = '+str(permuteIdx_ch1[i])+',  peaks_permute_ch1['+str(i)+'] = '+str(peaks_permute_ch1[i]) )
    #    for i in range(len(peaks_permute_ch2)):
    #        print('permuteIdx_ch2['+str(i)+'] = '+str(permuteIdx_ch2[i])+',  peaks_permute_ch2['+str(i)+'] = '+str(peaks_permute_ch2[i]) )

    # keep the zero'th region (before the first peak) at the beginning (do no permute)
    regions_permute_ch1 = np.append(peak_regions_ch1[0],
                                    np.array(peak_regions_ch1[1:len(peak_regions_ch1)])[permuteIdx_ch1])
    regions_permute_ch2 = np.append(peak_regions_ch2[0],
                                    np.array(peak_regions_ch2[1:len(peak_regions_ch2)])[permuteIdx_ch2])

    zSTFT_ch1 = np.zeros((int(NFFT / 2 + 1), ySTFT_ch1.shape[1]), dtype='complex64')
    zSTFT_ch2 = np.zeros((int(NFFT / 2 + 1), ySTFT_ch2.shape[1]), dtype='complex64')

    if ULR == 2:

        # fill in region before first peak
        for i in range(int(regions_permute_ch2[0])):
            zSTFT_ch1[:, i] = ySTFT_ch1[:, i]
            zSTFT_ch2[:, i] = ySTFT_ch2[:, i]
        # permutate spectral regions & build new STFT
        zIdx_ch2 = regions_permute_ch2[0]
        for j in range(len(peaks_permute_ch2)):
            for k in range(int(regions_permute_ch2[j + 1])):
                zSTFT_ch1[:, int(zIdx_ch2 + k)] = ySTFT_ch1[:, int(peaks_permute_ch2[j] + k)]
                zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_ch2[:, int(peaks_permute_ch2[j] + k)]
            zIdx_ch2 += regions_permute_ch2[j + 1]

    elif ULR == 1:

        # fill in region before first peak
        for i in range(int(regions_permute_ch1[0])):
            zSTFT_ch1[:, i] = ySTFT_ch1[:, i]
            zSTFT_ch2[:, i] = ySTFT_ch2[:, i]
        # permutate spectral regions & build new STFT
        zIdx_ch1 = regions_permute_ch1[0]
        for j in range(len(peaks_permute_ch1)):
            for k in range(int(regions_permute_ch1[j + 1])):
                zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_ch1[:, int(peaks_permute_ch1[j] + k)]
                zSTFT_ch2[:, int(zIdx_ch1 + k)] = ySTFT_ch2[:, int(peaks_permute_ch1[j] + k)]
            zIdx_ch1 += regions_permute_ch1[j + 1]

    else:

        # fill in region before first peak
        for i in range(int(regions_permute_ch1[0])):
            zSTFT_ch1[:, i] = ySTFT_ch1[:, i]
        zIdx_ch1 = regions_permute_ch1[0]
        for j in range(len(peaks_permute_ch1)):
            for k in range(int(regions_permute_ch1[j + 1])):
                zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_ch1[:, int(peaks_permute_ch1[j] + k)]
            zIdx_ch1 += regions_permute_ch1[j + 1]

        # fill in region before first peak
        for i in range(int(regions_permute_ch2[0])):
            zSTFT_ch2[:, i] = ySTFT_ch2[:, i]
        zIdx_ch2 = regions_permute_ch2[0]
        for j in range(len(peaks_permute_ch2)):
            for k in range(int(regions_permute_ch2[j + 1])):
                zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_ch2[:, int(peaks_permute_ch2[j] + k)]
            zIdx_ch2 += regions_permute_ch2[j + 1]

    # pdb.set_trace()

    # // *-----------------------------------------------------------------* //
    # // *--- Plot region mutated Spectrogram ---*
    # STFT Visualization:

    D_ch1 = amplitude_to_db(np.abs(zSTFT_ch1), ref=np.max)
    D_ch2 = amplitude_to_db(np.abs(zSTFT_ch2), ref=np.max)

    # Linear-frequency power spectrogram

    #    fig203 = plt.figure(num=203, facecolor='silver', edgecolor='k', figsize=(12, 8))
    #
    #    specshow(D, x_axis='time', y_axis='linear', sr=48000, cmap=plt.cm.viridis)
    #    plt.colorbar(format='%+2.0f dB')
    #    #plt.imshow(Z, cmap=plt.cm.cubehelix_r)
    #    plt.title('Linear-frequency power spectrogram')

    # Log-frequency power spectrogram

    plt.figure(num=203, facecolor='silver', edgecolor='k', figsize=(12, 8))
    specshow(D_ch1, x_axis='time', y_axis='log', sr=48000, cmap=plt.cm.viridis)
    plt.colorbar(format='%+2.0f dB')
    plt.title('CH1: Log-frequency power spectrogram')

    plt.figure(num=204, facecolor='silver', edgecolor='k', figsize=(12, 8))
    specshow(D_ch2, x_axis='time', y_axis='log', sr=48000, cmap=plt.cm.viridis)
    plt.colorbar(format='%+2.0f dB')
    plt.title('CH2: Log-frequency power spectrogram')

    plt.show()

    # // *-----------------------------------------------------------------* //

    # Invert the stft
    yExp_ch1 = istft(zSTFT_ch1, dtype=y.dtype)
    yExp_ch2 = istft(zSTFT_ch2, dtype=y.dtype)

    yExp = np.transpose(np.column_stack((yExp_ch1, yExp_ch2)))

    return yExp


#   ___Single Channel EXP___

#    onset_env = onset_strength(y, fs, hop_length=512, aggregate=np.median)
#    
#    #peaks = peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
#    peaks = peak_pick(onset_env, 6, 6, 6, 6, 0.5, 8)
#    
#    peak_onsets = np.array(onset_env)[peaks]
#
#
#    # // *-----------------------------------------------------------------* //
#    # // *--- Calculate Peak Regions (# frames of peak regions) ---*    
#    peak_regions = np.zeros((len(peaks)+1))
#    for i in range(len(peaks)+1):
#        if i == 0:
#            peak_regions[0] = peaks[0]
#        elif i == len(peaks):
#            peak_regions[i] = len(onset_env) - peaks[i-1]
#        else:
#            peak_regions[i] = peaks[i] - peaks[i-1]
#
#
#    # // *-----------------------------------------------------------------* //
#    # // *--- Perform the STFT ---*
#    ySTFT = stft(y, NFFT)
#
#    assert (ySTFT.shape[1] == len(onset_env)), "Number of STFT frames != len onset_env"    
#    
#    
#    # // *-----------------------------------------------------------------* //
#    # // *--- Spectral region mutator v1 ---*    
#    
#    np.random.seed(ySeed)
#    permuteIdx = np.random.permutation(len(peaks))
#    #permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))
#    peaks_permute = np.array(peaks)[permuteIdx]
#    
#    # temp inspect
#    for i in range(len(peaks_permute)):
#        print('permuteIdx['+str(i)+'] = '+str(permuteIdx[i])+',  peaks_permute['+str(i)+'] = '+str(peaks_permute[i]) )
#    
#        
#    # keep the zero'th region (before the first peak) at the beginning (do no permute)
#    regions_permute = np.append( peak_regions[0], 
#                                 np.array(peak_regions[1:len(peak_regions)])[permuteIdx] )
#    
#    
#    zSTFT = np.zeros((int(NFFT/2+1),ySTFT.shape[1]), dtype='complex64')
#    # fill in region before first peak
#    for i in range(int(regions_permute[0])):
#        zSTFT[:, i] = ySTFT[:, i]
#    zIdx = regions_permute[0]
#    for j in range(len(peaks_permute)):
#        for k in range(int(regions_permute[j+1])):
#            zSTFT[:, int(zIdx + k)] = ySTFT[:, int(peaks_permute[j]+k)]
#        zIdx += regions_permute[j+1]
#    
#    #pdb.set_trace()
#
#    
#    # // *-----------------------------------------------------------------* //
#    # // *--- Plot region mutated Spectrogram ---*
#        #STFT Visualization:
#    
#    D = amplitude_to_db(np.abs(zSTFT), ref=np.max)
#    
#    
#    # Linear-frequency power spectrogram
#    
##    fig203 = plt.figure(num=203, facecolor='silver', edgecolor='k', figsize=(12, 8))
##    
##    specshow(D, x_axis='time', y_axis='linear', sr=48000, cmap=plt.cm.viridis)
##    plt.colorbar(format='%+2.0f dB')
##    #plt.imshow(Z, cmap=plt.cm.cubehelix_r)
##    plt.title('Linear-frequency power spectrogram')
#    
#    
#    # Log-frequency power spectrogram
#    
#    plt.figure(num=204, facecolor='silver', edgecolor='k', figsize=(12, 8))
#    specshow(D, x_axis='time', y_axis='log', sr=48000, cmap=plt.cm.viridis)
#    plt.colorbar(format='%+2.0f dB')
#    plt.title('Log-frequency power spectrogram')
#    
#    plt.show()
#
#    # // *-----------------------------------------------------------------* //
#    
#
#    # Invert the stft
#    yExp = istft(zSTFT, dtype=y.dtype)
#
#    return yExp


def spectraMutate2CH(a, b, NFFT, fs, ySeed, peakCtrl=32, ULR=0):
    '''Time-stretch an audio series by a fixed rate.

    Parameters
    ----------
    a/b : np.ndarray [shape=(n, 2)] audio time series
    
    NFFT : number of FFT points
    
    fs : audio sample rate
    
    ySeed : permutation seed
    
    ppkSel : peak pick parameter selection - parameterizes peak pick algorithm
           : smaller number allows for closer peaks, larger number more spaced peaks
           : '3s', '6s', '9s', '12s'
    
    ULR : processes L/R channels separately (stereo), or uses only L or R channel
        : 0=stereo / 1=left / 2=right

    Returns
    -------
    yExp : np.ndarray [shape=(rate * n,)]
        audio time series spectrally mutated


    Examples:
        NFFT = 2048, 
        pmseed = 5
        ULR = 2
        yRxExp = spectrExp(ySrc, NFFT, fs, pmseed, ULR=ULR) '''

    # peak_pick

    # peaks = peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)

    #    pre_max   : int >= 0 [scalar]
    #        number of samples before `n` over which max is computed
    #
    #    post_max  : int >= 1 [scalar]
    #        number of samples after `n` over which max is computed
    #
    #    pre_avg   : int >= 0 [scalar]
    #        number of samples before `n` over which mean is computed
    #
    #    post_avg  : int >= 1 [scalar]
    #        number of samples after `n` over which mean is computed
    #
    #    delta     : float >= 0 [scalar]
    #        threshold offset for mean
    #
    #    wait      : int >= 0 [scalar]
    #        number of samples to wait after picking a peak
    #
    #    Returns
    #    -------
    #    peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
    #        indices of peaks in `x`

    Ach1 = a[:, 0]
    Ach2 = a[:, 1]

    Bch1 = b[:, 0]
    Bch2 = b[:, 1]

    # pdb.set_trace()

    onset_env_Ach1 = onset_strength(Ach1, fs, hop_length=int(NFFT / 4), aggregate=np.median)
    onset_env_Ach2 = onset_strength(Ach2, fs, hop_length=int(NFFT / 4), aggregate=np.median)

    onset_env_Bch1 = onset_strength(Bch1, fs, hop_length=int(NFFT / 4), aggregate=np.median)
    onset_env_Bch2 = onset_strength(Bch2, fs, hop_length=int(NFFT / 4), aggregate=np.median)

    pkctrl = peakCtrl

    peaks_Ach1 = peak_pick(onset_env_Ach1, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)
    peaks_Ach2 = peak_pick(onset_env_Ach2, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)

    peaks_Bch1 = peak_pick(onset_env_Bch1, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)
    peaks_Bch2 = peak_pick(onset_env_Bch2, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)

    # peak_onsets_Ach1 = np.array(onset_env_Ach1)[peaks_Ach1]
    # peak_onsets_Ach2 = np.array(onset_env_Ach2)[peaks_Ach2]

    # // *-----------------------------------------------------------------* //
    # // *--- Calculate Peak Regions (# frames of peak regions) ---*    

    peak_regions_Ach1 = get_peak_regions(peaks_Ach1, len(onset_env_Ach1))
    peak_regions_Ach2 = get_peak_regions(peaks_Ach2, len(onset_env_Ach2))

    peak_regions_Bch1 = get_peak_regions(peaks_Bch1, len(onset_env_Bch1))
    peak_regions_Bch2 = get_peak_regions(peaks_Bch2, len(onset_env_Bch2))

    numPeaksA = [len(peaks_Ach1), len(peaks_Ach2)]
    numPeaksB = [len(peaks_Bch1), len(peaks_Bch2)]

    print('numPeaksA = ' + str(numPeaksA[1]) + ',  numPeaksB = ' + str(numPeaksB[1]))

    # pdb.set_trace()

    # // *-----------------------------------------------------------------* //
    # // *--- Perform the STFT ---*

    ySTFT_Ach1 = stft(Ach1, NFFT)
    ySTFT_Ach2 = stft(Ach2, NFFT)

    ySTFT_Bch1 = stft(Bch1, NFFT)
    ySTFT_Bch2 = stft(Bch2, NFFT)

    assert (ySTFT_Ach1.shape[1] == len(onset_env_Ach1)), "CHA: # of STFT frames != len onset_env"
    assert (ySTFT_Bch1.shape[1] == len(onset_env_Bch1)), "CHB: # of STFT frames != len onset_env"

    print('ySTFT_Ach2.shape = ' + str(ySTFT_Ach2.shape))

    # // *-----------------------------------------------------------------* //
    # // *--- Plot Peak-Picking results vs. Spectrogram ---*

    if 1:
        # times_ch1 = frames_to_time(np.arange(len(onset_env_Ach1)), fs, hop_length=512)
        times_ch1 = frames_to_time(np.arange(len(onset_env_Ach1)), fs, hop_length=int(NFFT / 4))

        plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
        ax = plt.subplot(2, 1, 1)
        specshow(amplitude_to_db(magphase(ySTFT_Ach1)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
        plt.title('CH1: Spectrogram (STFT)')

        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(times_ch1, onset_env_Ach1, alpha=0.66, label='Onset strength')
        plt.vlines(times_ch1[peaks_Ach1], 0, onset_env_Ach1.max(), color='r', alpha=0.8,
                   label='Selected peaks')
        plt.legend(frameon=True, framealpha=0.66)
        plt.axis('tight')
        plt.tight_layout()

        plt.xlabel('time')
        plt.ylabel('Amplitude')
        plt.title('Onset Strength detection & Peak Selection')

        # times_ch2 = frames_to_time(np.arange(len(onset_env_Ach2)), fs, hop_length=512)
        times_ch2 = frames_to_time(np.arange(len(onset_env_Ach2)), fs, hop_length=int(NFFT / 4))

        plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
        ax = plt.subplot(2, 1, 1)
        specshow(amplitude_to_db(magphase(ySTFT_Ach2)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
        plt.title('CH2: Spectrogram (STFT)')

        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(times_ch2, onset_env_Ach2, alpha=0.66, label='Onset strength')
        plt.vlines(times_ch2[peaks_Ach2], 0, onset_env_Ach2.max(), color='r', alpha=0.8,
                   label='Selected peaks')
        plt.legend(frameon=True, framealpha=0.66)
        plt.axis('tight')
        plt.tight_layout()

        plt.xlabel('time')
        plt.ylabel('Amplitude')
        plt.title('Onset Strength detection & Peak Selection')

    # // *-----------------------------------------------------------------* //
    # // *--- Spectral region mutator v2 ---*    

    np.random.seed(ySeed)

    permuteIdx_Bch1 = np.random.permutation(len(peaks_Bch1))
    # permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))

    peaks_permute_Bch1 = np.array(peaks_Bch1)[permuteIdx_Bch1]

    permuteIdx_Bch2 = np.random.permutation(len(peaks_Bch2))
    # permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))

    peaks_permute_Bch2 = np.array(peaks_Bch2)[permuteIdx_Bch2]

    # keep the zero'th region (before the first peak) at the beginning (do no permute)
    regions_permute_Bch1 = np.append(peak_regions_Bch1[0],
                                     np.array(peak_regions_Bch1[1:len(peak_regions_Bch1)])[permuteIdx_Bch1])
    regions_permute_Bch2 = np.append(peak_regions_Bch2[0],
                                     np.array(peak_regions_Bch2[1:len(peak_regions_Bch2)])[permuteIdx_Bch2])

    zSTFT_ch1 = np.zeros((int(NFFT / 2 + 1), ySTFT_Ach1.shape[1]), dtype='complex64')
    zSTFT_ch2 = np.zeros((int(NFFT / 2 + 1), ySTFT_Ach2.shape[1]), dtype='complex64')

    # pdb.set_trace()

    if ULR == 2:
        # Process with features from A CH 2 (A right channel)
        # fill in region before first peak
        for i in range(int(peak_regions_Ach2[0])):
            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
        # permutate spectral regions & build new STFT
        zIdx = peak_regions_Ach2[0]
        for j in range(len(peaks_Ach2)):
            for k in range(int(peak_regions_Ach2[j + 1])):
                if (j % 2) == 0:
                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Ach1[:, int(peaks_Ach2[j] + k)]
                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Ach2[:, int(peaks_Ach2[j] + k)]
                else:
                    idx1 = int(min(j, numPeaksB[1] - 1))
                    idx2 = int(peaks_permute_Bch2[idx1] + k) % int(ySTFT_Bch2.shape[1] - 1)
                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Bch1[:, idx2]
                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Bch2[:, idx2]
            zIdx += peak_regions_Ach2[j + 1]

    elif ULR == 1:
        # Process with features from A CH 1 (A left channel)
        # fill in region before first peak
        for i in range(int(peak_regions_Ach1[0])):
            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
        # permutate spectral regions & build new STFT
        zIdx = peak_regions_Ach1[0]
        for j in range(len(peaks_Ach1)):
            for k in range(int(peak_regions_Ach1[j + 1])):
                if (j % 2) == 0:
                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Ach1[:, int(peaks_Ach1[j] + k)]
                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Ach2[:, int(peaks_Ach1[j] + k)]
                else:
                    idx1 = int(min(j, numPeaksB[0] - 1))
                    idx2 = int(peaks_permute_Bch1[idx1] + k) % int(ySTFT_Bch1.shape[1] - 1)
                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Bch1[:, idx2]
                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Bch2[:, idx2]
            zIdx += peak_regions_Ach1[j + 1]

    else:
        # Process with features from A, Left & Right un-coupled
        # fill in region before first peak
        for i in range(int(peak_regions_Ach1[0])):
            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
        zIdx_ch1 = peak_regions_Ach1[0]
        for j in range(len(peaks_Ach1)):
            for k in range(int(peak_regions_Ach1[j + 1])):
                if (j % 2) == 0:
                    zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_Ach1[:, int(peaks_Ach1[j] + k)]
                else:
                    idx1 = int(min(j, numPeaksB[0] - 1))
                    idx2 = int(peaks_permute_Bch1[idx1] + k) % int(ySTFT_Bch1.shape[1] - 1)
                    zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_Bch1[:, idx2]
            zIdx_ch1 += peak_regions_Ach1[j + 1]

        # fill in region before first peak
        for i in range(int(peak_regions_Ach2[0])):
            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
        zIdx_ch2 = peak_regions_Ach2[0]
        for j in range(len(peaks_Ach2)):
            for k in range(int(peak_regions_Ach2[j + 1])):
                if (j % 2) == 0:
                    zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_Ach2[:, int(peaks_Ach2[j] + k)]
                else:
                    idx1 = int(min(j, numPeaksB[1] - 1))
                    idx2 = int(peaks_permute_Bch2[idx1] + k) % int(ySTFT_Bch2.shape[1] - 1)
                    zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_Bch2[:, idx2]
            zIdx_ch2 += peak_regions_Ach2[j + 1]

    # // *-----------------------------------------------------------------* //
    # // *--- Plot region mutated Spectrogram ---*
    # STFT Visualization:

    D_ch1 = amplitude_to_db(np.abs(zSTFT_ch1), ref=np.max)
    D_ch2 = amplitude_to_db(np.abs(zSTFT_ch2), ref=np.max)

    # Linear-frequency power spectrogram

    #    fig203 = plt.figure(num=203, facecolor='silver', edgecolor='k', figsize=(12, 8))
    #
    #    specshow(D, x_axis='time', y_axis='linear', sr=48000, cmap=plt.cm.viridis)
    #    plt.colorbar(format='%+2.0f dB')
    #    #plt.imshow(Z, cmap=plt.cm.cubehelix_r)
    #    plt.title('Linear-frequency power spectrogram')

    # Log-frequency power spectrogram

    if 0:
        plt.figure(num=203, facecolor='silver', edgecolor='k', figsize=(12, 8))
        specshow(D_ch1, x_axis='time', y_axis='log', sr=fs, cmap=plt.cm.viridis)
        plt.colorbar(format='%+2.0f dB')
        plt.title('CH1: Log-frequency power spectrogram')

        plt.figure(num=204, facecolor='silver', edgecolor='k', figsize=(12, 8))
        specshow(D_ch2, x_axis='time', y_axis='log', sr=fs, cmap=plt.cm.viridis)
        plt.colorbar(format='%+2.0f dB')
        plt.title('CH2: Log-frequency power spectrogram')

        plt.show()

    # // *-----------------------------------------------------------------* //

    # Invert the stft
    yExp_ch1 = istft(zSTFT_ch1, dtype=a.dtype)
    yExp_ch2 = istft(zSTFT_ch2, dtype=a.dtype)

    ySpectraMutateMix = np.transpose(np.column_stack((yExp_ch1, yExp_ch2)))

    return ySpectraMutateMix


#    # // *-----------------------------------------------------------------* //
#    # // *--- Spectral region mutator v2 ---*    
#    
#    np.random.seed(ySeed)
#    
#    permuteIdx_Ach1 = np.random.permutation(len(peaks_Ach1))
#    permuteIdx_Ach2 = np.random.permutation(len(peaks_Ach2))
#    #permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))
#    
#    peaks_permute_Ach1 = np.array(peaks_Ach1)[permuteIdx_Ach1]
#    peaks_permute_Ach2 = np.array(peaks_Ach2)[permuteIdx_Ach2]
#    
#    
#    permuteIdx_Bch1 = np.random.permutation(len(peaks_Bch1))
#    permuteIdx_Bch2 = np.random.permutation(len(peaks_Bch2))
#    #permuteIdx = np.random.RandomState(seed=ySeed).permutation(len(peaks))
#    
#    peaks_permute_Bch1 = np.array(peaks_Bch1)[permuteIdx_Bch1]
#    peaks_permute_Bch2 = np.array(peaks_Bch2)[permuteIdx_Bch2]
#    
#    
#    # temp inspect
#    for i in range(len(peaks_permute_ch1)):
#        print('permuteIdx_ch1['+str(i)+'] = '+str(permuteIdx_ch1[i])+',  peaks_permute_ch1['+str(i)+'] = '+str(peaks_permute_ch1[i]) )
#    for i in range(len(peaks_permute_ch2)):
#        print('permuteIdx_ch2['+str(i)+'] = '+str(permuteIdx_ch2[i])+',  peaks_permute_ch2['+str(i)+'] = '+str(peaks_permute_ch2[i]) )
#        
#    # keep the zero'th region (before the first peak) at the beginning (do no permute)
#    regions_permute_Ach1 = np.append( peak_regions_Ach1[0], 
#                                 np.array(peak_regions_Ach1[1:len(peak_regions_Ach1)])[permuteIdx_Ach1] )
#    regions_permute_Ach2 = np.append( peak_regions_Ach2[0], 
#                                 np.array(peak_regions_Ach2[1:len(peak_regions_Ach2)])[permuteIdx_Ach2] )
#    
#
#    regions_permute_Bch1 = np.append( peak_regions_Bch1[0], 
#                                 np.array(peak_regions_Bch1[1:len(peak_regions_Bch1)])[permuteIdx_Bch1] )
#    regions_permute_Bch2 = np.append( peak_regions_Bch2[0], 
#                                 np.array(peak_regions_Bch2[1:len(peak_regions_Bch2)])[permuteIdx_Bch2] )    
#    
#    
#    zSTFT_ch1 = np.zeros((int(NFFT/2+1),ySTFT_Ach1.shape[1]), dtype='complex64')
#    zSTFT_ch2 = np.zeros((int(NFFT/2+1),ySTFT_Ach2.shape[1]), dtype='complex64')    
#    
#    #pdb.set_trace()
#    
#    
#    if ULR == 2:
#        # Process with features from A CH 2 (A right channel)
#        # fill in region before first peak
#        for i in range(int(regions_permute_Ach2[0])):
#            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
#            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
#        # permutate spectral regions & build new STFT
#        zIdx = regions_permute_Ach2[0]
#        for j in range(len(peaks_permute_Ach2)):
#            for k in range(int(regions_permute_Ach2[j+1])):
#                if (j%2) == 0:
#                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Ach1[:, int(peaks_permute_Ach2[j]+k)]
#                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Ach2[:, int(peaks_permute_Ach2[j]+k)]
#                else:
#                    idx1 = int(min(j,numPeaksB[1]-1))
#                    idx2 = int(peaks_permute_Bch2[idx1]+k) % int(ySTFT_Bch2.shape[1]-1)
#                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Bch1[:, idx2]
#                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Bch2[:, idx2]
#            zIdx += regions_permute_Ach2[j+1]
#            
#
#    elif ULR == 1:
#        # Process with features from A CH 1 (A left channel)
#        # fill in region before first peak
#        for i in range(int(regions_permute_Ach1[0])):
#            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
#            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
#        # permutate spectral regions & build new STFT
#        zIdx = regions_permute_Ach1[0]
#        for j in range(len(peaks_permute_Ach1)):
#            for k in range(int(regions_permute_Ach1[j+1])):
#                if (j%2) == 0:
#                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Ach1[:, int(peaks_permute_Ach1[j]+k)]
#                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Ach2[:, int(peaks_permute_Ach1[j]+k)]
#                else:
#                    idx1 = int(min(j,numPeaksB[0]-1))
#                    idx2 = int(peaks_permute_Bch1[idx1]+k) % int(ySTFT_Bch1.shape[1]-1)
#                    zSTFT_ch1[:, int(zIdx + k)] = ySTFT_Bch1[:, idx2]
#                    zSTFT_ch2[:, int(zIdx + k)] = ySTFT_Bch2[:, idx2]                    
#            zIdx += regions_permute_Ach1[j+1]        
#        
#        
#    else:
#        # Process with features from A, Left & Right un-coupled
#        # fill in region before first peak
#        for i in range(int(regions_permute_Ach1[0])):
#            zSTFT_ch1[:, i] = ySTFT_Ach1[:, i]
#        zIdx_ch1 = regions_permute_Ach1[0]
#        for j in range(len(peaks_permute_Ach1)):
#            for k in range(int(regions_permute_Ach1[j+1])):
#                if (j%2) == 0:
#                    zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_Ach1[:, int(peaks_permute_Ach1[j]+k)]
#                else:
#                    idx1 = int(min(j,numPeaksB[0]-1))
#                    idx2 = int(peaks_permute_Bch1[idx1]+k) % int(ySTFT_Bch1.shape[1]-1)
#                    zSTFT_ch1[:, int(zIdx_ch1 + k)] = ySTFT_Bch1[:, idx2]                    
#            zIdx_ch1 += regions_permute_Ach1[j+1]
#            
#        
#        # fill in region before first peak
#        for i in range(int(regions_permute_Ach2[0])):
#            zSTFT_ch2[:, i] = ySTFT_Ach2[:, i]
#        zIdx_ch2 = regions_permute_Ach2[0]
#        for j in range(len(peaks_permute_Ach2)):
#            for k in range(int(regions_permute_Ach2[j+1])):
#                if (j%2) == 0:                
#                    zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_Ach2[:, int(peaks_permute_Ach2[j]+k)]
#                else:
#                    idx1 = int(min(j,numPeaksB[1]-1))
#                    idx2 = int(peaks_permute_Bch2[idx1]+k) % int(ySTFT_Bch2.shape[1]-1)
#                    zSTFT_ch2[:, int(zIdx_ch2 + k)] = ySTFT_Bch2[:, idx2]
#            zIdx_ch2 += regions_permute_Ach2[j+1]
#
#    
#    # // *-----------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //
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

srcSel = 0

# STEREO source signal
# wavSrc = 'dsvco.wav'
# wavSrc = 'detectiveOctoSpace_one.wav'
# wavSrc = 'ebolaCallibriscian_uCCrhythm.wav'
# wavSrc = 'zoroastorian_mdychaos1.wav'
# wavSrc = 'The_Amen_Break_odmk.wav'

# MONO source signal
# wavSrc = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

# wavSrcA = 'astroman2020_vox136bpmx03.wav'
wavSrcA = '30 ambient 138 bpm.wav'
# wavSrcA = 'sebek_bt133x001.wav'

# wavSrcB = 'astroman2020_vox136bpmx04.wav'
# wavSrcB = 'astroman2020_bts136bpmx01.wav'
# wavSrcB = 'scoolreaktor_beatx03.wav'

# wavSrcB = 'astroman2020_hurdy136bpmx02.wav'
# wavSrcB = 'dsvco.wav'
wavSrcB = '30 ambient 138 bpm.wav'

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
audioSrcB = audioSrcDir + "/" + wavSrcB

[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrcA, wavLength)
print('\n// Loaded .wav file [ ' + audioSrcA + ' ]\n')

[bSrc, bNumChannels, bfs, bLength, bSamples] = load_wav(audioSrcB, wavLength)
print('\n// Loaded .wav file [ ' + audioSrcB + ' ]\n')

if aNumChannels == 2:
    aSrc_ch1 = aSrc[:, 0]
    aSrc_ch2 = aSrc[:, 1]
else:
    aSrc_ch1 = aSrc
    aSrc_ch2 = 0

if bNumChannels == 2:
    bSrc_ch1 = bSrc[:, 0]
    bSrc_ch2 = bSrc[:, 1]
else:
    bSrc_ch1 = bSrc
    bSrc_ch2 = 0

# length of input signal - '0' => length of input .wav file
print('Channel A Source Audio:')
print('aSrc Channels = ' + str(len(np.shape(aSrc))))
print('length of input signal in seconds: ----- ' + str(aLength))
print('length of input signal in samples: ----- ' + str(aSamples))
print('audio sample rate: --------------------- ' + str(afs))
print('wav file datatype: ' + str(sf.info(audioSrcA).subtype))

print('\nChannel B Source Audio:')
print('bSrc Channels = ' + str(len(np.shape(aSrc))))
print('length of input signal in seconds: ----- ' + str(bLength))
print('length of input signal in samples: ----- ' + str(bSamples))
print('audio sample rate: --------------------- ' + str(bfs))
print('wav file datatype: ' + str(sf.info(audioSrcB).subtype))

sr = afs
aT = 1.0 / sr
print('\nSystem sample rate: -------------------- ' + str(sr))
print('System sample period: ------------------ ' + str(aT))

# if sample rates are different, resample B to A's rate
# ???


# // *--- Plot - source signal ---*

if 1:
    fnum = 3
    pltTitle = 'Input Signals: aSrc_ch1'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, len(aSrc_ch1), len(aSrc_ch1))

    xodplt.xodPlot1D(fnum, aSrc_ch1, xaxis, pltTitle, pltXlabel, pltYlabel)

    plt.show()

# pdb.set_trace()

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


if 0:
    print('\n')
    print('// *---:: Phase Vocoder Time-Stretch EFX test ::---*')

    '''
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_fast = librosa.effects.time_stretch(y, 2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, 0.5) '''

    # Time Compress rate:
    R = 1.23

    # Time Expand rate:
    S = 0.3

    yRxFast_ch1 = time_stretch(aSrc_ch1, 2.0)
    yRxFast_ch2 = time_stretch(aSrc_ch2, 2.0)

    yRxFast = np.transpose(np.column_stack((yRxFast_ch1, yRxFast_ch2)))

    print('\nPerformed time_stretch by R (Rxfast)')

    # pdb.set_trace()

    ySxSlow_ch1 = time_stretch(aSrc_ch1, 0.5)
    ySxSlow_ch2 = time_stretch(aSrc_ch2, 0.5)

    ySxSlow = np.transpose(np.column_stack((ySxSlow_ch1, ySxSlow_ch2)))

    print('\nPerformed time_stretch by S (Sxslow)')

    print('\n// *---:: Write .wav files ::---*')

    outFilePath = xdir.audioOutDir + 'yOriginal.wav'
    write_wav(outFilePath, aSrc, afs)

    outFilePath = xdir.audioOutDir + 'yRxFast.wav'
    write_wav(outFilePath, yRxFast, sr)

    outFilePath = xdir.audioOutDir + 'ySxSlow.wav'
    write_wav(outFilePath, ySxSlow, sr)

    print('\n\nOutput directory: ' + xdir.audioOutDir)
    print('\nwrote .wav file yOriginal.wav')
    print('\nwrote .wav file yRxFast.wav')
    print('\nwrote .wav file ySxSlow.wav')

if 0:
    print('\n// *---:: spectraMutate test ::---*')

    ''' Experimental '''

    pmseed = 5

    # ULR : 0=uncoupled / 1=left / 2=right (default)
    ULR = 2

    ySpectraMutate = spectraMutate(aSrc, NFFT, sr, pmseed, ULR=ULR)

    #    yRxExp_ch1 = spectrExp(ySrc_ch1, NFFT, fs, pmseed)
    #    yRxExp_ch2 = spectrExp(ySrc_ch2, NFFT, fs, pmseed)
    #
    #    yRxExp = np.transpose( np.column_stack((yRxExp_ch1, yRxExp_ch2)) )

    print('\nPerformed spectral Experiment 1')

    print('\n// *---:: Write .wav files ::---*')

    outFilePath = xdir.audioOutDir + 'ySpectraMutate.wav'
    write_wav(outFilePath, ySpectraMutate, sr)

    print('\n\nOutput directory: ' + xdir.audioOutDir)
    print('\nwrote .wav file ySpectraMutate.wav')

if 1:
    print('\n// *---:: spectraMutateMix test ::---*')

    ''' Experimental '''

    wavOutNm = 'astroman2020_mx136bpm.wav'

    pmseed = 23

    # peak_pick control 
    ppkSel = 128

    # ULR : 0=uncoupled  (default) / 1=left / 2=right
    ULR = 0

    ySpectraMutate2CH = spectraMutate2CH(aSrc, bSrc, NFFT, sr, pmseed, ppkSel, ULR=ULR)

    #    yRxExp_ch1 = spectrExp(ySrc_ch1, NFFT, fs, pmseed)
    #    yRxExp_ch2 = spectrExp(ySrc_ch2, NFFT, fs, pmseed)
    #
    #    yRxExp = np.transpose( np.column_stack((yRxExp_ch1, yRxExp_ch2)) )

    print('\nPerformed spectral Experiment 1')

    print('\n// *---:: Write .wav files ::---*')

    outFilePath = xdir.audioOutDir + wavOutNm
    write_wav(outFilePath, ySpectraMutate2CH, sr)

    print('\n\nOutput directory: ' + xdir.audioOutDir)
    print('\nwrote .wav file wavOutNm.wav')

# // *---------------------------------------------------------------------* //

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //
