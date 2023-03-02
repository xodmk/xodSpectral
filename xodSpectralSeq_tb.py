# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodSpectralSeq_tb.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
# Python testbench for XODMK Spectral Sequencer
#
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


sys.path.insert(0, rootDir+'/xodUtil')
import xodPlotUtil as xodplt

sys.path.insert(2, rootDir+'/xodDSP')
import xodClocks as clks


# temp python debugger - use >>>pdb.set_trace() to set break
#import pdb


# // *---------------------------------------------------------------------* //

print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---:: XODMK Spectral Sequencer test ::---*')
print('// *--------------------------------------------------------------* //')
print('// //////////////////////////////////////////////////////////////// //')


# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
# // *--Math Functions--*
# // *---------------------------------------------------------------------* //

def cyclicZn(n):
    ''' calculates the Zn roots of unity '''
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)        # Define cyclic group Zn points
        cZn[k] = np.cos(((k)*2*np.pi)/n) + np.sin(((k)*2*np.pi)/n)*1j   # Euler's identity

    return cZn



# // *---------------------------------------------------------------------* //
# // *--Primary parameters--*
# // *---------------------------------------------------------------------* //

# audio sample rate:
fs = 48000.0

# sample period
T = 1.0 / fs

# video frames per second:
framesPerSec = 30.0

bpm = 133.0

# time signature: 4 = 4/4; 3 = 3/4
timeSig = 4

# calculate length of 1　bar in seconds:
spb = 60.0 / bpm
secondsPerBar = timeSig * spb

# length of x in seconds:
xLength = secondsPerBar

# plot clock signals
xPlots = 1;


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Instantiate clock & signal Generator objects::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //

tbClocks = clks.xodClocks(xLength, fs, bpm, framesPerSec)


tbxLength = tbClocks.xLength
tbFs = tbClocks.fs
tbBpm = tbClocks.bpm
tbframesPerSec = tbClocks.framesPerSec

# *---Audio secondary Parameters---*

# set bps - beats per second
tbBps = tbClocks.bps
# set spb - Seconds per beat
tbSpb = tbClocks.spb
# set samplesPerBeat
tbsamplesPerBeat = tbClocks.samplesPerBeat
# set samples per bar / 1 bar = tsig beats (ex. 4/4: 4*samplesPerBeat)
tbsamplesPerBar = tbClocks.samplesPerBar

# set totalSamples - Total audio samples in x
tbtotalSamples = tbClocks.totalSamples
# set totalBeats - total beats in x
tbtotalBeats = tbClocks.totalBeats


# *---Video secondary Parameters---*

# set samplesPerFrame - Num audio samples per video frame
tbsamplesPerFrame = tbClocks.samplesPerFrame
# set framesPerBeat - Num video frames per beat
tbframesPerBeat = tbClocks.framesPerBeat

# set totalFrames - Total video frames in x
TBtotalFrames = tbClocks.totalFrames


tbclkDownBeats = tbClocks.clkDownBeats()

tbclkDownBeatsGnr = np.array([y for y in tbClocks.clkDownBeatsGnr()])

#alt
#tbclkDownBeatsGnr = np.zeros([tbxLength, 1])
#for y in range(tbxLength):
#    tbclkDownBeatsGnr[y] = tbClocks.clkDownBeatsGnr()


tbclkDownFrames = tbClocks.clkDownFrames()

tbclkQtrBeat = tbClocks.clkQtrBeat()

nbar = 3
tbclkQtrBeatBar = tbClocks.clkQtrBeatBar(nbar)

n = 7
tbclkDivNBeat = tbClocks.clkDivNBeat(n)

Xnote = 8
tbclkXbeat = tbClocks.clkXBeat(Xnote)


Xnote2 = 6
pulseWidth = int(tbClocks.samplesPerBeat/3)
tbclkXpulse = tbClocks.clkXPulse(Xnote2, pulseWidth)


print('\nCreated odmkClocks object "tbClock" with the following parameters:')
print('\nAn odmkClocks object has been instanced with:')
print('xLength = '+str(tbxLength))
print('fs = '+str(tbFs))
print('bpm = '+str(tbBpm))
print('framesPerSec = '+str(tbframesPerSec))
print('beatsPerSecond = '+str(tbBps))
print('secondsPerBeat = '+str(tbSpb))
print('samplesPerBeat = '+str(tbsamplesPerBeat))
print('samplesPerBar = '+str(tbsamplesPerBar))
print('totalSamples = '+str(tbtotalSamples))
print('totalBeats = '+str(tbtotalBeats))
print('samplesPerFrame = '+str(tbsamplesPerFrame))
print('framesPerBeat = '+str(tbframesPerBeat))
print('totalFrames = '+str(TBtotalFrames))


# // *---------------------------------------------------------------------* //

if xPlots == 1:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Plotting::---*')
    print('// *--------------------------------------------------------------* //')

    # // *---------------------------------------------------------------------* //
    # // *---tbclkDivNBeat---*
    # // *---------------------------------------------------------------------* //
    
    # define a sub-range for wave plot visibility
    tLen = int(tbtotalSamples)
    
    fnum = 1
    pltTitle = 'Input Signal tbclkXbeat (first '+str(tLen)+' samples)'
    pltXlabel = 'tbclkXbeat: '+str(n)+' beats per bar'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)
    
    xodplt.xodPlot1D(fnum, tbclkXbeat[0:tLen], xaxis, pltTitle, pltXlabel, pltYlabel)
    
   
    # define a sub-range for wave plot visibility
    tLen = int(tbtotalSamples)
    
    fnum = 2
    pltTitle = 'Input Signal tbclkPulse (first '+str(tLen)+' samples)'
    pltXlabel = 'tbclkPulse: '+str(n)+' beats per bar'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)
    
    xodplt.xodPlot1D(fnum, .97*tbclkXpulse[0:tLen], xaxis, pltTitle, pltXlabel, pltYlabel)   
   
    
#    # // *---------------------------------------------------------------------* //
#    # // *---Multi Plot - source signal array vs. FFT MAG out array---*
#    # // *---------------------------------------------------------------------* //
#
#    fnum = 3
#    pltTitle = 'Input Signals: sinArray (first '+str(tLen)+' samples)'
#    pltXlabel = 'sinArray time-domain wav'
#    pltYlabel = 'Magnitude'
#    
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xaxis = np.linspace(0, tLen, tLen)
#    
#    odmkMultiPlot1D(fnum, sinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMp='gist_stern')
#    
#    
#    fnum = 4
#    pltTitle = 'FFT Mag: yScaleArray multi-osc '
#    pltXlabel = 'Frequency: 0 - '+str(fs / 2)+' Hz'
#    pltYlabel = 'Magnitude (scaled by 2/N)'
#    
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#    
#    odmkMultiPlot1D(fnum, yScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMp='gist_stern')
#    
#    
#    # // *---------------------------------------------------------------------* //
#    # // *---Orthogonal Sine Plot - source signal array vs. FFT MAG out array---*
#    # // *---------------------------------------------------------------------* //
#    
#    fnum = 5
#    pltTitle = 'Input Signals: orthoSinArray (first '+str(tLen)+' samples)'
#    pltXlabel = 'orthoSinArray time-domain wav'
#    pltYlabel = 'Magnitude'
#    
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xaxis = np.linspace(0, tLen, tLen)
#    
#    odmkMultiPlot1D(fnum, orthoSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMp='hsv')
#    
#    
#    fnum = 6
#    pltTitle = 'FFT Mag: yOrthoScaleArray multi-osc '
#    pltXlabel = 'Frequency: 0 - '+str(fs / 2)+' Hz'
#    pltYlabel = 'Magnitude (scaled by 2/N)'
#    
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#    
#    odmkMultiPlot1D(fnum, yOrthoScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMp='hsv')
#    
#    # // *-----------------------------------------------------------------* //
#        
#    
#    # define a sub-range for wave plot visibility
#    tLen = 500
#    
#    fnum = 7
#    pltTitle = 'Input Signal tri2_5K (first '+str(tLen)+' samples)'
#    pltXlabel = 'tri2_5K: '+str(testFreq1)+' Hz'
#    pltYlabel = 'Magnitude'
#    
#    sig = tri2_5K[0:tLen]
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xaxis = np.linspace(0, tLen, tLen)
#    
#    odmkPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)
#    
#    # // *-----------------------------------------------------------------* //    
#    
#    fnum = 8
#    pltTitle = 'Scipy FFT Mag: y1_FFTscale '+str(testFreq1)+' Hz'
#    pltXlabel = 'Frequency: 0 - '+str(fs / 2)+' Hz'
#    pltYlabel = 'Magnitude (scaled by 2/N)'
#    
#    # sig <= direct
#    
#    # define a linear space from 0 to 1/2 Fs for x-axis:
#    xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#    
#    odmkPlot1D(fnum, y3tri_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)    
#
#    # // *-----------------------------------------------------------------* //

else:    # comment-off/on: toggle plots below
    print('\n')  
    print('// *---::No Plotting / Debugging::---*')    
    
    plt.show()

# // *---------------------------------------------------------------------* //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //


#tLen = 200
#
## Input signal
#fig1 = plt.figure(num=1, facecolor='silver', edgecolor='k')
#odmkSrcplt1 = plt.plot(y[0:tLen])
#plt.setp(odmkSrcplt1, color='red', ls='-', linewidth=1.00)
#plt.xlabel('monosin5K: '+str(testFreq)+' Hz')
#plt.ylabel('Magnitude')
#plt.title('Input Signal (first '+str(tLen)+' samples)')
#plt.grid(color='c', linestyle=':', linewidth=.5)
#plt.grid(True)
## plt.xticks(np.linspace(0, Fs/2, 10))
#ax = plt.gca()
#ax.set_axis_bgcolor('black')
#
## define a linear space from 0 to 1/2 Fs for x-axis:
#xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
## FFT Magnitude out plot (0-fs/2)
#fig2 = plt.figure(num=2, facecolor='silver', edgecolor='k')
#odmkFFTplt1 = plt.plot(xfnyq, yfscale)
#plt.setp(odmkFFTplt1, color='red', ls='-', linewidth=1.00)
#plt.xlabel('Frequency: 0 - '+str(fs / 2)+' Hz')
#plt.ylabel('Magnitude (scaled by 2/N)')
#plt.title('Scipy FFT: Fs = '+str(fs)+' N = '+str(N))
#plt.grid(color='c', linestyle=':', linewidth=.5)
#plt.grid(True)
## plt.xticks(np.linspace(0, Fs/2, 10))
#ax = plt.gca()
#ax.set_axis_bgcolor('black')
