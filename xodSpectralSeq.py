# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodSpectralSeq.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
# Python ODMK timing sequencer module
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


sys.path.insert(0, rootDir+'/xodma')
from xodmaAudioTools import load_wav, write_wav, valid_audio, resample
from xodmaAudioTools import samples_to_time, time_to_samples, fix_length
from xodmaSpectralTools import amplitude_to_db, stft, istft, magphase, peak_pick
from xodmaOnset import onset_strength
from xodmaVocoder import pvTimeStretch, pvPitchShift
from xodmaSpectralUtil import frames_to_time
from xodmaSpectralPlot import specshow

# from odmkSpectralTools import phase_vocoder, magphase


sys.path.insert(1, rootDir+'/xodDSP')
import xodClocks as clks
import xodWavGen as wavGen


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *********************************************************************** //
# // *---util functions
# // *********************************************************************** //


def cyclicZn(n):
    ''' calculates the Zn roots of unity '''
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)                         # Define cyclic group Zn points
        cZn[k] = np.cos(((k)*2*np.pi)/n) + np.sin(((k)*2*np.pi)/n)*1j    # Euler's identity
    return cZn


def randomIdx(n, k):
    '''for an array of k elements, returns a list of random indexes
       of length n (n integers rangin from 0:k-1)'''
    randIdx = []
    for i in range(n):
        randIdx.append(round(random.random()*(k-1)))
    return randIdx


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

class xodSpectralSeq:
    ''' odmk audio/video clocking modules 
        usage: myOdmkClks = odmkClocks(outLength, fs, bpm, framesPerSec, tsig)
        xLength => defines length of seq in seconds (total track length)
        fs => audio sample rate
        bpm => bpm
        framesPerSec => video frames per second
        tsig => time signature: currently = number of quarters per bar (defaults to 4/4)
        ex: xDownFrames = myOdmkClks.clkDownFrames() ->
            returns an np.array of 1's at 1st frame of downbeat, 0's elsewhere
    '''

    def __init__(self, xLength, fs, bpm, framesPerSec, tsig=4):

        # *---set primary parameters from inputs---*

        self.xLength = xLength
        self.fs = fs
        self.bpm = bpm
        self.framesPerSec = framesPerSec
        self.tsig = tsig        
        
        print('\nAn odmkClocks object has been instanced with:')
        print('xLength = '+str(self.xLength)+', fs = '+str(fs)+', bpm = '+str(bpm)+', framesPerSec = '+str(framesPerSec))

        # *---Define Audio secondary Parameters---*

        # set bps - beats per second
        self.bps = bpm / 60
        # set spb - Seconds per beat
        self.spb = 60.0 / bpm
        # set samplesPerBeat
        self.samplesPerBeat = fs * self.spb
        # set samples per bar / 1 bar = tsig beats (ex. 4/4: 4*samplesPerBeat)
        self.samplesPerBar = tsig * self.samplesPerBeat

        # set totalSamples - Total audio samples in x
        self.totalSamples = int(np.ceil(xLength * fs))
        # set totalBeats - total beats in x
        self.totalBeats = self.totalSamples / self.samplesPerBeat

        # *---Define Video secondary Parameters---*

        # set samplesPerFrame - Num audio samples per video frame
        self.samplesPerFrame = framesPerSec * fs
        # set framesPerBeat - Num video frames per beat
        self.framesPerBeat = self.spb * framesPerSec

        # set totalFrames - Total video frames in x
        self.totalFrames = int(np.ceil(xLength * framesPerSec))


    # // ******************************************************************* //
    # // *---sequence generators
    # // ******************************************************************* //

    # // *-----------------------------------------------------------------* //
    # // *---gen downbeat sequence
    # // *-----------------------------------------------------------------* //

    def clkDownBeats(self):
        ''' generates an output array of 1s at downbeat, 0s elsewhere '''

        xClockDown = np.zeros([self.totalSamples])
        for i in range(self.totalSamples):
            if i % np.ceil(self.samplesPerBeat) == 0:
                xClockDown[i] = 1
            else:
                xClockDown[i] = 0
        return xClockDown
    
    # Python Generator
    def clkDownBeatsGnr(self):
        ''' generates an output array of 1s at downbeat, 0s elsewhere '''
        for i in range(self.totalSamples):
            if i % np.ceil(self.samplesPerBeat) == 0:
                yield 1
            else:
                yield 0

    def clkDownFrames(self):
        ''' generates 1s at frames corresponding to
            downbeats, 0s elsewhere '''

        xFramesDown = np.zeros([self.totalSamples])
        for i in range(self.totalSamples):
            if i % np.ceil(self.framesPerBeat) == 0:
                xFramesDown[i] = 1
            else:
                xFramesDown[i] = 0
        return xFramesDown
    
    def clkDownFramesGnr(self):
        ''' generates 1s at frames corresponding to
            downbeats, 0s elsewhere '''

        for i in range(self.totalSamples):
            if i % np.ceil(self.framesPerBeat) == 0:
                yield 1
            else:
                yield 0

    # // *-----------------------------------------------------------------* //
    # // *---gen note sequence (xLength samples)
    # // *-----------------------------------------------------------------* //

    def clkQtrBeat(self):
        ''' Output a 1 at Qtr downbeat for xLength samples '''

        # set samplesPerBeat
        samplesPerQtr = self.samplesPerBeat    # assume 1Qtr = 1Beat

        xQtrBeat = np.zeros([self.totalSamples])
        for i in range(self.totalSamples):
            if i % np.ceil(samplesPerQtr) == 0:
                xQtrBeat[i] = 1
            else:
                xQtrBeat[i] = 0
        return xQtrBeat
    
    def clkQtrBeatGnr(self):
        ''' Output a 1 at Qtr downbeat for xLength samples '''

        # set samplesPerBeat
        samplesPerQtr = self.samplesPerBeat    # assume 1Qtr = 1Beat

        for i in range(self.totalSamples):
            if i % np.ceil(samplesPerQtr) == 0:
                yield 1
            else:
                yield 0
        

    # // *-----------------------------------------------------------------* //
    # // *---gen X note sequence (xLength samples)
    # // *-----------------------------------------------------------------* //

    def clkXBeat(self, Xnote):
        ''' Output a Xnote downbeat for xLength samples 
            Xnote: 1=whole, 2=half, 3=third, 4=quarter, 5=fifth...'''

        XBeat = np.zeros([self.totalSamples])
        XnoteInv = 1/Xnote
        Xsamples = XnoteInv*self.samplesPerBar
        for i in range(self.totalSamples):
            if i % np.ceil(Xsamples) == 0:
                XBeat[i] = 1
            else:
                XBeat[i] = 0
        return XBeat
        
    def clkXBeatGnr(self, Xnote):
        ''' Output a Xnote downbeat for xLength samples 
            Xnote: 1=whole, 2=half, 3=third, 4=quarter, 5=fifth...'''

        XnoteInv = 1/Xnote
        Xsamples = XnoteInv*self.samplesPerBar
        for i in range(self.totalSamples):
            if i % np.ceil(Xsamples) == 0:
                yield 1
            else:
                yield 0
        
    # // *-----------------------------------------------------------------* //
    # // *---gen note sequence (xLength samples)
    # // *-----------------------------------------------------------------* //

    def clkXPulse(self, Xnote, pulseWidth):
        ''' Output a Xnote downbeat for xLength samples 
            Xnote: 1=whole, 2=half, 3=third, 4=quarter, 5=fifth...
            pulseWidth = number of samples per pulse'''

        XPulse = np.zeros([self.totalSamples])
        XnoteInv = 1/Xnote
        Xsamples = int(np.ceil(XnoteInv*self.samplesPerBar))
        for i in range(self.totalSamples):
            if i % Xsamples == 0:
                pulseCnt = pulseWidth
            if pulseCnt > 0:
                XPulse[i] = 1
                pulseCnt -= 1
            else:
                XPulse[i] = 0
        return XPulse
    
    def clkXPulseGnr(self, Xnote, pulseWidth):
        ''' Output a Xnote downbeat for xLength samples 
            Xnote: 1=whole, 2=half, 3=third, 4=quarter, 5=fifth...
            pulseWidth = number of samples per pulse'''

        XnoteInv = 1/Xnote
        Xsamples = int(np.ceil(XnoteInv*self.samplesPerBar))
        for i in range(self.totalSamples):
            if i % Xsamples == 0:
                pulseCnt = pulseWidth
            if pulseCnt > 0:
                pulseCnt -= 1
                yield 1
            else:
                yield 0


    # // *-----------------------------------------------------------------* //
    # // *---gen note sequence (nBar # of bars)
    # // *-----------------------------------------------------------------* //

    def clkQtrBeatBar(self, nBar=1):
        ''' Output a 1 at Qtr downbeat for 'nBar' bars (4/4, 4 qtr notes)
            optional nBar parameter: default nBar = 1 bar '''

        numSamples = int(np.ceil(nBar * self.samplesPerBar))
        xQtrBar = np.zeros([numSamples])
        for i in range(numSamples):
            if i % np.ceil(self.samplesPerBeat) == 0:
                xQtrBar[i] = 1
            else:
                xQtrBar[i] = 0
        return xQtrBar

    def clkQtrBeatBarGnr(self, nBar=1):
        ''' Output a 1 at Qtr downbeat for 'nBar' bars (4/4, 4 qtr notes)
            optional nBar parameter: default nBar = 1 bar '''

        numSamples = int(np.ceil(nBar * self.samplesPerBar))
        for i in range(numSamples):
            if i % np.ceil(self.samplesPerBeat) == 0:
                yield 1
            else:
                yield 0

    # // *-----------------------------------------------------------------* //
    # // *---gen note sequence (xLength samples)
    # // *-----------------------------------------------------------------* //

#    def clkOne3Beat(self):
#        ''' Output a 1 - 3 divisions per bar for xLength samples '''
#
#        # set samplesPerBeat
#        # samplesPerBar = self.samplesPerBar    # assume 1Qtr = 1Beat
#        xDiv3Beat = np.ceil(self.samplesPerBar / 3)
#
#
#        xDiv3 = np.zeros([self.totalSamples, 1])
#        for i in range(self.totalSamples):
#            if i % xDiv3 == 0:
#                xDiv3Beat[i] = 1
#            else:
#                xDiv3Beat[i] = 0
#        return xDiv3Beat


    def clkDivNBeat(self, n):
        ''' Output a pulse every bar/n samples for xLength samples '''

        # set samplesPerBeat
        # samplesPerBar = self.samplesPerBar    # assume 1Qtr = 1Beat
        clkDivN = np.ceil(self.samplesPerBar / n)

        clkDivNBeat = np.zeros([self.totalSamples])
        for i in range(self.totalSamples):
            if i % clkDivN == 0:
                clkDivNBeat[i] = 1
            else:
                clkDivNBeat[i] = 0
        return clkDivNBeat
    
    def clkDivNBeatGnr(self, n):
        ''' Output a pulse every bar/n samples for xLength samples '''

        # set samplesPerBeat
        # samplesPerBar = self.samplesPerBar    # assume 1Qtr = 1Beat
        clkDivN = np.ceil(self.samplesPerBar / n)

        for i in range(self.totalSamples):
            if i % clkDivN == 0:
                yield 1
            else:
                yield 0

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
