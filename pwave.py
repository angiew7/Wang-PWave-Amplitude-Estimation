import csv
import os
import pathlib
import re
import wfdb
import random
import matplotlib as mpl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats

from scipy.io import loadmat
from matplotlib.lines import Line2D
from scipy.linalg import lstsq
#from scipy.signal import filtfilt, butter, convolve, find_peaks
from scipy.stats import pearsonr


#from ecgdetectors import Detectors
def get_figure(n_rows=None,n_columns=None):
    if n_rows is None:
        n_rows = 1
    if n_columns is None:
        n_columns = 1
    
    golden_ratio   = 1.618
    figure_width   = 4.0
    figure_height  = (figure_width/golden_ratio)*(n_rows/n_columns)
    figure = plt.figure(figsize=(figure_width,figure_height))
    return figure
def draw_array_axes(n_rows, n_columns, i_row, i_column, 
                  axis_boundaries=[0.02,0.02,0.17,0.15], # Top, Right, Bottom, Left
                  horizontal_buffer=0.02, 
                  vertical_buffer=0.03,
                  sharex=None,
                  figure=None):
    if not figure:
        figure = plt.gcf()
    topArrayEdge    = axis_boundaries[0]
    rightArrayEdge  = axis_boundaries[1]
    bottomArrayEdge = axis_boundaries[2]
    leftArrayEdge   = axis_boundaries[3]
    axesWidth       = (1-leftArrayEdge-rightArrayEdge -(n_columns-1)*horizontal_buffer)/n_columns
    axesHeight      = (1-topArrayEdge -bottomArrayEdge-(n_rows   -1)*vertical_buffer  )/n_rows 
    leftEdge        = leftArrayEdge  +(i_column -1)*(axesWidth  + horizontal_buffer)
    bottomEdge      = bottomArrayEdge+(n_rows-i_row)*(axesHeight + vertical_buffer  )

    axes = figure.add_axes([leftEdge,bottomEdge,axesWidth,axesHeight],sharex=sharex)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')

    return (axes)
def format_figure_text(figure=None, fontSize=8, fontName='Times New Roman'):  
    if not figure:
        figure = plt.gcf()
    textHandles = [h for h in figure.findobj() if type(h) == mpl.text.Text]
    for th in textHandles:
        if fontName:
            th.set_fontname(fontName)
        if fontSize:
            th.set_fontsize(fontSize)
def get_subject_from_mat_filename(mat_filename):
    filename_base = os.path.splitext(os.path.basename(mat_filename))[0]
    if filename_base.endswith('m'):
        filename_base = filename_base[:-1]
    else:
        assert(False) # This shouldn't happen  
    return filename_base
def get_subject_list(data_directory):
    mat_directory = os.path.join(data_directory,'mat')
    mat_file_list = os.listdir(mat_directory)
    mat_file_list.sort()
    subject_list = [get_subject_from_mat_filename(mat_file) for mat_file in mat_file_list]    
    subject_list.sort()
    return subject_list
def read_annotations(subject,data_directory):
    annotation_path = os.path.join(data_directory,'annotations','qt',subject) # Replace with correct file path
    annotations = wfdb.rdann(annotation_path, extension='q1c')   
    return annotations
def get_sample_rate(data_directory,subject=None):
    if subject is None:
        subject_list = get_subject_list(data_directory)
        subject = subject_list[0]   
    annotations = read_annotations(subject,data_directory)
    return annotations.fs
def read_ecg(subject,annotations,data_directory):
    mat_filepath = os.path.join(data_directory,'mat',subject) +'m.mat'
    ecg                    = dict()
    val                    = loadmat(mat_filepath)['val'].astype(np.float64)
    # for i_channel in range(val.shape[0]):
    #     val[i_channel] -= np.mean(val[i_channel]) 
    #     val[i_channel] /= np.std(val[i_channel])
    ecg['raw']             = val[0]  # Channel 0
    ecg['raw_all']         = val # All channels
    ecg['raw_sample_rate'] = annotations.fs
    return ecg
def smooth_binomial(x,duration,sample_rate,figures_directory=None):
    # Use the binomial as a finite-domain approximation to the Gaussian
    # as the impulse response for an FIR smoothing filter. 
    # If figures_directory is specified, then a figure is generated and saved.
    # Otherwise, the filtered signal is returned.
    
    # Ensure that the number of taps is odd. If needed, add 1 to make it odd
    n_taps = int(duration*sample_rate)
    if n_taps%2==0:
        n_taps += 1
    n_taps = int(n_taps)
    
    b = stats.binom.pmf(np.arange(n_taps),n_taps-1,0.5)
    if figures_directory is not None:
        plot_filter_design(sample_rate,'Smoothing_Filter',figures_directory) 
        return None
    assert(len(b)==n_taps)
    
    # Append samples to the beginning and end of x so that the filter is centered
    # and there is no phase delay.
    n_pad = int((n_taps-1)/2)
    if len(x.shape)==1:
        x_padded = np.concatenate((x,x[-1]*np.ones(n_pad)))
    elif len(x.shape)==2:
        n_channels = x.shape[0]
        x_padded = np.concatenate((x,x[:,-1].reshape(n_channels,1)@np.ones((1,n_pad))),axis=1)
    x_smooth = signal.lfilter(b,1,x_padded)
    if len(x.shape)==1:
        x_truncated = x_smooth[n_pad:]
    elif len(x.shape)==2:
        x_truncated = x_smooth[:,n_pad:] 
    assert(len(x_truncated)==len(x))
    return x_truncated   
def plot_filter_design(b,a,sample_rate,figures_directory):
    # Calculate the magnitude response of the filter from a,b
    f, h = signal.freqz(b, a, fs=sample_rate, worN=2**12)
    h_abs = np.abs(h)**2 # Squared because the filter is applied forwards and backwards

    figure = get_figure()
    axes = draw_array_axes(1,1,1,1,figure=figure)    
    axes.axhline(1.0,color='k',linewidth=0.5,linestyle=':')
    axes.plot(f,h_abs,linewidth=1.5,color='tab:red')
    axes.set_xlim([f[0],50]) # Note: this is specific to this filter
    axes.set_ylim([0.0,1.1])
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Magnitude')
    format_figure_text(figure)    
    figure_path = os.path.join(figures_directory,'ecg_filter_magnitude_response.pdf')        
    figure.savefig(figure_path) 
    
    # Calculate the impulse response of the filter from a,b
    n      = np.arange(-100,101)
    t      = n/sample_rate
    x      = np.zeros_like(n)
    x[n==0] = 1.0    
    y = signal.filtfilt(b,a,x)
    
    figure = get_figure()
    axes = draw_array_axes(1,1,1,1,figure=figure)
    axes.plot(t,y,linewidth=1.5,color='tab:blue')
    axes.set_xlim([t[0],t[-1]])
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Impulse Response')    
    format_figure_text(figure)        
    figure_path = os.path.join(figures_directory,'ecg_filter_impulse_response.pdf')        
    figure.savefig(figure_path) 
def ecg_filter_design(sample_rate,figures_directory=None):
    cutoff_frequencies = [0.3, 30]
    filter_order       = 4

    cutoffFrequenciesNormalized = np.array(cutoff_frequencies) / (sample_rate / 2)
    b, a = signal.butter(filter_order,cutoffFrequenciesNormalized, btype='band')
    
    if figures_directory is not None:   
        plot_filter_design(b,a,sample_rate,figures_directory)        
        
    return b,a
def filter_ecg(ecg):
    # User-specified Parameters    
    # cutoff_frequencies = [4.0, 20] # Units of Hz
    # filter_order        = 4 # The filter order is really 2x this since filtfilt is used
    
    # # Integrity checks
    # sample_rate = ecg['raw_sample_rate']
    # assert(cutoff_frequencies[0]<cutoff_frequencies[1])
    # assert(cutoff_frequencies[1]<sample_rate/2)
    
    # # Design a 4th order Butterworth Bandpass filter with the specified cutoff frequencies
    # cutoffFrequenciesNormalized = np.array(cutoff_frequencies) / (ecg['raw_sample_rate'] / 2)
    # b, a = signal.butter(filter_order,cutoffFrequenciesNormalized, btype='band')

    # Filter used by Balaji for annotations
    upsample_rate       = 1
    
    sample_rate = ecg['raw_sample_rate']
    b,a = ecg_filter_design(sample_rate)    
    
    x     = signal.filtfilt(b, a, ecg['raw'    ]       )
    x_all = signal.filtfilt(b, a, ecg['raw_all'],axis=1)

    # Filter Tuned by James
    if False:
        # x     = signal.filtfilt(b, a, ecg['raw'])
        # x_all = signal.filtfilt(b, a, ecg['raw_all'],axis=1)
        
        d1 = 0.04 # Duration of the binomial impulse response in units of seconds
        d2 = 2.50 # Duration of the baseline impulse response in units of seconds

        x1 = smooth_binomial(ecg['raw'],d1, sample_rate, figures_directory=figures_directory) # Smoothed (lowpass filter)
        x2 = smooth_binomial(ecg['raw'],d2, sample_rate, figures_directory=figures_directory) # Baseline (subtract from x to highpass)
        x  = x1-x2    
        #x     = smooth_binomial(ecg['raw'    ],5,sample_rate,figures_directory=figures_directory)
        
        x1_all = smooth_binomial(ecg['raw_all'],d1, sample_rate, figures_directory=figures_directory) # Smoothed (lowpass filter)
        x2_all = smooth_binomial(ecg['raw_all'],d2, sample_rate, figures_directory=figures_directory) # Baseline (subtract from x to highpass)
        x_all = x1_all-x2_all 
        # x_all = smooth_binomial(ecg['raw_all'],5,sample_rate,figures_directory=figures_directory)

    # The order of this filter is actually 2*filterOrder because filtfilt filters it once forward and once backward
    if upsample_rate>1:
        x     = signal.resample_poly(x,1,upsample_rate)
        x_all = signal.resample_poly(x_all,1,upsample_rate,axis=1)
    ecg['filtered'            ] = x
    ecg['filtered_all'        ] = x_all
    ecg['filtered_sample_rate'] = ecg['raw_sample_rate']*upsample_rate
    ecg['upsampleRate'        ] = upsample_rate

    return ecg   
def get_p_waves_annotations(subject,data_directory):
    annotations  = read_annotations(subject,data_directory)
    p_waves = get_p_wave_annotations(annotations)
    return p_waves
def get_p_wave_annotations(annotations):
    p_wave = dict()
    p_waves = list()

    annotation_indices = annotations.sample
    annotation_symbols = annotations.symbol
    
    for i,symbol in enumerate(annotation_symbols):
        if i==0 or i==len(annotation_symbols)-1:
            continue
        if symbol=='p' and annotation_symbols[i-1]=='(' and annotation_symbols[i+1]==')':
            # Make sure there is a corresponding R wave with a symbol of 'N', else skip
            j = i+2
            while j<len(annotation_symbols) and annotation_symbols[j]!='N':
                j+=1
            if j==len(annotation_symbols):
                continue
            i_r_previous = None
            if j>=8: # As long as this isn't the first R wave annotation
                i_r_previous = annotation_indices[j-8]
            elif (j+8)<len(annotation_symbols): 
                # As long as this isn't the last R wave annotation, just the following R wave
                # to estimate the previous R wave index
                rri = annotation_indices[j+8]-annotation_indices[j]
                i_r_previous = annotation_indices[j] - rri
            assert(i_r_previous is not None)
                            
            p_wave['i_p_start'   ] = annotation_indices[i-1]
            p_wave['i_p_peak'    ] = annotation_indices[i  ]
            p_wave['i_p_end'     ] = annotation_indices[i+1]
            p_wave['i_r'         ] = annotation_indices[j  ]
            p_wave['i_r_previous'] = i_r_previous
            p_waves.append(p_wave.copy())
    return p_waves
def get_valid_subject_list(subjects,data_directory):
    subjects_list_new = list()
    for subject in subjects:
        annotations = read_annotations(subject,data_directory)        
        p_waves     = get_p_wave_annotations(annotations)  
        if len(p_waves)<10:
            continue
        subjects_list_new.append(subject)
    return subjects_list_new        
def find_first_right_maximum(x):
    # Find the first maximum in x starting at the right side of x
    # and return the index of this maximum
   
    i = len(x)-2 # Set i to the second-to-last last index in ix
    found_maximum = False
    while i>0:
        if x[i]>x[i-1] and x[i]>x[i+1]:
            found_maximum = True
            break
        i -= 1 
    if not found_maximum:
        return None
    return i
def find_nearest_maximum(x,ix):
    assert(ix>0 and ix<len(x)-1)
    
    # Search Left
    il = ix
    while il>0 and x[il-1]>x[il]:        
        il -= 1

    # Search Right
    ir = ix            
    while ir<len(x)-1 and x[ir+1]>x[ir]:
        ir += 1
        
    # Find the maximum over the full search range
    imax = max(ix-il,ir-ix)
    i0 = max(0       ,ix-imax)
    i1 = min(len(x)-1,ix+imax) 
           
    ipeak = np.argmax(x[i0:i1+1])+i0
                        
    return ipeak
def find_p_wave_peak(subject,data_directory):
    # User-Specified Parameters
    t_start_p_wave = -0.40 # Units of seconds relative to the R wave
    t_end_p_wave   = -0.08 # Units of seconds relative to the R wave
        
    # Read data
    annotations  = read_annotations(subject,data_directory)
    ecg          = read_ecg(subject,annotations,data_directory)
    ecg_filtered = filter_ecg(ecg)
    p_waves      = get_p_wave_annotations(annotations)    
    
    assert(len(p_waves)>=10)
    
    # Preprocessing
    sample_rate      = ecg['filtered_sample_rate']
    i_start_p_wave_offset = int(t_start_p_wave*sample_rate) # Offset in samples relative to the R wave index
    i_end_p_wave_offset   = int(t_end_p_wave  *sample_rate) # Offset in samples relative to the R wave index    
    
    i_offset_p_wave = np.arange(i_start_p_wave_offset,i_end_p_wave_offset+1)
        
    n_channels = ecg_filtered['filtered_all'].shape[0]    
    
    i_channel = 0
    x = ecg_filtered['filtered_all'][i_channel]                

    average_p_wave = np.zeros(len(i_offset_p_wave))

    for p_wave in p_waves[:10]: # Just the first 10, to match the other annotations
        i_r_wave = p_wave['i_r']      
        segment_p_wave = x[i_r_wave + i_offset_p_wave]
        average_p_wave  += segment_p_wave
    average_p_wave  /= len(p_waves)
    
    ixp = find_first_right_maximum(average_p_wave)
    i_p_wave_peak = i_offset_p_wave[ixp]
        
    t_p_peak = i_p_wave_peak/sample_rate        
    return i_p_wave_peak,t_p_peak
class BumpModel:
    def __init__(self,sample_rate):
        self.t_width          =  0.15  # Width of bump in seconds
        self.t_center         = -0.17  # Duration of p wave peak  relative to R wave      in seconds
        self.t_baseline       =  0.05  # Duration of baseline stub prior to start of bump 
        self.t_segment_start  = -0.33 # Time of the start of the segment relative to the R wave
        self.t_segment_end    = -0.08 # Time of the end of the segment relative to the R wave
        self.interval_portion =  0.6 # Portion of the RR interval to use for the segment being modeled
    def bump(self,tx):
        u = (tx-self.t_center)/(self.t_width/2)
        b = np.zeros_like(tx)
        m = np.abs(u)<=1 # Mask
        b[m] = (1-np.abs(u[m])**2)**2
        return b    
    def get_center_prior(self,tc=None):
        prior_center = -0.20
        prior_sigma  =  0.04
        uniform_delta = 0.07
        
        if tc is None:
            tc = np.arange(-0.28,-0.095,0.005)  
                 
        #prior = np.exp(-(tc-prior_center)**2/(2*prior_sigma**2))
        #i_uniform = np.logical_and(tc>=prior_center-uniform_delta,tc<=prior_center+uniform_delta)
        #prior[i_uniform] = prior[i_uniform].min()

        tk    = np.array([-0.30,-0.25,-0.12,-0.10])
        
        prior = np.zeros_like(tc)
        m1    = np.logical_and(tc>tk[0],tc<=tk[1]) # Region 1 (ramp up from 0 to 1)
        m2    = np.logical_and(tc>tk[1],tc<=tk[2]) # Region 2 (flat at 1)
        m3    = np.logical_and(tc>tk[2],tc<=tk[3]) # Region 3 (ramp down from 1 to 0)
        
        prior[m1] = (tc[m1]-tk[0])/(tk[1]-tk[0])
        prior[m2] = 1.0
        prior[m3] = 1.0-(tc[m3]-tk[2])/(tk[3]-tk[2])
                
        return tc,prior     
    def get_sigma_prior(self,ts=None):
        #prior_center =  0.07
        
        prior_mean  = 0.100
        prior_sigma = 0.025
        
        if ts is None:
            ts = np.arange(0.02,0.18,0.005)    # Range of half width times, roughly the mean +/- 3.2 sigma
            
        prior = np.exp(-(ts-prior_mean)**2/(2*prior_sigma**2))
        #prior = np.ones_like(ts) # Uniform prior for sigma
        
        #prior = np.exp(-(t_sigma_possible-prior_sigma)**2/(2*prior_sigma**2))        
        #i_uniform = np.logical_and(t_sigma_possible>=prior_sigma-uniform_delta,t_sigma_possible<=prior_sigma+uniform_delta)
        #prior[i_uniform] = prior[i_uniform].min()
        
        return ts,prior
    def optimize(self,x,sample_rate,measurement_sigma,rr_interval=None):
        # Use a Bayesian MAP approach to estimate the center of the p wave                   
        t_center_possible,prior_center = self.get_center_prior()
        t_sigma_possible,prior_sigma = self.get_sigma_prior()
        
        likelihood = np.zeros_like(t_center_possible)        
        sigma_best = np.zeros_like(t_center_possible)
        for ic,t_center in enumerate(t_center_possible):
            self.t_center = t_center

            # Optimize with uniform prior for sigma            
            ll = np.zeros_like(t_sigma_possible)
            for isig,t_sigma in enumerate(t_sigma_possible):
                self.t_width = t_sigma
                xm = self.get_modeled_segment(x,sample_rate,rr_interval)
                ll[isig] = np.exp(-np.nanmean((x-xm)**2)/(2*measurement_sigma**2))*prior_sigma[isig]
                
            i_ml = np.argmax(ll)
            self.t_width   = t_sigma_possible[i_ml]
            likelihood[ic] = ll[i_ml]
            sigma_best[ic] = t_sigma_possible[i_ml]
            
        posterior = prior_center*likelihood        
        i_map = np.argmax(posterior)  
        self.t_center = t_center_possible[i_map]
        self.t_width  = sigma_best[i_map]
        self.get_modeled_segment(x,sample_rate,rr_interval) # Update the other model coefficients                              
    def get_modeled_segment(self,x,sample_rate,rr_interval=None,axes=None):        
        n_samples = len(x)            
        ix = np.arange(-(n_samples-1),1)
        tx = ix/sample_rate
        
        #t0  = max(self.t_segment_start,-rr_interval*self.interval_portion) # Time of the start of the segment relative to the R wave
        t0  = self.t_segment_start # -0.33 # self.t_center - self.t_sigma - self.t_baseline
        t1  = self.t_segment_end   # -0.08 # self.t_center + self.t_sigma
        if rr_interval is not None:
            t0 = max(t0,-rr_interval*self.interval_portion)

        assert(rr_interval is not None) # DEBUGHERE
        assert(t0<t1)     

        # Great Performance with These        
        # t0  = -0.27 # self.t_center - self.t_sigma - self.t_baseline
        # t1  = -0.10 # self.t_center + self.t_sigma
        ms  = np.logical_and(tx>=t0,tx<=t1) # Mask for the segment
        txf = tx[ms] # Times used to fit model
             
        A = np.zeros((len(txf),4))
        A[:,0] = 1.0
        A[:,1] = txf
        A[:,2] = txf**2        
        A[:,3] = self.bump(txf)
        
        w = lstsq(A,x[ms])[0]
        
        # If the bump had a negative coefficient, resolve for just the baseline
        # This is equivalent to having a uniform prior over the non-negative
        # values of the bump coefficient and a value of zero for negative values
        zero_bump = False
        if w[-1]<0:
            zero_bump = True
            A = A[:,:-1] # Trim of the bump column
            w = lstsq(A,x[ms])[0]
        
        xf = A @ w                              
        xm = np.ones_like(x)*np.nan          
        xm[ms] = xf    

        self.bump_amplitude = 0.0        
        if not zero_bump:  
            # Solve for the bump amplitude using a linear baseline model
            # from the greater of the start of the segment or the start of the bump
            # to the lesser of the end of the segment or the end of the bump  
                    
            i_min = np.argmin(np.abs(txf-(self.t_center-self.t_width/2)))
            i_ctr = np.argmin(np.abs(txf-(self.t_center               )))
            i_max = np.argmin(np.abs(txf-(self.t_center+self.t_width/2)))
            
            assert(not np.isnan(xf[i_min]))
            assert(not np.isnan(xf[i_ctr]))
            assert(not np.isnan(xf[i_max]))
            assert(i_min<=i_ctr and i_ctr<=i_max)

            if i_min<i_max:
                slope               = (xf[i_max]-xf[i_min])/(txf[i_max]-txf[i_min])
                baseline_at_ctr     = xf[i_min] + slope*(txf[i_ctr]-txf[i_min])        
            else:
                assert(i_min==i_max and i_min==i_ctr)
                baseline_at_ctr = xf[i_ctr]
            self.bump_amplitude = xf[i_ctr]-baseline_at_ctr
        
            if axes is not None:
                tm = tx[ms]                
                axes.plot([tm[i_min],tm[i_max]],[xf[i_min]      ,xf[i_max]],color='tab:green',linewidth=0.6,marker='.',markersize=2)
                axes.plot([tm[i_ctr],tm[i_ctr]],[baseline_at_ctr,xf[i_ctr]],color='tab:green',linewidth=0.6,marker='.',markersize=2)
        
        return xm  
                 
class TentModel:
    def __init__(self,x,sample_rate):
        self.x           = x.copy()
        self.sample_rate = sample_rate
        self.n_samples   = len(self.x)
        
        t_p_wave_peak           =  0.165 # Duration of p wave peak  relative to R wave      in seconds
        t_p_wave_peak_start     =  0.065 # Duration of p wave start relative to p wave peak in seconds
        t_p_wave_peak_end       =  0.055 # Duration of p wave end   relative to p wave peak in seconds
        t_p_wave_baseline_start =  0.150 # Time prior to start of p wave to use for baseline calculation in seconds  
        
        self.i_peak           = (self.n_samples-1) - int(t_p_wave_peak          *self.sample_rate)
        self.n_peak_start     = int(t_p_wave_peak_start    *self.sample_rate)
        self.n_peak_end       = int(t_p_wave_peak_end      *self.sample_rate)
        self.n_baseline_start = int(t_p_wave_baseline_start*self.sample_rate) 
    def copy(self):
        tm = TentModel(self.x,self.sample_rate)
        tm.i_peak           = self.i_peak
        tm.n_peak_start     = self.n_peak_start
        tm.n_peak_end       = self.n_peak_end
        tm.n_baseline_start = self.n_baseline_start
        return tm       
    def fit(self):                             
        self.i_peak  = find_nearest_maximum(self.x,self.i_peak) # Index of p-wave peak
                                         
        #ixp = find_first_right_maximum(average_p_wave)
        #ixp = find_nearest_maximum(average_p_wave,i_p_wave_peak_prior)
        #i_p_wave_peak = i_p_wave_segment[ixp]
        self.i_peak_start     = max(self.i_peak       - self.n_peak_start    ,0)
        self.i_peak_end       = min(self.i_peak       + self.n_peak_end      ,self.n_samples-1)    
        self.i_baseline_start = max(self.i_peak_start - self.n_baseline_start,0)                    
    def optimize(self):
        # User-Specified Parameters
        n_points_dither = 1 # Number of points to dither the p wave peak index        
        i_dither =  np.arange(-n_points_dither,n_points_dither+1)
        for c_iteration in range(15):
            changed = False
            for dither_parameter in ['i_peak','n_peak_start']: # DEBUGHERE
                i_peak           = self.i_peak
                n_peak_start     = self.n_peak_start
                n_peak_end       = self.n_peak_end
                n_baseline_start = self.n_baseline_start
                
                mae = np.ones(len(i_dither))*np.inf
                for c,id in enumerate(i_dither): 
                    if dither_parameter=='i_peak':
                        if i_peak + id<0 or i_peak + id>=self.n_samples:
                            continue
                        self.i_peak       = i_peak       + id
                        self.n_peak_start = n_peak_start + id
                        self.n_peak_end   = n_peak_end   - id
                    elif dither_parameter=='n_peak_start':  
                        if n_peak_start+id<0:
                            continue                  
                        self.n_peak_start = n_peak_start + id
                        self.n_peak_end   = n_peak_start + id # DEBUGHERE
                    elif dither_parameter=='n_peak_end':
                        if n_peak_end+id<0:
                            continue
                        self.n_peak_end = n_peak_end + id
                    elif dither_parameter=='n_baseline_start':
                        if n_baseline_start+id<0:
                            continue 
                        self.n_baseline_start = n_baseline_start + id
                    xm = self.segment_model()
                    mae[c] = np.nanmean(np.abs(self.x-xm))            
                i_best = np.argmin(mae)
                if i_dither[i_best]!=0:
                    changed = True
                if dither_parameter=='i_peak':
                    self.i_peak       = i_peak       + i_dither[i_best]
                    self.n_peak_start = n_peak_start + i_dither[i_best]
                    self.n_peak_end   = n_peak_end   - i_dither[i_best]
                    xm = self.segment_model()
                elif dither_parameter=='n_peak_start':
                    self.n_peak_start = n_peak_start + i_dither[i_best]
                elif dither_parameter=='n_peak_end':
                    self.n_peak_end = n_peak_end + i_dither[i_best]
                elif dither_parameter=='n_baseline_start':
                    self.n_baseline_start = n_baseline_start + i_dither[i_best]              
            if not changed:
                break
    def segment_model(self):
        xm = np.ones(self.n_samples)*np.nan # Model of the p wave segment - fill in with nans where model output is not specified
    
        assert(self.n_peak_start>0)
        assert(self.n_peak_end>0)
        assert(self.n_baseline_start>0)
    
        i_peak           = self.i_peak
        i_peak_start     = max(i_peak       - self.n_peak_start    ,0)
        i_peak_end       = min(i_peak       + self.n_peak_end      ,self.n_samples-1)    
        i_baseline_start = max(i_peak_start - self.n_baseline_start,0)
                   
        # baseline to p wave start
        i_bl_pws     = np.arange(i_baseline_start,i_peak_start+1)    
        x_start      = self.x[i_baseline_start]
        x_end        = self.x[i_peak_start]
        xm[i_bl_pws] = np.linspace(x_start,x_end,len(i_bl_pws))
        
        # p wave start to p wave peak
        i_pws_pwp    = np.arange(i_peak_start,i_peak+1)
        x_start      = self.x[i_peak_start]
        x_end        = self.x[i_peak]
        xm[i_pws_pwp] = np.linspace(x_start,x_end,len(i_pws_pwp))
                
        # p wave peak to p wave end
        i_pwp_pwe    = np.arange(i_peak,i_peak_end+1)
        x_start      = self.x[i_peak]
        x_end        = self.x[i_peak_end]
        xm[i_pwp_pwe] = np.linspace(x_start,x_end,len(i_pwp_pwe))
        
        return xm     
    def draw_knots(self,axes):      
        # User-Specified Parameters
        markersize = 4
        
        t = np.arange(-(self.n_samples-1),1)/self.sample_rate
                                        
        ipwp = self.i_peak
        ipws = max(ipwp - self.n_peak_start    ,0)
        ipwe = min(ipwp + self.n_peak_end      ,self.n_samples-1)  
        ibls = max(ipws - self.n_baseline_start,0)

        axes.plot(t[ipwp      ],self.x[ipwp],color='tab:green',marker='.',markersize=markersize,label='P wave peak',alpha=0.7)
        axes.plot(t[ipws      ],self.x[ipws],color='tab:red'  ,marker='.',markersize=markersize,label='P wave peak',alpha=0.7)
        axes.plot(t[ipwe      ],self.x[ipwe],color='tab:red'  ,marker='.',markersize=markersize,label='P wave peak',alpha=0.7)
        axes.plot(t[ibls      ],self.x[ibls],color='tab:blue' ,marker='.',markersize=markersize,label='P wave peak',alpha=0.7)   
    def draw_model(self,axes):
        t = np.arange(-(self.n_samples-1),1)/self.sample_rate
                                        
        ipwp = self.i_peak
        ipws = max(ipwp - self.n_peak_start    ,0)
        ipwe = min(ipwp + self.n_peak_end      ,self.n_samples-1)  
        ibls = max(ipws - self.n_baseline_start,0)
        
        i_linear   = [ibls,ipws,ipwp,ipwe] # Indices of linear spline model
        i_baseline = [ipws,ipwe]
        
        axes.plot(t[i_linear  ],self.x[i_linear  ],color='blue',linewidth=0.8,label='Linear spline model',alpha=0.9,marker='.',markersize=2)
        axes.plot(t[i_baseline],self.x[i_baseline],color='blue',linewidth=0.8,label='Baseline'           ,alpha=0.9)
    def get_amplitude(self):
        
        ipwp = self.i_peak
        ipws = max(ipwp - self.n_peak_start    ,0)
        ipwe = min(ipwp + self.n_peak_end      ,self.n_samples-1)  
        
        x = self.x[ipws:ipwe+1]
        
        baseline_slope   = (x[-1]-x[0])/(ipwe-ipws)
        baseline_offset  = x[0]
        baseline_at_peak = baseline_slope*(ipwp-ipws)+baseline_offset
        p_wave_amplitude = x[ipwp-ipws]-baseline_at_peak
            
        return p_wave_amplitude
def plot_bump_prior(figures_directory):    
    # The sample rate doesn't affect the plot, but is a required argument
    sample_rate = 250 
    bump_model = BumpModel(sample_rate)
    
    step   = 0.0005
    tc     = np.arange(-0.30,0.0,step)
    _,prior  = bump_model.get_center_prior(tc=tc)
    prior /= step*np.sum(prior)
            
    figure = get_figure()
    axes = draw_array_axes(1,1,1,1,figure=figure)
    axes.plot(tc,prior,linewidth=1.5,color='tab:blue')
    axes.set_xlim([tc[0],tc[-1]])
    axes.set_ylim([0.0,1.05*np.max(prior)])
    axes.set_xlabel('Time Pior to R Wave (s)')
    axes.set_ylabel('$a_5$ Prior Density')
    
    format_figure_text(figure)
    
    figure_path = os.path.join(figures_directory,'bump_prior.pdf')
    figure.savefig(figure_path)        
def get_average_p_waves(subject,data_directory):
    t_start_p_wave = -0.40 # Units of seconds relative to the R wave
    t_end_p_wave   = -0.00 # Units of seconds relative to the R wave
            
    annotations  = read_annotations(subject,data_directory)
    ecg          = read_ecg(subject,annotations,data_directory)
    ecg_filtered = filter_ecg(ecg)
    p_waves      = get_p_wave_annotations(annotations)    
    
    assert(len(p_waves)>=10)
    assert(ecg['raw_sample_rate']==ecg['filtered_sample_rate'])
    
    sample_rate     = ecg['filtered_sample_rate']
    i_start_offset  = int(t_start_p_wave*sample_rate) # Offset in samples relative to the R wave index
    i_end_offset    = int(t_end_p_wave  *sample_rate) # Offset in samples relative to the R wave index        
    i_offset        = np.arange(i_start_offset,i_end_offset+1)
      
    averages = dict()
    averages['sample_rate'] = sample_rate
    for signal_type in ['raw','filtered']:        
        x = ecg_filtered[signal_type]
        average_p_wave = np.zeros(len(i_offset))    
        for p_wave in p_waves:
            i_r_wave = p_wave['i_r']      
            segment_p_wave = x[i_r_wave + i_offset]
            average_p_wave  += segment_p_wave
        average_p_wave  /= len(p_waves)
        averages[signal_type] = average_p_wave
    return averages
def get_p_wave_segment(i_r,subject,data_directory):
    t_start_p_wave = -0.40 # Units of seconds relative to the R wave
    t_end_p_wave   = -0.00 # Units of seconds relative to the R wave
            
    annotations         = read_annotations(subject,data_directory)
    ecg                 = read_ecg(subject,annotations,data_directory)
    ecg_filtered        = filter_ecg(ecg)
    p_waves_annotations = get_p_wave_annotations(annotations)    
    
    assert(len(p_waves_annotations)>=10)
    assert(ecg['raw_sample_rate']==ecg['filtered_sample_rate'])
    
    sample_rate     = ecg['filtered_sample_rate']
    i_start_offset  = int(t_start_p_wave*sample_rate) # Offset in samples relative to the R wave index
    i_end_offset    = int(t_end_p_wave  *sample_rate) # Offset in samples relative to the R wave index        
    i_offset        = np.arange(i_start_offset,i_end_offset+1)
      
    for p_wave_annotations in p_waves_annotations:
        i_r_wave = p_wave_annotations['i_r']    
        if i_r_wave == i_r:
            p_wave_segment = dict()
            p_wave_segment['raw'     ] = ecg_filtered['raw'     ][i_r_wave + i_offset]
            p_wave_segment['filtered'] = ecg_filtered['filtered'][i_r_wave + i_offset]
            p_wave_segment['raw_sigma'     ] = np.std(p_wave_segment['raw'     ])
            p_wave_segment['filtered_sigma'] = np.std(p_wave_segment['filtered'])
            return p_wave_segment
    return None
def get_scatter_data(subject,annotator,data_directory):
    if annotator=='balaji':
        annotations = get_balaji_annotations(subject,data_directory)
    else:
        assert(False)

    sample_rate         = get_sample_rate(data_directory,subject=subject)
    p_waves_annotations = get_p_waves_annotations(subject,data_directory)                        

    bump_model = BumpModel(sample_rate)
    
    x  = list()
    xh = list()
    for pws in p_waves_annotations:
        i_r   = pws['i_r'] 
        i_r_p = pws['i_r_previous']
        rri   = (i_r-i_r_p)/sample_rate
        if not (annotations['iRWave']==i_r).any():
            continue
        
        pws = get_p_wave_segment(i_r,subject,data_directory)        
        ms  = 0.1*pws['filtered_sigma']
        bump_model.optimize(pws['filtered'],sample_rate,ms,rr_interval=rri)
   
        i_row = np.where(annotations['iRWave']==i_r)[0][0]
        x .append(annotations['pWaveAmplitude'][i_row])                
        xh.append(bump_model.bump_amplitude)    
    if any(np.array(xh)<0):
        print(f'Negative Bump Amplitude for Subject {subject}')
    return x,xh
def plot_scatter_plots(subject_list,figures_directory):
    n_subjects = len(subject_list[:])
    data = dict()
    x_all  = list()
    xh_all = list()
    xd_all = list()
    deviations = dict()
    for c,subject in enumerate(subject_list[:n_subjects]):
        #print('Processing subject: ',subject)
        x,xh = get_scatter_data(subject,'balaji',data_directory)
        data[subject] = dict()
        data[subject]['x' ] = x
        data[subject]['xh'] = xh
        
        xd = np.array(xh)-np.array(x)
        
        x_all.extend(x)
        xh_all.extend(xh)
        xd_all.extend(xd)
        deviations[subject] = np.mean(np.abs(xd))
    x_all = np.array(x_all)
    xh_all = np.array(xh_all)
    xd_all = np.array(xd_all) 
    
    # Sort the deviations dictionary by value from largest to smallest
    deviations = {k: v for k, v in sorted(deviations.items(), key=lambda item: item[1],reverse=True)}
    
    # Print the top 10 worst subjects    
    print('Top 10 Worst Subjects:')
    for c,subject in enumerate(deviations.keys()):
        print(f'{c+1:2d}: {subject:10s} {deviations[subject]:.3f}')
        if c==9:
            break            
    
    # Draw a Scatter Plot
    figure = get_figure()
    axes = draw_array_axes(1,1,1,1,figure=figure)
    colormap = mpl.colormaps.get_cmap('hsv')    
    max_value = -np.inf
    min_value =  np.inf
    for c,subject in enumerate(subject_list[:n_subjects]):
        color = colormap(c/n_subjects)
        x = data[subject]['x']
        xh = data[subject]['xh']
        axes.plot(x,xh,color=color,marker='.',markersize=3,alpha=0.5,linestyle='None',label=subject)
        max_value = max(max_value,max(x),max(xh))    
        min_value = min(min_value,min(x),min(xh))    
    axes.axhline(0,color='tab:gray',linewidth=0.5,linestyle=':')
    axes.axvline(0,color='tab:gray',linewidth=0.5,linestyle=':')
    axes.plot([min_value,max_value],[min_value,max_value],color='tab:gray',linewidth=0.5)    
    axes.set_xlim([min_value,max_value*1.02])
    axes.set_ylim([min_value,max_value*1.02])
    axes.set_xlabel('Expert (SB) Amplitude')
    axes.set_ylabel('Algorithm Amplitude')

    format_figure_text(figure)
    
    figure_path = os.path.join(figures_directory,'scatter_alg_vs_sb.pdf')   
    figure.savefig(figure_path)

    # Draw a Bland-Altman Plot
    figure = get_figure()
    axes = draw_array_axes(1,1,1,1,figure=figure)
    
    xd_sigma     = np.std(xd_all)
    xd_average   = np.average(xd_all)
    scale_factor = stats.t.ppf(0.975,len(xd_all)-1)
    xd_low       = xd_average - scale_factor*xd_sigma
    xd_high      = xd_average + scale_factor*xd_sigma
    axes.axhspan(xd_low,xd_high,color=np.ones(3)*0.8,alpha=0.2)
    axes.axhline(np.average(xd_all),color='tab:gray',linewidth=1.0)    
    
    colormap = mpl.colormaps.get_cmap('hsv')
    for c,subject in enumerate(subject_list[:n_subjects]):
        color = colormap(c/n_subjects)
        x  = data[subject]['x']
        xh = data[subject]['xh']
        xa = (np.array(x)+np.array(xh))/2.0
        xd = np.array(xh)-np.array(x)        
        axes.plot(xa,xd,color=color,marker='.',markersize=3,alpha=0.5,linestyle='None',label=subject)
    axes.axhline(0,color='tab:gray',linewidth=0.5,linestyle=':')
    
    y_min  = np.min(xd_all)
    y_max  = np.max(xd_all)
    y_rng  = y_max-y_min
    y_low  = y_min - 0.05*y_rng
    y_high = y_max + 0.05*y_rng
    axes.set_ylim([y_low,y_high])
    axes.set_xlim([0,max_value*1.02])
    axes.set_xlabel('Average Amplitude (Alg and SB)')
    axes.set_ylabel('Difference')
    
    format_figure_text(figure)
    
    figure_path = os.path.join(figures_directory,'bland-altman_alg_vs_sb.pdf')   
    figure.savefig(figure_path)    
          
    # Calculate the Pearson and Spearman correlation coefficients
    r_pearson,p_pearson = stats.pearsonr(x_all,xh_all)
    r_spearman,p_spearman = stats.spearmanr(x_all,xh_all)
    print(f'Pearson Correlation Coefficient : {r_pearson:.3f} (p={p_pearson:.3f})')
    print(f'Spearman Correlation Coefficient: {r_spearman:.3f} (p={p_spearman:.3f})')   
    print(f'Average Absolute Difference     : {np.average(np.abs(xd_all)):.3f}')
    print(f'Lower Limit of Agreement        : {xd_low:.3f}')
    print(f'Upper Limit of Agreement        : {xd_high:.3f}')
    print(f'95% Confidence Interval Width   : {xd_high-xd_low:.3f}')
    print(f'Average Bias                    : {xd_average:.3f}')
def draw_annotation_model(subject,data_directory,annotator,i_r,axes,label=None):
    if annotator=='balaji':
        annotations = get_balaji_annotations(subject,data_directory)
    else:
        assert(False)
    
    annotations_temp = read_annotations(subject,data_directory)
    sample_rate      = annotations_temp.fs

    if not (annotations['iRWave']==i_r).any():
        return None 
    i_row = np.where(annotations['iRWave']==i_r)[0][0]
    
    tr = annotations['iRWave'][i_row]/sample_rate    
    ts = annotations['tStart'][i_row] - tr
    tp = annotations['tPeak' ][i_row] - tr
    te = annotations['tEnd'  ][i_row] - tr
    xs = annotations['yStart'][i_row]
    xp = annotations['yPeak' ][i_row]
    xe = annotations['yEnd'  ][i_row]
        
    slope = (xe-xs)/(te-ts)
    xb    = xs + slope*(tp-ts)
    
    if label is None:
        label = subject
    
    t3 = np.array([ts,tp,te])
    x3 = np.array([xs,xp,xe])        
    axes.plot(t3,x3,color='tab:red',linewidth=0.5,marker='.',markersize=2,label=label)  
    
    t2 = np.array([ts,te])
    x2 = np.array([xs,xe])
    axes.plot(t2,x2,color='tab:red',linewidth=0.5)
    
    t2 = np.array([tp,tp])
    x2 = np.array([xb,xp])
    axes.plot(t2,x2,color='tab:red',linewidth=0.5)               
def plot_pulses(subject,
                data_directory,
                figures_directory,
                n_rows=5,
                n_columns=2,
                show_model=True,
                show_expert=False,
                show_raw_ecg=True):
    # User-Specified Parameters
    t_start_p_wave = -0.40 # Units of seconds relative to the R wave
    t_end_p_wave   = -0.00 # Units of seconds relative to the R wave
    t_start_plot   = -0.6  # Units of seconds relative to the R wave
    t_end_plot     =  0.1  # Units of seconds relative to the R wave
            
    # Read data
    annotations         = read_annotations(subject,data_directory)
    ecg                 = read_ecg(subject,annotations,data_directory)
    ecg_filtered        = filter_ecg(ecg)
    p_waves_annotations = get_p_wave_annotations(annotations)    
    
    assert(len(p_waves_annotations)>=10)
        
    # Preprocessing
    sample_rate           = ecg['filtered_sample_rate']
    i_start_plot_offset   = int(t_start_plot  *sample_rate) # Offset in samples relative to the R wave index
    i_end_plot_offset     = int(t_end_plot    *sample_rate) # Offset in samples relative to the R wave index
    i_start_p_wave_offset = int(t_start_p_wave*sample_rate) # Offset in samples relative to the R wave index
    i_end_p_wave_offset   = int(t_end_p_wave  *sample_rate) # Offset in samples relative to the R wave index    
    
    i_offset_plot    = np.arange(i_start_plot_offset  ,i_end_plot_offset  +1)
    i_offset_p_wave  = np.arange(i_start_p_wave_offset,i_end_p_wave_offset+1)
    
    tx   = i_offset_plot/sample_rate
    t_segment_p_wave = i_offset_p_wave/sample_rate

    # Build the Initial Model
    x = ecg_filtered['filtered']
    average_p_wave = np.zeros(len(i_offset_p_wave))
    for c,p_wave_annotations in enumerate(p_waves_annotations):
        i_r_wave = p_wave_annotations['i_r']      
        spw = x[i_r_wave + i_offset_p_wave]
        average_p_wave  += spw
    average_p_wave  /= len(p_waves_annotations)
    
    #tent_model = TentModel(average_p_wave,sample_rate)
    #tent_model.fit()            
    #tent_model.optimize()

    # Create an Array of Plots Showing the Model for Each Pulse
    figure = get_figure(n_rows=n_rows,n_columns=n_columns)
    
    x  = ecg_filtered['raw']
    xf = ecg_filtered['filtered']
    for c,p_wave_annotations in enumerate(p_waves_annotations):
        if c>=n_rows*n_columns:
            break
        i_r_wave          = p_wave_annotations['i_r']
        i_r_wave_previous = p_wave_annotations['i_r_previous']
        rr_interval       = (i_r_wave-i_r_wave_previous)/sample_rate  
        spw  = xf[i_r_wave + i_offset_p_wave]
        sx   = x [i_r_wave + i_offset_plot  ]
        sxf  = xf[i_r_wave + i_offset_plot  ] 
                        
        axes = draw_array_axes(n_rows,n_columns,int(c/n_columns)+1,c%n_columns+1,figure=figure)
        
        axes.axhline(0,color='tab:gray',linewidth=0.5,linestyle=':')
        axes.axvline(0,color='tab:gray',linewidth=0.5,linestyle=':')
        
        if show_raw_ecg:
            axes.plot(tx,sx ,color='tab:gray',linewidth=0.8,label='ECG'     ,alpha=0.4)
        axes.plot(tx,sxf,color='tab:blue',linewidth=0.8,label='Filtered ECG',alpha=0.4)
        
        draw_annotation_model(subject,data_directory,'balaji',i_r_wave,axes,label='Our Expert (SB)')
        
        if show_model:
            bump_model = BumpModel(sample_rate)
            bump_model.optimize(spw,sample_rate,0.1*np.std(x),rr_interval=rr_interval)
            xh = bump_model.get_modeled_segment(spw,sample_rate,rr_interval=rr_interval,axes=axes)
            axes.plot(t_segment_p_wave,xh,color='tab:green',linewidth=0.6,label='Bump model')
        
        if False:        
            y_min   = min(sx.min(),sxf.min())
            y_max   = max(sx.max(),sxf.max())
            y_range = y_max - y_min
            y_lower = y_min - 0.01*y_range
            y_upper = max(y_max - 0.40*y_range,np.abs(max(spw))*1.05)
        if True:
            ms      = np.logical_and(tx>-0.35,tx<-0.1)
            assert(np.sum(ms)>0)
            y_min   = sxf[ms].min()
            y_max   = sxf[ms].max()
            if show_raw_ecg:
                y_min = min(y_min,sx[ms].min())
                y_max = max(y_max,sx[ms].max())
            y_range = y_max - y_min
            y_lower = y_min - 1.0*y_range
            y_upper = y_max + 1.0*y_range
                        
        if show_expert:         
            i_start = p_wave_annotations['i_p_start']
            i_end   = p_wave_annotations['i_p_end']
            i_peak  = p_wave_annotations['i_p_peak']
            
            # Find the index of sx that corresponds to i_start (the index relative to the R wave index)         
            isxs  = - i_start_plot_offset + (i_start - i_r_wave) 
            isxe  = - i_start_plot_offset + (i_end   - i_r_wave)
            isxp  = - i_start_plot_offset + (i_peak  - i_r_wave)    
            
            sxs   = sx[isxs]
            sxe   = sx[isxe]
            sxp   = sx[isxp]
            
            slope = (sxe-sxs)/(tx[isxe]-tx[isxs])
            sxb   = sxs + slope*(tx[isxp]-tx[isxs])
            
            i_expert  = [isxs,isxp,isxe]      
              
            axes.plot(tx[i_expert],sx[i_expert],color='tab:olive',linewidth=0.5,marker='.',markersize=3,label='QT Expert (q1c)')
            axes.plot(tx[[isxs,isxe]],[sxs,sxe],color='tab:olive',linewidth=0.5)
            axes.plot(tx[[isxp,isxp]],[sxb,sxp],color='tab:olive',linewidth=0.5)
        
        axes.set_xlim([t_start_plot,t_end_plot])
        axes.set_ylim([y_lower,y_upper])   
        axes.set_xlabel(f'Time Since R Wave (s)')        
        
        if c%n_columns==0:
            axes.set_ylabel(f'Amplitude (adu)')
        else:
            axes.set_yticklabels([])
        
        if n_rows==1 and n_columns==1:
            axes.legend()
    
    format_figure_text(figure)
    
    # Save figure
    pulses_directory = os.path.join(figures_directory,'pulses')
    if not os.path.exists(pulses_directory):
        os.makedirs(pulses_directory)
    figure_path = os.path.join(pulses_directory,subject+'.pdf')
    figure.savefig(figure_path)    
    plt.close(figure)
def plot_pulse_overlaps(subject,data_directory,figures_directory):
    # User-Specified Parameters
    t_start_p_wave = -0.40 # Units of seconds relative to the R wave
    t_end_p_wave   = -0.00 # Units of seconds relative to the R wave
    t_start_plot   = -0.5  # Units of seconds relative to the R wave
    t_end_plot     =  0.1  # Units of seconds relative to the R wave
            
    # Read data
    annotations         = read_annotations(subject,data_directory)
    ecg                 = read_ecg(subject,annotations,data_directory)
    ecg_filtered        = filter_ecg(ecg)
    p_waves_annotations = get_p_wave_annotations(annotations)    
    
    assert(len(p_waves_annotations)>=10)
    
    # Preprocessing
    sample_rate           = ecg['filtered_sample_rate']
    i_start_plot_offset   = int(t_start_plot  *sample_rate) # Offset in samples relative to the R wave index
    i_end_plot_offset     = int(t_end_plot    *sample_rate) # Offset in samples relative to the R wave index
    i_start_p_wave_offset = int(t_start_p_wave*sample_rate) # Offset in samples relative to the R wave index
    i_end_p_wave_offset   = int(t_end_p_wave  *sample_rate) # Offset in samples relative to the R wave index    
    
    i_offset_plot    = np.arange(i_start_plot_offset  ,i_end_plot_offset  +1)
    i_offset_p_wave  = np.arange(i_start_p_wave_offset,i_end_p_wave_offset+1)
    
    t_segment_plot   = i_offset_plot/sample_rate
    t_p_wave_plot    = i_offset_p_wave/sample_rate
    
    figure = get_figure()
    
    n_channels = ecg_filtered['filtered_all'].shape[0]    
    
    for i_channel in range(n_channels):
        for c_type in range(2):
            if c_type==0:
                x = ecg_filtered['filtered_all'][i_channel]                
            elif c_type==1:
                x = ecg_filtered['raw_all'][i_channel]
            average_p_wave = np.zeros(len(i_offset_p_wave))
            average_segment = np.zeros(len(i_offset_plot))                              
            axes = draw_array_axes(n_channels,2,1+i_channel,c_type+1,figure=figure)  
            axes.axvline(0,color='k',linewidth=0.5,linestyle=':')
            axes.axhline(0,color='k',linewidth=0.5,linestyle=':')
            for c,p_wave in enumerate(p_waves_annotations):
                i_r_wave = p_wave['i_r']      
                segment_plot   = x[i_r_wave + i_offset_plot]
                segment_p_wave = x[i_r_wave + i_offset_p_wave]
                average_segment += segment_plot
                average_p_wave  += segment_p_wave
                axes.plot(t_segment_plot,segment_plot,color='tab:gray',alpha=0.2,linewidth=0.1)
            average_segment /= len(p_waves_annotations)
            average_p_wave  /= len(p_waves_annotations)
            
            y_min   = average_p_wave.min()
            y_max   = average_p_wave.max()
            y_range = y_max-y_min
            y_lower = y_min - 0.01*y_range
            y_upper = y_max + 0.01*y_range
            
            #if i_channel==0 and c_type==0 and average_p_wave[-1]<0:
            #    print(f'Inverted average p wave for subject {subject}')
            
            axes.plot(t_segment_plot,average_segment,color=0.3*np.ones(3),linewidth=0.5,label='Average segment')
            
            # tent_model = TentModel(average_p_wave,sample_rate)
            # tent_model.fit()
                                    
            # tent_model.draw_knots(axes)
            
            # tent_model.optimize()
            # tent_model.draw_model(axes)
            
            bump_model = BumpModel(sample_rate)
            #xh = bump_model.get_modeled_segment(average_p_wave,sample_rate)
            #axes.plot(t_p_wave_plot,xh,color='tab:orange',linewidth=1.0,label='Bump model')

            bump_model.optimize(average_p_wave,sample_rate,0.1*np.std(x))
            xh = bump_model.get_modeled_segment(average_p_wave,sample_rate)
            axes.plot(t_p_wave_plot,xh,color='tab:green',linewidth=1.2,label='Bump model')

            t_center_possible,prior = bump_model.get_center_prior()
            prior *= 0.5*y_upper/np.max(prior)
            axes.plot(t_center_possible,prior,color='tab:blue',linewidth=1.0,label='Prior')
            
            #segment_model = tent_model.segment_model()
            #t_p_wave_plot = i_offset_p_wave/sample_rate
            #axes.plot(t_p_wave_plot,segment_model,color='tab:orange',linewidth=0.5,label='Model',alpha=0.5)
        
            axes.grid(axis='x',color='k',linestyle=':',linewidth=0.5)

            
            axes.set_xlim([t_start_plot,t_end_plot])
            axes.set_ylim([y_lower,y_upper])
            
            if i_channel==n_channels-1:
                axes.set_xlabel('Time Since R Wave (s)')
            if c_type==0:
                axes.set_ylabel('Amplitude')
            else: # Remove the y-axis tick labels 
                axes.set_yticklabels([])
    
    format_figure_text(figure)
    
    # Save figure
    overlaps_directory = os.path.join(figures_directory,'overlaps')
    if not os.path.exists(overlaps_directory):
        os.makedirs(overlaps_directory)
    figure_path = os.path.join(overlaps_directory,subject+'.pdf')
    figure.savefig(figure_path)    
    plt.close(figure)
def plot_filter_magnitude_response(subject,data_directory,figures_directory):
    subject      = subject_list[0]
    annotations  = read_annotations(subject,data_directory)
    ecg          = read_ecg(subject,annotations,data_directory)  
    filter_ecg(ecg,figures_directory)  
def get_ecg_sample_rate(subject,data_directory):
    annotations = read_annotations(subject,data_directory)
    sample_rate = annotations.fs
    return sample_rate
def get_balaji_annotations(subject,data_directory):
    directory = os.path.join(data_directory,'annotations','balaji')
    filepath = os.path.join(directory,subject+'.csv')
    if not os.path.isfile(filepath):
        return None
    data  = pd.read_csv(filepath,index_col=False)
    return data
def get_p_wave_peak_average(subject,data_directory):
    annotations  = read_annotations(subject,data_directory)
    p_waves      = get_p_wave_annotations(annotations)    
    sample_rate  = annotations.fs
    assert(len(p_waves)>=10)

    pwp_avg = list()    
    for p_wave in p_waves[:10]: # Just the first 10, to match the other annotations
        pwp = (p_wave['i_p_peak'] - p_wave['i_r'])/sample_rate
        pwp_avg.append(pwp)
    pwp_avg = np.mean(pwp_avg)
    return pwp_avg
def get_p_wave_width_average(subject,data_directory):
    annotations  = read_annotations(subject,data_directory)
    p_waves      = get_p_wave_annotations(annotations)    
    sample_rate  = annotations.fs
    assert(len(p_waves)>=10)

    pww_avg = list()    
    for p_wave in p_waves[:10]: # Just the first 10, to match the other annotations
        pww = (p_wave['i_p_end'] - p_wave['i_p_start'])/sample_rate
        pww_avg.append(pww)
    pww_avg = np.mean(pww_avg)
    return pww_avg
def plot_p_wave_peak_histograms(subject_list,figures_directory):
    tps  = list()
    bpt  = list()  # Balaji p-Waves times
    bpts = list() # Balaji p-Wave start times
    bpte = list() # Balaji p-Wave end times
    n_beats = 0
    for subject in subject_list:
        #print(f'Processing subject: {subject}')
        data = get_balaji_annotations(subject,data_directory)
        n_beats += data.shape[0]
        if any(data['tStart']>data['tPeak']) or any(data['tPeak']>data['tEnd']):
            print(f'Balaji Messed up on subject {subject}')
        sample_rate = get_ecg_sample_rate(subject,data_directory)
        btp  = np.mean(data['tPeak' ]-data['iRWave']/sample_rate)
        btps = np.mean(data['tStart']-data['tPeak'])
        btpe = np.mean(data['tEnd'  ]-data['tPeak'])
        bpt.append(btp)
        bpts.append(btps)
        bpte.append(btpe)
        
        tp = get_p_wave_peak_average(subject,data_directory)
        tps.append(tp)
            
        #save_overlap_plot(subject,data_directory,figures_directory)

    print('Balaji Annotations')
    print('t Peak   Median: ',np.median(bpt))
    print('t Peak   Sigma : ',np.std(bpt))
    print('t Before Median: ',np.median(bpts))
    print('t After  Median: ',np.median(bpte))
    print(f'Number of Subjects: {len(bpt)}')
    print(f'Number of Annotated P Waves: {n_beats}')

    # Generate histogram of tps
    bins = np.linspace(-0.3,-0.0,40)
    figure = get_figure()
    axis_boundaries = [0.02,0.02,0.16,0.15] # Top, Right, Bottom, Left

    axes = draw_array_axes(2,1,1,1,axis_boundaries=axis_boundaries,figure=figure)
    axes.hist(tps,bins=bins,density=True)
    mu    = np.mean(tps)
    sigma = np.std(tps)
    x     = np.linspace(mu-3*sigma,mu+3*sigma,100)
    axes.plot(x,stats.norm.pdf(x,mu,sigma),color='tab:red',linewidth=1.5)    
    
    #axes.axvline(np.mean(tps),color='tab:green',linewidth=1)
    #axes.set_xlabel('Time Since R Wave (s)')
    axes.set_ylabel(f'q1c Mean: {np.mean(tps):.3f} s\nStd: {np.std(tps):.3f} s')
    axes.set_xlim([bins[0],bins[-1]])
    axes.set_xticklabels([])

    axes = draw_array_axes(2,1,2,1,axis_boundaries=axis_boundaries,figure=figure)
    axes.hist(bpt,bins=bins,density=True)
    mu   = np.mean(bpt)
    sigma = np.std(bpt)
    x     = np.linspace(mu-3*sigma,mu+3*sigma,100)
    axes.plot(x,stats.norm.pdf(x,mu,sigma),color='tab:red',linewidth=1.5)
    
    #axes.axvline(np.mean(bpt),color='tab:green',linewidth=1)
    axes.set_xlabel('Time Since R Wave (s)')
    axes.set_ylabel(f'Mean: {np.mean(bpt):.3f} s\nStd: {np.std(bpt):.3f} s')
    axes.set_xlim([bins[0],bins[-1]])
    
    format_figure_text(figure)
    
    figure_path = os.path.join(figures_directory,'p_wave_peak_histograms.pdf')
    figure.savefig(figure_path)
def plot_p_wave_width_histograms(subject_list,figures_directory):
    qt_pww = list()
    sb_pww = list()
    for subject in subject_list:
        data = get_balaji_annotations(subject,data_directory)
        pww = np.mean(data['tEnd'  ]-data['tStart'])
        sb_pww.append(pww)
                
        pww = get_p_wave_width_average(subject,data_directory)
        qt_pww.append(pww)
        
    # Generate histogram of tps
    bins = np.linspace(0.0,0.2,40)
    figure = get_figure()
    axis_boundaries = [0.02,0.02,0.16,0.15] # Top, Right, Bottom, Left

    axes = draw_array_axes(2,1,1,1,axis_boundaries=axis_boundaries,figure=figure)
    axes.hist(qt_pww,bins=bins,density=True,color='tab:gray')
    mu    = np.mean(qt_pww)
    sigma = np.std(qt_pww)
    x     = np.linspace(mu-3*sigma,mu+3*sigma,100)
    axes.plot(x,stats.norm.pdf(x,mu,sigma),color='tab:red',linewidth=1.5)    
    
    axes.set_ylabel(f'q1c Mean: {np.mean(qt_pww):.3f} s\nStd: {np.std(qt_pww):.3f} s')
    axes.set_xlim([bins[0],bins[-1]])
    axes.set_xticklabels([])

    axes = draw_array_axes(2,1,2,1,axis_boundaries=axis_boundaries,figure=figure)
    axes.hist(sb_pww,bins=bins,density=True,color='tab:gray')
    mu    = np.mean(sb_pww)
    sigma = np.std(sb_pww)
    x     = np.linspace(mu-3*sigma,mu+3*sigma,100)
    axes.plot(x,stats.norm.pdf(x,mu,sigma),color='tab:red',linewidth=1.5)
    
    axes.set_xlabel('P Wave Width (s)')
    axes.set_ylabel(f'Mean: {np.mean(sb_pww):.3f} s\nStd: {np.std(sb_pww):.3f} s')
    axes.set_xlim([bins[0],bins[-1]])
    
    format_figure_text(figure)
    
    figure_path = os.path.join(figures_directory,'p_wave_width_histograms.pdf')
    figure.savefig(figure_path)    
#%% Main
if __name__=='__main__':
    # If not in the EMBC subdirectory, change to it
    if not os.getcwd().endswith('EMBC'):
        os.chdir('EMBC')

    data_directory = os.path.join(os.getcwd(),'data')
    figures_directory = os.path.join(os.getcwd(),'figures')

    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    subject_list = get_subject_list(data_directory)
    subject_list = get_valid_subject_list(subject_list,data_directory)
    
    subjects_exclude = ['sele0129','sel42']    
    subject_list = [sj for sj in subject_list if sj not in subjects_exclude]  
    
    subject_list.sort()

    #subject_list = ['sele0114','sel40','sel38','sel891','sel41','sel15814','sel847','sele0129','sel46','sel43']
    #subjects_inverted = ['sel14046','sel14172','sel15814','sel38','sel40','sel41','sel43','sel821']
    #subjects_inverted.extend(['sele0106','sele0107','sele0110','sele0112','sele0114','sele126','sele129','sele0133'])
    #subject_list = subjects_inverted
    
    # EMBC Paper Figures
    #sample_rate = get_ecg_sample_rate(data_directory)
    #ecg_filter_design(sample_rate,figures_directory)
    
    #plot_p_wave_peak_histograms (subject_list,figures_directory)
    #plot_p_wave_width_histograms(subject_list,figures_directory)
    
    #plot_bump_prior(figures_directory)
    
    plot_pulses('sel114',data_directory,figures_directory,n_rows=1,n_columns=1,show_model=False,show_expert=True)
    plot_pulses('sel100',data_directory,figures_directory,n_rows=1,n_columns=1,show_expert=False,show_raw_ecg=False)
    plot_pulses('sel45',data_directory,figures_directory,n_rows=1,n_columns=1,show_expert=False,show_raw_ecg=False)

    #plot_filter_magnitude_response(subject_list[0],data_directory,figures_directory)
    
    #print('Number of subjects: ',len(subject_list))

    #%% Draw Scatter Plots
    
    # Balaji Exclude
    # 45: sel38m.mat
    # 49: sel42m.mat
    # 75: sele0107m.mat
    # 78: sele0112m.mat
    # 79: sele0114m.mat
    # 80: sele0116m.mat
    # 85: sele0129m.mat
    # 87: sele0136m.mat
        
    #subjects_exclude = ['sel42','sele0409','sel213','sele0129','sel39']
    
    #subjects_negative = ['sel104','sel40','sel34','sel46','sel891','sele0112','sele0114','sele0116']
    #subjects_negative.sort()
    
    #plot_scatter_plots(subject_list[:],figures_directory)

    #%% Draw Pulse Plots
    
    # # subject_list = ['sel42','sele0409','sel891','sel15814','sel104']
    #subject_list = ['sel891','sel46','sele0409','sel16272']
    
    # for subject in subject_list[:50]:
    #     plot_pulses(subject,
    #                 data_directory,
    #                 figures_directory,
    #                 n_rows=1,
    #                 n_columns=1,
    #                 show_expert=False,
    #                 show_raw_ecg=False)
        
    # #%% Draw Overlap Plots    
    # for subject in subjects_exclude:
    #   draw_overlap_plot(subject,data_directory,figures_directory)
