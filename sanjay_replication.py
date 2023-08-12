# Simulate syntax in working memory
# coded from Sun Lin methods by SGM 2022.06.10
# fixed by Sun Lin 20/6/22
# param search
# 6/2023 - with different syntaxes
# 7/2023 - use strings as input
# 8/2023 - create EEGs
# 7.8.23 tring to get russian to work
#%% 

from numpy import *    # make life easy
from pdb import pm
from enum import Enum
"""
Overall structure
  - class Model - model parameters and long-term weights.
  - class Language - words and syntax as strings.
  - class Input - the sequence of input vectors for the model and language
  - class State - the neural network activations F,C, and short-term weights wfc,wcc

do_one_timestep():
  - the main function that computes neural activity.
main_...():
  - functions to run specific simulations.  

 1) Create a Model(), then a Language()
 2) Call model.setup_language(l) to store the language in the model's synapses.
 3) Create an Input(model) based on the language
 4) Create a blank State(model)
 5) Call single_trial(model, input, state) to run one trial

"""

class Language:
  """ Specify syntax and words of a language """
  def __init__(L):
    """		Basic default language definition:    """

    # roles begin with a letter indicating the word class
    # e.g. R=article, N=noun, J=adjective, V=verb
    # followed by a specifier. e.g.
    # 'NO' could be used to mean "noun in the object position".
    L.roles = 'RS NS V RO NO JS'.split()

    # allow these role sequences:
    L.permitted_sentences = [
      'RS NS V RO NO',
      'RS JS NS V RO NO'
    ]

    # words begin with a word class and an identifier.
    L.words = 'R0 N0 V0 R1 N1 J0 J1'.split()

    # Input string
    L.I_string = 'R0 N0 V0 R1 N1'.split()

#################################################################

class Model:
  Regime = Enum('Regime', ['Simple','Morphemes','Aphasias','PrimeCompr','PrimeProd'])
  def __init__( M, regime = Regime.Morphemes ):
    """ return a model M.
        Long term weights are in model, M
        Short-term weights are in the state, S.
        Word names are in the language, M.L
        """
    #####################
    ## 1. Model definition
    M.decay     = 1.0    # decay of activity per timestep = beta (1 = no decay)
    M.learnCC   = 30     # learning rate for CF weights
    M.learnFC   = 2      #               for CC weights
    M.gainWff   = -0.5   # inhibition between features of same dimension
    M.gainWccL  = 5.7    # gain of long-term weights  k_L
    M.gainWfcL  = 1.7    # 
    M.rangeWfc  = (-1,2) # constrain short term weights to this range (-k, eps-k)
    M.rangeWcc  = (-5,1)
    M.noiseC    = 0.02
    M.recallBias= 0.5    # change in gain at recall
    
    if regime == Model.Regime.Morphemes:
      M.gainWfcL = 3.7
      M.rangeWfc = (-2,4)
      M.gainWccL = 15.7
      M.learnCC  = 50
      M.rangeWcc = (-15,8)
      M.gainWff  = -1
    elif regime == Model.Regime.Aphasias:
      pass
      
    M.dt        = 0.5    # timestep size

    # three useful sigmoid functions
    M.lim_01  = lambda x: maximum(0,minimum(1,x))
    M.sigmoid = lambda x,k: 1/(1+exp(-10*(x-k)))
    M.lim_to  = lambda x,r: maximum(r[0], minimum(r[1],x))


  from numba import jit
  @jit(forceobj=True) # speed up the maths 
  def do_one_timestep(M,S):
    """update the state S by one time step
       S = current model state. Must contain the current input, as S.stim.
       M = model parameters.
    This function only changes S.F and S.C, the feature and conjunction activations,
    and S.Wfc and S.Wcc, the short-term weights.
    """

    Wfc  = S.Wfc + M.gainWfcL * M.Wfc_l    # overall weights =
    Wcc  = S.Wcc + M.gainWccL * M.Wcc_l    # short term plus long term weights
    # last column of input = whether we are in recall phase
    isnotRecall = S.stim[-1] == 0
    inputF   = S.stim[:-2]      # feature inputs

    # 1. Update Feature units
    dF = -M.decay  * S.F + M.sigmoid(
                      Wfc.T  @ S.C       # input from conjunctions
     +  M.gainWff * M.Wff    @ S.F       # lateral inhibition between features
     ,  isnotRecall * M.recallBias) + inputF


    # 2. Update Conjunctive units
    dC = -M.decay  * S.C + M.sigmoid(
                    Wfc  @ S.F       # input from features
      + 0.5       * Wcc  @ S.C       # conjunction-conjunction
      + M.noiseC  * random.normal( S.C )
    , isnotRecall * M.recallBias )
    dC[0] += S.stim[-2]       # at start of stim and recall, activate C1

    if isnotRecall: #no rapid plasticity during retrieval
        # 3. Change in feature-conjunction weights
        dWfc    = ( (S.C   - S.C.sum()/4) *
                    (S.F.T - S.F.sum()/4)   )  - S.C.sum()*S.F.sum()/16
        S.Wfc   = S.Wfc + M.learnFC * dWfc
        S.Wfc   = M.lim_to( S.Wfc, M.rangeWfc )  # limit to range

        # 4. Change in conjunction-conjunction weights
        g       = multiply( S.C,  M.lim_01( Wfc @ S.F ) ) # hadamard product
        dWcc    = S.C * (g-g.sum()/3).T  -  g @ S.C.T / 3
        S.Wcc   = S.Wcc + M.learnCC * dWcc.T
        S.Wcc   = M.lim_to( S.Wcc, M.rangeWcc )
        fill_diagonal(S.Wcc, 0) # zero the self connections on the diagonal

    S.F = M.lim_01( S.F + dF * M.dt )
    S.C = M.lim_01( S.C + dC * M.dt )

    return S

  def setup_language(M,L):
    """
    # Set up synapses based on this specified language
    """
    M.L = L
    M.NF = len(L.words) # number of words (features)
    M.NC = len(L.roles) # number of roles (conjunctions)

    # set up long-term word-to-role synapses, based on the language
    M.Wfc_l = zeros((M.NC,M.NF))
    for i,w in enumerate(L.words):  # for each word neuron
     for j,r in enumerate(L.roles): # for each role
        # does the word start with the same letter as the role does?
        if w[0] == r[0]:
          M.Wfc_l[ j,i ] = +1  # if so, make a synapse
    
 	  # set up long-term role-to-role synapses, based on language:
    M.Wcc_l = zeros((M.NC,M.NC))
    def allow(sentence):
      """ allow a particular syntax """
      u=sentence.split() # get the sequence of roles
      for i in range(len(u)-1):
        # allow the connection between role i and i+1
        M.Wcc_l [ L.roles.index(u[i+1]), L.roles.index(u[i]) ]  += 1
      M.Wcc_l = minimum(M.Wcc_l,1) # don't let values go above 1
    for s in L.permitted_sentences:
      allow( s )
    # normalise each row to add up to 1
    norm_val = M.Wcc_l.sum(axis=1)  # sum of weights for each row 
    norm_val[norm_val==0] = 1       # leave zero-rows as zero  (but what if it's close to zero??...)
    M.Wcc_l = M.Wcc_l / norm_val[:,None]  # divide by the total
    
    # long-term word-to-word inhibition
    M.Wff = ones((M.NF,M.NF))

#################################################################

class Input:
  def __init__(I,M):
    """
    Setup Input sequence - creates input I for a model,
    based on the model M and language M.L. 
    Input.stimulus is the matrix of word neuron activations, 
    for each time stage. 
      Stages of input to the feature units (each row is one stage)
      Each F unit receives input to keep it at the given value.
      NaN means no external input to the network
      Last column means "recall phase".
        2 = stimulate c[0]. >0 = use recall bias.
      These get put into S.stim, at the appropriate timesteps.
    """
    string = M.L.I_string # get words to use as input

    # Duration for each stage of the input
    # 1 timestep of -1, then 50 ts for each word.
    # then 100 ts delay, 5 ts retrieval cue, then 200 ts retrieval time.
    I.durations = [50] + [50]*len(string) + [100,  5, 200]
    # Calculate the time at start of recall
    I.recall_time = sum(I.durations[:-2])

    # create the word stimulus array
    I.word_indices = [ M.L.words.index(w) for w in string ]
    word_stim = array([
      array([w==stim_word  for w in M.L.words])*2-1
      for stim_word in string
    ])
    # add -1 inputs at start, at end, and then zeros for recall.
    I.stimulus = c_[
      [-1]*M.NF ,
      word_stim.T,
      [-1]*M.NF,
      [ 0]*M.NF,
      [ 0]*M.NF  ].T
    # add two columns, one for the recall cue (input to the first role unit),
    # and another for the recall phase (1 or 0)
    I.stimulus = c_[ I.stimulus,
                    [[1,0]] + [[0,0]]*(len(string)+1) + [[1,1],[0,1]] ]

#################################################################

def single_trial(M,I,S, plot_interval = 10):
    """run a single trial, with model M, inputs I, and initial state S"""
    time  = 0
    stage = 0
    stage_end_times = cumsum(I.durations) # when does each stage end?
    # create nonlinear transfer function

    ended = False
    # create time series for storing conjunctive and feature unit activity
    F_t = nan*zeros( (stage_end_times[-1],  S.F.size) )
    C_t = nan*zeros( (stage_end_times[-1],  S.C.size) )

    # start trial with zero activity
    S.C[:] = 0
    S.F[:] = 0

    while not ended:

        # 1. Get current Input stimulus
        time += 1

        if time > stage_end_times[stage]:
            stage += 1
        if stage >= stage_end_times.size:
            ended=True
            break
        S.stim = I.stimulus[stage,:][:,None]   # select a row, convert to column

        # 2. Run one timestep
        S = M.do_one_timestep(S)
        if (time % plot_interval)==0:
          S.F_t = F_t
          S.C_t = C_t
          display_result(S,M, update= time>plot_interval)

        F_t[time-1,:] = S.F.T        # store current activity
        C_t[time-1,:] = S.C.T

    S.F_t = F_t     # finished one trial - save activity timecourse.
    S.C_t = C_t
    # get the sequence of maximally active features.
    winF_t  = S.F_t[I.recall_time:,:].argmax( axis=1 ) # winner at each time
    changes = diff(concatenate(([-1], winF_t)))    # points where winner changes
    S.out   = winF_t[ abs(changes)>0 ]             # grab values at change points
    import matlib
    S.acc   = matlib.slide_match( S.out, I.word_indices ) # best accuracy
    
    return S



############################################################
class State:
  def __init__(s,M):
    """return a new blank state, based on model M.
     Initialise states (rates and short-term weights)  of units.
     return an object with F, C, wcij and wfij."""
    # 1. Firing rates of Feature units
    s.F = zeros((M.NF,1))
    # 2. Firing rates of Conjunction units
    s.C = asmatrix( zeros((M.NC,1)) )
    # 3. Short-term connection weights from features to conjunctions
    s.Wfc = asmatrix(random.normal(size=(s.C.size,s.F.size)))
    s.Wfc = M.rangeWfc[0]
    # 4. Short-term connection weights from conjunctions to conjunctions
    s.Wcc = asmatrix(random.normal(size=(s.C.size,s.C.size)))
    s.Wcc = M.rangeWcc[0]


################### Graph of single trial activation timecourse
def display_result(S,M, update=False, save=False):
    """ plot resulting state """
    # it is very silly that we cannot import * inside a function. 
    from matplotlib.pyplot import clf, subplot, imshow, ylabel, xlabel, legend, \
        colorbar, xticks, yticks, figure, show, xlim
    import time
    f=figure(1)
    if not update:
      clf()
      subplot(3,1,1)
      display_result.fig_f = imshow(S.F_t.T,aspect='auto',interpolation='nearest')
      ylabel('F')
      subplot(3,1,2)
      display_result.fig_c = imshow(S.C_t.T,aspect='auto', interpolation='nearest')
      ylabel('C')
      xlim((0,S.C_t.shape[0]))
      legend(['0','1','2','3','4'],loc ="lower right")
      subplot(3,4,9)
      # store the plot so it can be updated
      display_result.fig_wfc = imshow(S.Wfc)
      colorbar()
      ylabel('C')
      xlabel('F')
      xticks([]); yticks([])
      subplot(3,4,10)
      imshow(M.Wfc_l*M.gainWfcL)
      xticks([]); yticks([])
      colorbar()

      subplot(3,4,11)
      display_result.fig_wcc = imshow(S.Wcc)
      colorbar()
      ylabel('C_o')
      xlabel('C_i')
      xticks([]); yticks([])
      subplot(3,4,12)
      imshow(M.Wcc_l*M.gainWccL)
      xticks([]); yticks([])
      colorbar()
      show()
    else:
      display_result.fig_wfc.set_data( S.Wfc )
      display_result.fig_wcc.set_data( S.Wcc )
      display_result.fig_f.set_data( S.F_t.T )
      display_result.fig_c.set_data( S.C_t.T )
      f.canvas.draw()
      f.canvas.flush_events()
    if save:
      f.savefig('sunlin_test_02.svg')
      time.sleep(1e-2)

#################### Entry points

def main_one_trial_only(S=None, L=None):

    """run a single trial and display graph.
      Can pass a starting state S, if needed,
      or an input I.
     """
    M = Model()  # create Model
    # create language 
    if L is None: 
      L = Language()
	# run the language setup, based on L
    M.setup_language(L)
    I = Input(M) # create Input for this model and language
    if S is None:
      S = State(M) # initialise the state of the units
    S = single_trial(M,I,S)
    display_result(S,M)
    # convert numeric output into a sentence
    S.out_words = [ M.L.words[i] for i in  S.out ]
    return S


def main_range():
    param = {
        "gainWccL": [ 5.3, 5.5, 5.7, 5.9, 6.1 ]
        }
    num_repeats = 7
    for pkey,pvals in param.items(): # for each parameter to be varied
      out = []
      acc = []
      M = Model()
      I = Input(M)
      for pval in pvals: # for each value of that parameter
        S = State(M)
        setattr(M,pkey,pval) # set the parameter in the model
        for j in range(num_repeats):   # repeat each condition 5 times
          # run simulation without plotting
          S = single_trial(M,I,S, plot_interval=1e10)
          out.append(S.out) # store the output words
          # accuracy: what proportion of the input words are 
          # recalled in order?
          acc.append( S.acc )

    return array(acc).reshape((-1,num_repeats))


##% EEG SIM SECTION #######################################################
import sys
sys.path.append("D:\Experiment\Sanjay\Hu")
import numpy as np
import matlib


def make_eeg(S, # state of the model after simulating one trial
             lag=15,
             pre_smoothing=0,#10,
             temporal_derivative=True,
             neurons = "all",
             post_smoothing=30,
             type="energy"
             ):
  """
  type = "bipolar"
  To produce an EEG, from a single trial, we take the 
  take the activation of all the neurons, apply temporal smoothing of 
  width 10 timesteps, then take the absolute differences between all pairs 
  of neurons, to simulate many dipoles. We then apply a temporal 
  derivative of these, and add a 15 timestep lag. 
  "linear"
  "energy"
  To produce an EEG from a single trial, we add up the squared
  activation of all the neurons, apply a temporal derivative, 
  then a 30 timestep gaussian smoothing, and a 15 timestep delay. 

  neurons = "all", "conjunctive", "word"
  """
  
  if neurons == "conjunctive":
    y = S.C_t             # take conjunctive neurons only
  elif neurons == "word":
    y = S.F_t             # take feature neurons only
  else:
    y = c_[S.C_t, S.F_t]  # take all neurons

  # temporal smoothing each channel separately
  if pre_smoothing:
    y = matlib.smoothn( y , pre_smoothing, kernel='gauss' )

  if type=="energy":
    y = (y-0.)**2        # deviation or energy
    y = y.sum( axis=1 )   # sum across channels
  elif type=="bipolar":
    # compute array of differences between channels
    # - yields ( time x channels x channels ) matrix
    bipolar = y[:,:,None]-y[:,:,None].transpose((0,2,1))
    y = abs(bipolar)
    y = y.sum( axis = (1,2) )  # collapse all pairwise differences 
  elif type=="linear": 
    pass

  if temporal_derivative:
    y = np.diff( y ) # time derivative

  if post_smoothing:
    y = matlib.smoothn( y , post_smoothing,kernel='gauss' )

  # finally introduce a lag
  y = r_[nan*ones(lag), y[:-lag]]

  return y


def simulate_sentences(
             sentences,
             num_trials = 30, # trials per condition
             compute_eeg = False,
             shuffle_trials = True
             ):
  """ 
  Run simulations for each condition.

  Conditions are defined by
  sentences = { condition_name: [list of words] , ... }
  num_trials:        number of trials per condition
  compute_eeg=True:  calls make_eeg then plots the average EEGs
  shuffle_trials:    randomly interleave the conditions

  If condition names contain an underscore, the part before the underscore is a 
  condition group that is averaged and plotted together.
  """

  L = Language()       # create default language
  # collect together all words in the sentences
  # in some versions, we need to change this to 
  # __builtins__['sum']( ... )
  L.words = list(set(__builtins__.sum(sentences.values(),[]))) 
  M = Model()          # create Model
  M.setup_language(L)  # set up model weights from language

  S = State(M) # initialise the state of the units

  # create a random trial sequence
  conditions = list(sentences.keys())
  trial_sequence = list(range(len(conditions)))*num_trials
  if shuffle_trials:
    np.random.shuffle(trial_sequence) # in-place shuffle

  results = { k:[] for k in conditions } # empty lists for each result
  import copy
  #from tqdm import tqdm
  for ci in trial_sequence:
    cond = conditions[ci] # name of the current condition
    M.L.I_string = sentences[cond]
    I = Input(M) 
    S = single_trial(M,I,S,plot_interval = 1e10)  
    results[cond].append( copy.deepcopy(S) ) # store copy of state

  # compute mean accuracy for each condition
  acc = { 
    cond: np.array([ i.acc for i in results[cond] ])
    for cond in conditions
    }

  import matplotlib.pyplot as plt
  if compute_eeg:
    import collections
    # convert results into EEGs, calling make_eeg() for each trial
    # aggregate matching conditions
    cond_labels = list(set([ c.split('_')[0] for c in results.keys() ]))
    eegs = collections.defaultdict(list)
    for k,v in results.items(): # for each condition
      label = k.split('_')[0] # work out which group it belongs to
      for s in v: # for each trial in that condition
        eegs[ label ].append( make_eeg(s) ) # compute EEG and store
    eegs = {k:np.array(v) for k,v in eegs.items()} # convert to arrays
    
    # PLOT EEG
    plt.clf()
    plt.subplot(2,1,1)
    for cond,data in eegs.items(): # plot each condition's mean EEG
      matlib.errorBarPlot( data )
    plt.legend([
      [sentences[i] for i in conditions if i.startswith(l)][0]
      for l in cond_labels
      ]) # add legend for each group of conditions
    # add event markers as vertical dotted lines
    plt.plot( cumsum(I.durations)[None,:]*array([[1],[1]]), plt.ylim(),'k:')
    plt.subplot(2,1,2) 
    # plot difference between pairs of conditions:
    legends = []
    for c1 in cond_labels:
      for c2 in cond_labels:
        if c1 != c2 and c1<c2: # only plot each pair once
          plt.plot( mean(eegs[c1],axis=0) - mean(eegs[c2],axis=0) )
          legends.append( c1 + " - " + c2 )
    # add event markers as vertical dotted lines+
    plt.plot( cumsum(I.durations)[None,:]*array([[1],[1]]), plt.ylim(),'k:')
    plt.legend(legends)
    plt.show()
    savefig("eeg.svg")
    return eegs, acc
  else: # PLOT ACCURACY
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.clf()
    pd.DataFrame(acc).mean().plot(kind="bar")
    plt.show()
    return acc
##% test EEG simulation
def main_eeg():
  eegs, acc = simulate_sentences({
    'normal_0':  "R0 N0 V0 R1 N1".split(),
    'normal_1':  "R0 N1 V0 R1 N0".split(),
    'normal_2':  "R0 N0 V0 R1 N2".split(),
    'normal_3':  "R0 N2 V0 R1 N0".split(),
    'normal_4':  "R0 N1 V0 R1 N2".split(),
    'normal_5':  "R0 N2 V0 R1 N1".split(),
    'synviol_0': "R0 N0 V0 N1 R1".split(), 
    'synviol_1': "R0 N1 V0 N0 R1".split(), 
    'synviol_2': "R0 N0 V0 N2 R1".split(), 
    'synviol_3': "R0 N2 V0 N0 R1".split(), 
    'synviol_4': "R0 N1 V0 N2 R1".split(), 
    'synviol_5': "R0 N2 V0 N1 R1".split() 
    }  , compute_eeg=True , num_trials=10) 
##% compare adjectives and swapping the nouns
def main_syntaxes():
  return simulate_sentences( { 
    'normal':     "R0 N0 V0 R1 N1".split() ,
    'swapnoun':   "R0 N1 V0 R1 N0".split() ,
    'adjective':  "R0 J0 N0 V0 R1 N1".split() ,
    'repnoun':    "R0 N0 V0 R1 N0".split() ,
    'earlyviol':  "R0 V0 N0 R1 N1".split() ,
#    'lateviol':   "R0 N0 V0 N1 R1".split() #,
  }, compute_eeg=False, num_trials=10 , shuffle_trials=False)

def main_russian():
  L = Language()
  L.permitted_sentences = [
    'RS NS V', 'NS V RS', 'V RS NS',
    'V NS RS', 'NS RS V', 'RS V NS' ] # all possible orderings of a 3 word sentence
  L.permitted_sentences = []
  L.I_string = 'R0 N0 V0'.split()
  return main_one_trial_only(L=L)

  

# %%
def __main__():
  # return main_one_trial_only()
  # return main_range()
  return main_eeg()
  #return main_syntaxes()
  #return main_russian()

#%% RUN MAIN ##############################################################
result = __main__()