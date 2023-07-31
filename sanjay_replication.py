# Simulate syntax in working memory
# coded from Sun Lin methods by SGM 2022.06.10
# fixed by Sun Lin 20/6/22
# param search
# 6/2023 - with different syntaxes
# 7/2023 - use strings as input
# 8/2023 - create EEGs
#%% 

from numpy import *    # make life easy
from pdb import pm

"""
Overall structure
  - class Model - model parameters and long-term weights.
  - class Language - words and syntax as strings.
  - class Input - the sequence of input vectors for the model and language
  - class State - the neural network activations and short-term weights


 1) Create a Model(), then a Language()
 2) Call model.setup_language(l) to store the language in the model's synapses.
 3) Create an Input(model) based on the language
 4) Create a blank State(model)
 5) Call single_trial(model, input, state) to run one trial

"""

class Language:
  """ Syntax and words of a language """
  def __init__(L):
    """		Basic default language definition:    """
    # words begin with a word class and an identifier.
    # e.g. R=article, N=noun, J=adjective
    L.words = 'R0 N0 V0 R1 N1 J0 J1'.split()

    # roles begin with a letter indicating the word class
    # followed by a specifier. e.g.
					# 'NO' means noun in the object position.
    L.roles = 'RS NS V RO NO JS'.split()

    # allow these role sequences:
    L.permitted_sentences = [
      'RS NS V RO NO',
      'RS JS NS V RO NO'
    ]

    # Input string
    L.I_string = 'R0 N0 V0 R1 N1'.split()


class Model:
  def __init__( M, low_accuracy = False):
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
    M.gainWccL  = 5.7    # gain of long-term weights
    M.gainWfcL  = 1.7
    M.rangeWfc  = (-1,2) # constrain short term weights to this range
    M.rangeWcc  = (-5,1)
    M.noiseC    = 0.02
    M.recallBias= 0.5    # change in gain at recall
    M.dt        = 0.5    # timestep size

    # three useful sigmoid functions
    M.lim_01  = lambda x: maximum(0,minimum(1,x))
    M.sigmoid = lambda x,k: 1/(1+exp(-10*(x-k)))
    M.lim_to  = lambda x,r: maximum(r[0], minimum(r[1],x))

  def setup_language(M,L):
    """
    # 3. Set up synapses based on this specified language
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
    # long-term word-to-word inhibition
    M.Wff = ones((M.NF,M.NF))

class Input:
  def __init__(I,M):
    """
    4. Setup Input sequence - creates input I for a model,
       based on the model M and language M.L
    """
    string = M.L.I_string # get words to use as input

    # duration for each stage of the input
    # 1 timestep of -1, then 50 ts for each word.
    # then 100 ts delay, 5 ts retrieval cue, then 200 ts retrieval time.
    I.durations = [1] + [50]*len(string) + [100,  5, 200]
    # calculate the time at start of recall
    I.recall_time = sum(I.durations[:-2])

    # Stages of input to the feature units (each row is one stage)
    # Each F unit receives input to keep it at the given value.
    # NaN means no external input to the network
    # Last column means "recall phase".
    #   2 = stimulate c[0]. >0 = use recall bias.
    # These get put into S.stim, at the appropriate timesteps.

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

def do_one_timestep(S,M):
    """update the state S with one time step
       S = current model state. Must contain the current input, as S.stim.
       M = model parameters
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
        S = do_one_timestep(S,M)
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
    S.acc   = slide_match( S.out, I.word_indices ) # best accuracy
    
    return S

def slide_match(x,y):
  """ slide two 1-dimensional arrays along each other to find best match.
  x = large array, in which to find y.
  Example: if x = [1,2,3,4,5,6,7,8,9,10] and y = [3,4,5], then the best match
  is at position 2, where x[2:5] = [3,4,5], and the match proportion is 1.0.
  If x = [1,2,3,4,5,6,7,8,9,10] and y = [7,6,5], then the best match is at
  any of positions 6,7,8, e.g. where x[6] = 6, and the match proportion is 1/3.
  """
  x=array(x) # ensure numpy arrays
  y=array(y)
  assert(len(x.shape)==1) # ensure 1 dimensional
  assert(len(y.shape)==1)
  if len(x)<len(y): # ensure x is at least as large as y
    x = r_[x, nan*ones(len(y)-len(x))]
  # how many different positions are there, if you slide y along x?
  num_slides = len(x)-len(y) + 1 
  # empty array for calculating the proportion of words that match, 
  # for each slide potition.
  match_prop = nan*zeros(num_slides)
  # for each sliding position
  for i in range( num_slides ):
    # calculate what proportion of the words in y match the words in x
    match_prop[i] = mean( x[i:(i+len(y))] == y )
  return match_prop.max()


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

#################### Entry point

def main_one_trial_only(S=None, L=None):

    """run a single trial and display graph.
      Can pass a starting state S, if needed,
      or an input I.
     """
    M = Model()  # create Model
    # create language and save it within the model
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


#%% EEG SIM SECTION #######################################################
import sys
sys.path.append("D:\Experiment\Sanjay\Hu")
import matlib
import numpy as np


def make_eeg(S, # state of the model after simulating one trial
             lag=15,
             pre_smoothing=10,
             temporal_derivative=True,
             neurons = "all",
             post_smoothing=30,
             type="bipolar"
             ):
  """
  To produce an EEG, from a single trial, we take the 
  take the activation of the neurons, apply temporal smoothing of 
  width 10 timesteps, then take the absolute differences between all pairs 
  of neurons, to simulate many dipoles. We then apply a temporal 
  derivative of these, and add a 15 timestep lag. 
  type = "bipolar", "linear", "energy".
  neurons = "all", "conjunctive", "word"
  """
  
  if neurons == "conjunctive":
    y = S.C_t             # take conjunctive neurons only
  elif neurons == "word":
    y = S.F_t             # take feature neurons only
  else:
    y = c_[S.C_t, S.F_t]  # take all neurons

  # temporal smoothing each channel separately
  y = matlib.smoothn( y , pre_smoothing, kernel='gauss' )

  if type=="energy":
    y = (y-0.2)**2        # deviation or energy
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

  y = matlib.smoothn( y , post_smoothing,kernel='gauss' )

  # finally introduce a lag
  y = r_[nan*ones(lag), y[:-lag]]

  return y


def simulate_sentences(
             sentences,
             num_trials = 30, # trials per condition
             compute_eeg = False
             ):
  """ 
  Run a simulation for each condition, and plot the EEG.
  Conditions are defined by sentences = { condition_name: [list of words] , ... }.
  """
  import copy

  L = Language()       # create default language
  M = Model()          # create Model
  M.setup_language(L)  # set up model weights from language

  S = State(M) # initialise the state of the units

  # create a random trial sequence
  conditions = list(sentences.keys())
  trial_sequence = list(range(len(conditions)))*num_trials
  np.random.shuffle(trial_sequence) # in-place shuffle

  results = { k:[] for k in conditions } # empty lists for each result
  for ci in trial_sequence:
    cond = conditions[ci] # name of the current condition
    M.L.I_string = sentences[cond]
    I = Input(M) 
    S = single_trial(M,I,S,plot_interval = 1e10)  
    results[cond].append( copy.deepcopy(S) )

  # compute mean accuracy for each condition
  acc = { 
    cond: mean(np.array([ i.acc for i in results[cond] ]))
    for cond in conditions
    }

  import matplotlib.pyplot as plt
  if compute_eeg:
    # convert results into EEGs, calling make_eeg() for each trial
    eegs = {
      cond: np.array([ make_eeg(r) for r in results[cond] ])
      for cond in conditions
      }
    
    # PLOT EEG
    plt.clf()
    plt.subplot(2,1,1)
    for cond,data in eegs.items(): # plot each condition's mean EEG
      matlib.errorBarPlot( data )
    plt.legend([sentences[i] for i in conditions])
    plt.plot( cumsum(I.durations)[None,:]*array([[1],[1]]), plt.ylim(),'k:')
    plt.subplot(2,1,2) 
    # plot difference between conditions 1 and 0:
    plt.plot( mean(eegs[conditions[0]],axis=0) - mean(eegs[conditions[1]],axis=0) )
    # add event markers as vertical dotted lines
    plt.plot( cumsum(I.durations)[None,:]*array([[1],[1]]), plt.ylim(),'k:')
    return eegs, acc
  else: # PLOT ACCURACY
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.clf()
    pd.DataFrame(acc, index=[0]).plot(kind="bar")
    plt.show()
    return acc
#%% test EEG simulation
def test_eeg():
  eegs, acc = simulate_sentences({
    'normal':  "R0 N0 V0 R1 N1".split(),
    'synviol': "R0 N0 V0 N1 R1".split() 
    }  , compute_eeg=True ) 
#%% compare adjectives and swapping the nouns
def test_syntaxes():
  return simulate_sentences( { 
    'normal':     "R0 N0 V0 R1 N1".split() ,
    'swapnoun':   "R0 N1 V0 R1 N0".split() ,
    'adjective':  "R0 J0 N0 V0 R1 N1".split(),
    'repnoun':    "R0 N0 V0 R1 N0".split() ,
    'earlyviol':  "R0 V0 N0 R1 N1".split() ,
    'lateviol':   "R0 N0 V0 N1 R1".split() ,
  }, compute_eeg=False, num_trials=100 )

# %%
run_profile = True
def __main__():
  # return main_one_trial_only()
  # return main_range()
  # return test_eeg()
  if run_profile:
    import cProfile
    cProfile.run('test_syntaxes()','syntax.prof')
    import pstats
    stats = pstats.Stats('syntax.prof')
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
#%% RUN MAIN ##############################################################
result = __main__()