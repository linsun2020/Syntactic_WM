import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn as sn
from itertools import groupby 


def sigmoid(x,sigmoid_slope): return 1/(1+np.exp(-sigmoid_slope*x)) # standard sigmoid function
def sigmoid_enc(x,sigmoid_slope): return sigmoid(x-0.5,sigmoid_slope) # sigmoid function for during encoding 
def sigmoid_rec(x,sigmoid_slope): return sigmoid(x,sigmoid_slope) # sigmoid function for recall


class in_in:
    '''Short for input_information; Class to record input information into the model.
    Each object will include information of which word or role to activate at the time step.
    Additional information can also be included.'''
    
    def __init__(self, word=None, role=None, morpheme=None, subject=False):
        self.word = word
        self.role = role
        self.morpheme = morpheme
        self.subject = subject


class feat_node:
    '''Class representing neurons.'''
    
    def __init__(self, index, sigmoid_slope, activation_decay, time_factor=1, recall=False, node_type='c', auto_gramm=False):
        self.node_type = node_type #either 'c' for role neuron, 'w' for word neuron or 'm' for morpheme neuron.
        self.index = index
        self.act = 0
        self.temp_act = 0
        self.direct_input = 0
        self.sigmoid_slope = sigmoid_slope
        self.activation_decay = activation_decay
        self.time_factor = time_factor
        self.recall = recall
        self.auto_gramm = auto_gramm


    def direct_update(self, direct_input):
        '''Direct update of activation bypassing the sigmoid function (i.e. direct activation from sensory).'''
        self.direct_input += direct_input
        
    
    def temp_update_activate(self, total_input):
        '''Updates the activation (i.e. input from other neurons).'''
        self.temp_act += total_input
        
        
    def update_decay(self):
        '''Natural decay of activation.'''
        self.act = self.act  - (self.activation_decay * self.act * self.time_factor)
        
    
    def update_activate(self):
        '''Final update of activation after collecting all the different inputs into the neuron during this timestep.'''
        # Note the total input at each timestep is filtered with the sigmoid function
        if self.auto_gramm: # The direct_input is within the sigmoid function
            if self.recall != True: # during encoding
                self.act += (sigmoid_enc(self.temp_act + self.direct_input,self.sigmoid_slope)) * self.time_factor
            else: # during recall
                self.act += (sigmoid_rec(self.temp_act + self.direct_input,self.sigmoid_slope)) * self.time_factor
        else:
            if self.recall != True: # during encoding
                self.act += (sigmoid_enc(self.temp_act,self.sigmoid_slope) + self.direct_input) * self.time_factor
            else: # during recall
                self.act += (sigmoid_rec(self.temp_act,self.sigmoid_slope) + self.direct_input) * self.time_factor
                      
        # activation has an upper bound (i.e. 1) and lower bound 0
        # this is akin to an additional sigmoid function
        self.act = np.minimum(self.act,1)
        self.act = np.maximum(self.act,0)
    
        # reset temporary activation value
        self.temp_act = 0
        self.direct_input = 0
        
        

class feat_connections:
    '''Class of connections between neurons.'''
    
    def __init__(self, parent_index, child_index, LR, max_connection, learnt_weight, long_learnt=False, connection_type='wc', floor_weight=-1, long_term_learning=False, pivot_grammar=False):
        
        self.parent_index = parent_index
        self.child_index = child_index
        self.temp_forward_learning = 0
        self.connection_type = connection_type
        self.long_learnt = long_learnt # this is whether there is any long term knowledge in this connection
        self.long_learnt_progress = 0
        self.floor_weight = floor_weight # lower bound of the connection weight
        self.long_term_learning = long_term_learning # this is whether this connection is undergoing long term learning
        self.learnt_weight = learnt_weight # weight of maximum long term knowledge      
        self.true_learnt_weight = 0    
        self.self_inhibit = 0
        self.current_learning = 0
        self.connection = self.floor_weight #combined strength of both long and short connections
        self.short_connection = self.floor_weight #strength of connection resulting from short term synaptic plasticity
        if self.parent_index == self.child_index and self.connection_type == 'cc': # Diag of W_cc matrix is fixed and unencodable
            self.connection = self.self_inhibit
        self.LR = LR
        self.max_connection = max_connection # upper bound of the connection weight   
        self.pivot_grammar = pivot_grammar      
        self.noise = 0  

        
    def update_connections(self, parent_act, child_act, input_act=-1, child_node=None):
        '''Temporary store of connection value.'''
        self.temp_forward_learning = 0  
        
        # equations for connection update
        if input_act == -1: # -1 means there is no input connection (e.g. if this is a connection between a word and role neuron)
            self.temp_forward_learning = (parent_act*child_act) * self.LR 
            self.current_learning += self.temp_forward_learning
        else: 
            self.temp_forward_learning = (parent_act*child_act*input_act) * self.LR
            self.current_learning += self.temp_forward_learning

        return self.temp_forward_learning

    
    def conjugal_update(self,conjugal_learning):
        '''Weakening of the connection.'''
        self.current_learning -= conjugal_learning
            
        return self.current_learning
    
    
    def final_update_connection(self):
        '''Updating the connection strength.'''
        self.short_connection += self.current_learning
        self.short_connection = np.maximum(self.short_connection,self.floor_weight)
        self.short_connection = np.minimum(self.short_connection,self.max_connection)
        
        if not self.pivot_grammar:
            if self.long_term_learning == True and self.connection_type == 'cc':
                # This part is for long term knowledge encoding between role neurons.
                self.long_learnt_progress += np.maximum(0,self.current_learning)
                self.long_learnt_progress = np.maximum(0,self.long_learnt_progress)
                self.long_learnt_progress = np.minimum(1,self.long_learnt_progress)
        else: # during pivot_grammar
            if self.long_term_learning == True and self.connection_type == 'wc':
                # This part is for long term knowledge encoding between role and word neurons.
                self.long_learnt_progress += self.current_learning
                self.long_learnt_progress = np.maximum(0,self.long_learnt_progress)
                self.long_learnt_progress = np.minimum(1,self.long_learnt_progress)
            elif self.connection_type == 'cc':
                self.short_connection = 0
        
        # actual learned weight is progress times maximum learning weight.
        self.true_learnt_weight = self.learnt_weight*self.long_learnt_progress
        
        # W = W_L + W_S + e
        self.connection = self.true_learnt_weight + self.short_connection + self.noise
        
        # In some cases, the connection is neither long nor short encodable.
        if self.parent_index == self.child_index and self.connection_type == 'cc':
            self.connection = self.self_inhibit
        
        # If the model is in long term learning, any short term encoding is reset back to floor at the end of the time step.
        if self.long_term_learning == True:
            self.short_connection = self.floor_weight
                
        # resetting learning of current time step    
        self.current_learning = 0
            
        return self.connection

       
        
class feature_layer:
    '''The working memory model.'''
    
    def __init__(self, n_role_neurons=10, cc_connectivity_factor=0.5, sigmoid_slope=10, activation_decay=0, LR_c=20, LR_w=1, cc_max_connection=1, cf_max_connection=2, n_word_neurons=15, time_factor=0.5, LT_wc_knowledge=None, cc_learnt_weight=0.7, cf_learnt_weight=0.7, input_node_connectivity=0, LT_cc_knowledge=None, cc_floor_weight=-1, cf_floor_weight=-1, long_term_learning=False, pivot_grammar=False, partial_long_learnt_progress=1, wernic=False, morph_nodes=None, mw_connection=False, unified_noise=0, cc_noise=False, wc_noise=False, mw_noise=False, auto_gramm=False, cf_conj_factor=2, closed_class_word_roles=None, closed_class_learnt_weight=None, mc_connectivity_factor=2.5, mm_connectivity_factor=-10, mw_max_connection=2, mw_floor_weight=-1, mw_learnt_weight=0.7, soft_constraint=False, j1=None, j2=None, k1=None, k2=None):
        
        # parameters for role neurons
        self.n_role_neurons = n_role_neurons
        self.role_neuron_dict = {}
        
        # parameters for word neurons
        self.n_word_neurons = n_word_neurons
        self.word_neuron_dict = {}
        
        # parameters for cc connections
        self.cc_conn_dict = {}
        self.cc_connectivity_factor = cc_connectivity_factor
        self.LR_c = LR_c
        self.cc_max_connection = cc_max_connection
        self.cc_floor_weight = cc_floor_weight
        self.LT_cc_knowledge = LT_cc_knowledge
        self.cc_learnt_weight = cc_learnt_weight
        
        # parameters for wc connections
        self.cf_conn_dict = {}
        self.LR_w = LR_w
        self.cf_max_connection = cf_max_connection
        self.cf_floor_weight = cf_floor_weight
        self.LT_wc_knowledge = LT_wc_knowledge
        self.cf_learnt_weight = cf_learnt_weight
        self.cf_conj_factor = cf_conj_factor
        
        # parameter for ww connection
        self.input_node_connectivity = input_node_connectivity # fixed and non-encodable
        
        # misc parameters in the equations
        self.activation_decay = activation_decay
        self.sigmoid_slope = sigmoid_slope
        self.time_factor = time_factor  
        
        # parameters for long-term learning
        self.long_term_learning = long_term_learning
        self.partial_long_learnt_progress = partial_long_learnt_progress
        self.pivot_grammar = pivot_grammar
        self.pivot_counter = 0
        
        # parameters for wernicke aphasia simulations
        self.wernic = wernic
        
        # parameter for simulating concurrent input and Broca's aphasia
        self.auto_gramm = auto_gramm
        self.closed_class_word_roles = closed_class_word_roles
        self.closed_class_learnt_weight = closed_class_learnt_weight
        
        # parameters for morpheme node simulations
        self.morph_nodes = morph_nodes
        self.morph_dict = {}
        self.morph_word_conn_dict = {}
        self.mw_connection = mw_connection
        self.mc_connectivity_factor = mc_connectivity_factor
        self.mm_connectivity_factor = mm_connectivity_factor
        self.mw_max_connection = mw_max_connection
        self.mw_floor_weight = mw_floor_weight
        self.mw_learnt_weight = mw_learnt_weight
        
        # parameters to introducing noise 
        self.unified_noise = unified_noise # magnitude of error
        self.cc_noise = cc_noise
        self.wc_noise = wc_noise
        self.mw_noise = mw_noise
        
        # temporary storage matrix
        self.temp_act_mat = np.zeros((n_role_neurons+1,n_role_neurons)) # extra row for input from outside of this layer; this mat has strength of connection accounted for

        # misc parameters for running the code
        self.recall = False
        
        # parameters for simulation S2
        self.soft_constraint = soft_constraint
        self.j1 = j1
        self.j2 = j2
        self.k1 = k1
        self.k2 = k2

        
        
        # initialise role neurons
        for n in range(self.n_role_neurons):
            self.role_neuron_dict[str(n)] = feat_node(index=n, sigmoid_slope=self.sigmoid_slope, activation_decay=self.activation_decay, time_factor=self.time_factor, node_type='c', auto_gramm=self.auto_gramm)
        
        # initialise connections between the role neurons
        for n in range(self.n_role_neurons): # parent index
            for i in range(self.n_role_neurons): # child index
                self.cc_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_c, self.cc_max_connection, self.cc_learnt_weight-self.cc_floor_weight, connection_type='cc', floor_weight=self.cc_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
                
                # Only for simulation S2
                if self.soft_constraint:
                    if n == 0 and i == 1:
                        self.cc_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_c, self.cc_max_connection, self.j1, connection_type='cc', floor_weight=self.cc_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
                    elif n == 0 and i == 2:
                        self.cc_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_c, self.cc_max_connection, self.j2, connection_type='cc', floor_weight=self.cc_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
                    
                if self.cc_noise == True:
                    self.cc_conn_dict[str(n)+str(i)].noise = np.random.random()*(self.unified_noise*2) - self.unified_noise
        
        # initialise the word nodes
        for n in range(self.n_word_neurons):
            self.word_neuron_dict[str(n)] = feat_node(index=n, sigmoid_slope=self.sigmoid_slope, activation_decay=self.activation_decay, time_factor=self.time_factor, node_type='w', auto_gramm=self.auto_gramm)
        
        # initialise the connections between the word nodes and the feature nodes
        for n in range(self.n_word_neurons): # from input
            for i in range(self.n_role_neurons): # to feature (conjunctive)
                # dictionary label is from input to feature
                self.cf_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_w, self.cf_max_connection, self.cf_learnt_weight-self.cf_floor_weight, connection_type='wc', floor_weight=self.cf_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
                
                # Only for simulation S2
                if self.soft_constraint:
                    if n == 1 and i == 1:
                        self.cf_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_w, self.cf_max_connection, self.k2, connection_type='wc', floor_weight=self.cf_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
                    elif n == 1 and i == 2:
                        self.cf_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_w, self.cf_max_connection, self.k1, connection_type='wc', floor_weight=self.cf_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
             
                if self.wc_noise == True:
                    self.cf_conn_dict[str(n)+str(i)].noise = np.random.random()*(self.unified_noise*2) - self.unified_noise   
        # if cf connections include any closed class words (to include closed class words)
        if self.closed_class_word_roles != None:
            for closed_class_word_role_pair in self.closed_class_word_roles:
                self.cf_conn_dict[str(closed_class_word_role_pair[0])+str(closed_class_word_role_pair[1])] = feat_connections(n, i, self.LR_w, self.cf_max_connection, self.closed_class_learnt_weight-self.cf_floor_weight, connection_type='wc', floor_weight=self.cf_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)
        
        # to include W^fc_L
        for n in range(len(self.LT_wc_knowledge)):
            self.cf_conn_dict[str(self.LT_wc_knowledge[n][0])+str(self.LT_wc_knowledge[n][1])].long_learnt = True
            self.cf_conn_dict[str(self.LT_wc_knowledge[n][0])+str(self.LT_wc_knowledge[n][1])].long_learnt_progress = 1
            
            if self.wernic:
                self.wernic_unified_noise = 0.005
                self.cf_conn_dict[str(self.LT_wc_knowledge[n][0])+str(self.LT_wc_knowledge[n][1])].noise = np.random.random()*(self.wernic_unified_noise*2) - self.wernic_unified_noise

        # to include W^cc_L
        if self.LT_cc_knowledge != None:
            for n in range(len(self.LT_cc_knowledge)):
                previous_node = None
                for i in self.LT_cc_knowledge[n]:
                    if previous_node != None:
                        self.cc_conn_dict[str(previous_node)+str(i)].long_learnt = True
                        self.cc_conn_dict[str(previous_node)+str(i)].long_learnt_progress = self.partial_long_learnt_progress # using self.partial_long_learnt_progress instead of 1 allows calculation of recall accuracy during long-term knowledge learning where 0 < long_learnt_progress < 1.              
                    previous_node = i
                    
        # W^cc_L between first and second role neuron is defined to be connected to mimic pivot grammar
        if self.pivot_grammar:
            self.cc_conn_dict[str(1)+str(2)].long_learnt = True
            self.cc_conn_dict[str(1)+str(2)].long_learnt_progress = 1
                           
        # initialise morpheme nodes if any.
        if self.morph_nodes != None:
            for n in self.morph_nodes:
                self.morph_dict[str(n)] = feat_node(index=n, sigmoid_slope=self.sigmoid_slope, activation_decay=self.activation_decay, time_factor=self.time_factor, node_type='m', auto_gramm=self.auto_gramm)
            
            # initialise morph to word node connections
            if self.mw_connection:
                for n in range(self.n_word_neurons): # from word nodes
                    for i in self.morph_nodes: # to morph nodes
                        self.morph_word_conn_dict[str(n)+str(i)] = feat_connections(n, i, self.LR_w, self.mw_max_connection, self.mw_learnt_weight-self.mw_floor_weight, connection_type='mw', floor_weight=self.mw_floor_weight, long_term_learning=self.long_term_learning, pivot_grammar=self.pivot_grammar)

                        if mw_noise == True:
                            self.morph_word_conn_dict[str(n)+str(i)].noise = np.random.random()*(self.unified_noise*2) - self.unified_noise
        
                        

    def input_activate(self, in_index=None):
        '''Function to directly activating a neuron.'''
        all_activated_words = []
        all_activated_morphemes = []
        current_step_multi_input = False
        
        try:
            if len(in_index) > 1:
                # More than one word activated at the same time.
                current_step_multi_input = False
                for n in in_index:
                    if n.word != None:
                        self.word_neuron_dict[str(n.word)].direct_update(0.8)
                        all_activated_words.append(n.word)

                    if n.morpheme != None:
                        try:
                            for morpheme_neuron in n.morpheme:
                                self.morph_dict[str(morpheme_neuron)].direct_update(0.8)
                                all_activated_morphemes.append(morpheme_neuron)
                        except:
                            self.morph_dict[str(n.morpheme)].direct_update(0.8)
                            all_activated_morphemes.append(n.morpheme)

        except:
            if in_index.word != None:
                self.word_neuron_dict[str(in_index.word)].direct_update(1)
                all_activated_words.append(in_index.word)
            if in_index.role != None:
                self.role_neuron_dict[str(in_index.role)].direct_update(1)
            if in_index.morpheme != None:
                try:
                    for morpheme_neuron in in_index.morpheme:
                        self.morph_dict[str(morpheme_neuron)].direct_update(1)
                        all_activated_morphemes.append(morpheme_neuron)
                except:
                    self.morph_dict[str(in_index.morpheme)].direct_update(1)
                    all_activated_morphemes.append(in_index.morpheme)
                
                
        if not self.recall and not current_step_multi_input:
            for i in range(self.n_word_neurons): # to direct input inhibition for non-activated word neurons
                if i not in all_activated_words:
                    if self.auto_gramm:
                        self.word_neuron_dict[str(i)].direct_update(-10)
                    else:
                        self.word_neuron_dict[str(i)].direct_update(-1)
        
        if not self.recall and not current_step_multi_input:
            if self.morph_nodes != None: # to direct input inhibition for non-activated morpheme neurons
                for i in self.morph_nodes:
                    if i not in all_activated_morphemes:
                        if self.auto_gramm:
                            self.morph_dict[str(i)].direct_update(-10)
                        else:
                            self.morph_dict[str(i)].direct_update(-1)


    def update_temp_activation(self):
        '''Function to consolidate all the activations resulting from connections between neurons within this layer.'''
        
        # storing activation from word neuron into each role neuron into the last row of temp_act_mat
        for n in range(self.n_role_neurons):
            for i in range(self.n_word_neurons):
                self.temp_act_mat[self.n_role_neurons,n] += self.cf_conn_dict[str(i)+str(n)].connection * self.word_neuron_dict[str(i)].act
        
        if self.morph_nodes != None:
            for n in self.morph_nodes: 
                self.temp_act_mat[self.n_role_neurons,n] += self.mc_connectivity_factor * self.morph_dict[str(n)].act # activation input from morph node to conjunctive node
                self.morph_dict[str(n)].temp_update_activate(self.mc_connectivity_factor*self.role_neuron_dict[str(n)].act) # activation input from conjunctive node to morph node
                for i in self.morph_nodes:
                        # inhibition within the morpheme layer
                        self.morph_dict[str(n)].temp_update_activate(self.mm_connectivity_factor*self.morph_dict[str(i)].act)
                if self.mw_connection:
                    for j in range(self.n_word_neurons):
                        # activation of morpheme neuron from word neurons
                        self.morph_dict[str(n)].temp_update_activate(self.morph_word_conn_dict[str(j)+str(n)].connection*self.word_neuron_dict[str(j)].act)
            
        # updating activations to and from between role neurons
        for n in range(self.n_role_neurons): #parent
            for i in range(self.n_role_neurons): #child
                self.role_neuron_dict[str(i)].temp_update_activate(self.cc_connectivity_factor*self.role_neuron_dict[str(n)].act*(self.cc_conn_dict[str(n)+str(i)].connection))
                # also update temporary store matrix
                self.temp_act_mat[n,i] = self.cc_connectivity_factor*self.role_neuron_dict[str(n)].act*(self.cc_conn_dict[str(n)+str(i)].connection)

        # update the input from word neurons to role neurons
        # this update also includes input from morpheme neurons if any.
        for n in range(self.n_role_neurons):
            self.role_neuron_dict[str(n)].temp_update_activate(self.temp_act_mat[self.n_role_neurons,n])
            
        # update the input from role neurons to word neurons
        for n in range(self.n_word_neurons):
            for i in range(self.n_role_neurons):
                self.word_neuron_dict[str(n)].temp_update_activate(self.role_neuron_dict[str(i)].act*(self.cf_conn_dict[str(n)+str(i)].connection))

        # update activation from morph neuron to word neuron
        if self.mw_connection:
            for n in range(self.n_word_neurons):
                for i in self.morph_nodes:
                    self.word_neuron_dict[str(n)].temp_update_activate(self.morph_dict[str(i)].act * self.morph_word_conn_dict[str(n)+str(i)].connection)
        
        # mutual inhibition among the word neurons
        for n in range(self.n_word_neurons):
            for i in range(self.n_word_neurons):
                self.word_neuron_dict[str(n)].temp_update_activate(self.word_neuron_dict[str(i)].act*self.input_node_connectivity)
    

    def update_connections(self):
        '''Temporary update of the connection weights between neurons based on learning rule and learning rate.'''

        for n in range(self.n_role_neurons): # parent
            for i in range(self.n_role_neurons): # child
                if not n == i:

                    # updating temporary connection store from parent to child
                    curr_learn = self.cc_conn_dict[str(n)+str(i)].update_connections(self.role_neuron_dict[str(n)].act,self.role_neuron_dict[str(i)].act, np.maximum(0,self.temp_act_mat[self.n_role_neurons,i]), self.role_neuron_dict[str(i)])

                    # inhibition update of connection between the two role neurons in the opposite direction
                    self.cc_conn_dict[str(i)+str(n)].conjugal_update(curr_learn) 

                    # inhibition update of connection from this parent role neuron to all other role neurons.
                    for j in range(self.n_role_neurons):
                        if j != i:
                            self.cc_conn_dict[str(n)+str(j)].conjugal_update(curr_learn/2)
        
        # updating connection weight changes between word and role neurons
        for n in range(self.n_word_neurons):
            for i in range(self.n_role_neurons):

                if self.wernic == False:
                    curr_learn = self.cf_conn_dict[str(n)+str(i)].update_connections(self.word_neuron_dict[str(n)].act, self.role_neuron_dict[str(i)].act, -1, self.role_neuron_dict[str(i)])
                else: # in Wernicke's aphasia, there is no rapid plasticity between word and role neurons
                    curr_learn = 0
                
#                 inhibition update of the above word neuron to all other role neurons
                for j in range(self.n_role_neurons):
                    if not j==i:
                        self.cf_conn_dict[str(n)+str(j)].conjugal_update(curr_learn/self.cf_conj_factor) 
                
#                 inhibition update of the above role neuron to all other word neurons
                for j in range(self.n_word_neurons):
                    if not j==n:
                        self.cf_conn_dict[str(j)+str(i)].conjugal_update(curr_learn/self.cf_conj_factor) 

        # if there are morpheme neurons, update connection weights between morpheme neurons and word neurons         
        if self.mw_connection:
            for n in range(self.n_word_neurons):
                for i in self.morph_nodes:
                    curr_learn = self.morph_word_conn_dict[str(n)+str(i)].update_connections(self.word_neuron_dict[str(n)].act,self.morph_dict[str(i)].act,-1,self.morph_dict[str(i)])
                    for j in self.morph_nodes:
                        if i != j:
                            self.morph_word_conn_dict[str(n)+str(j)].conjugal_update(curr_learn/2)
                    for j in range(self.n_word_neurons):
                        if n != j:
                            self.morph_word_conn_dict[str(j)+str(i)].conjugal_update(curr_learn/2)
                            
    def update_decay(self):
        '''Decay all nodes.'''
        
        for n in range(self.n_role_neurons):
            self.role_neuron_dict[str(n)].update_decay()
            
        for n in range(self.n_word_neurons):
            self.word_neuron_dict[str(n)].update_decay()
        
        if self.morph_nodes != None:
            for n in self.morph_nodes:
                self.morph_dict[str(n)].update_decay()
            
    def update_all_activation(self):
        '''Function to update all neurons and connection weights to their final state of activation at the end of this time step.'''
        
        # then update all activations
        for n in range(self.n_role_neurons):
            self.role_neuron_dict[str(n)].update_activate()
            
        for n in range(self.n_word_neurons):
            self.word_neuron_dict[str(n)].update_activate()
        
        if self.morph_nodes != None:
            for n in self.morph_nodes:
                self.morph_dict[str(n)].update_activate()

        # resetting the temporary activation matrix
        self.temp_act_mat = np.zeros((self.n_role_neurons+1,self.n_role_neurons))
        
        # also updates the final learnt connectivities at the end of this period
        for n in range(self.n_role_neurons):
            for i in range(self.n_role_neurons):
                self.cc_conn_dict[str(n)+str(i)].final_update_connection()                   
        
        for n in range(self.n_word_neurons):
            for i in range(self.n_role_neurons):
                self.cf_conn_dict[str(n)+str(i)].final_update_connection()
        
        if self.mw_connection == True:
            for n in range(self.n_word_neurons):
                for i in self.morph_nodes:
                    self.morph_word_conn_dict[str(n)+str(i)].final_update_connection()
        
        return self.role_neuron_dict,self.cc_conn_dict,self.word_neuron_dict,self.cf_conn_dict, self.morph_dict, self.morph_word_conn_dict
    
    
    
        
def data_collection(cc_conn_array, c_act_array, cf_conn_array, f_act_array, n_role_neurons, role_neuron_dict, conn_dict, word_neuron_dict, cf_conn_dict, n_word_neurons, long_encoding, cc_long_weights_list, wc_long_weights_list, morph_dict, morph_array, wm_conn_array, morph_word_conn_dict):
    '''Function to collect connection and activation data from a particular time step.'''
    
    
    if long_encoding:
        cc_long_weights = np.zeros((n_role_neurons,n_role_neurons))
        wc_long_weights = np.zeros((n_word_neurons,n_role_neurons))
      
    
    # collection of morpheme neuron activations
    if bool(morph_dict) == True:
        a = np.zeros((n_role_neurons,1))
        for n in morph_dict:
            a[int(n)] = morph_dict[n].act
        morph_array.append(a)
        
    
    # collection of rapid plasticity weights between word and morpheme neurons
    if bool(morph_word_conn_dict) == True:
        connectivity_mat = np.zeros((n_word_neurons,n_role_neurons)) # we assume max possible number of morpheme neurons is same as number of role neurons
        for n in range(n_word_neurons):
            for j in morph_dict:
                connectivity_mat[n,int(j)] = morph_word_conn_dict[str(n)+j].connection
        wm_conn_array.append(connectivity_mat)
    
    
    # this a collection of all historical matrices that records the short-term (cc_conn_array) and long-term (cc_long_weights_list) weights between role nuerons
    if not cc_conn_array == None:
        connectivity_mat = np.zeros((n_role_neurons,n_role_neurons))
        # at position i,j of matrix is connectivity of parent i to child j
        for n in range(n_role_neurons): #child
            for i in range(n_role_neurons): #parent
                connectivity_mat[i,n] = conn_dict[str(i)+str(n)].connection
                if long_encoding:
                    cc_long_weights[i,n] = conn_dict[str(i)+str(n)].long_learnt_progress                
        cc_long_weights_list.append(cc_long_weights)
        cc_conn_array.append(connectivity_mat)

        
    # collection of role neuron activations
    if not c_act_array == None:
        a = np.zeros((n_role_neurons,1))
        for n in range(n_role_neurons):
            a[n] = role_neuron_dict[str(n)].act
        c_act_array.append(a)
    
    
    # collection of connection weights (both rapid plasticity and long-term) between word and role neurons
    if not cf_conn_array == None:
        i_f_conn_mat = np.zeros((n_word_neurons,n_role_neurons))
        for n in range(n_word_neurons):
            for i in range(n_role_neurons):
                i_f_conn_mat[n,i] = cf_conn_dict[str(n)+str(i)].connection
                if long_encoding:
                    wc_long_weights[n,i] = cf_conn_dict[str(n)+str(i)].long_learnt_progress
        wc_long_weights_list.append(wc_long_weights)
        cf_conn_array.append(i_f_conn_mat)
              
            
    # collection of word neuron activations
    if not f_act_array == None:
        a = np.zeros((n_word_neurons,1))
        for n in range(n_word_neurons):
            a[n] = word_neuron_dict[str(n)].act
        f_act_array.append(a)
    
    
    return cc_conn_array, c_act_array, cf_conn_array, f_act_array, cc_long_weights_list, wc_long_weights_list, morph_array, wm_conn_array
        

    
def run_model(model, time_steps, sentence):
    '''Function to run a model for a particular number of timesteps with a specific sequence as sentence.'''
    
    n_period = int(time_steps/len(sentence)) # we will only complete steps until the last full sentence.
    time_steps = n_period * len(sentence)
    
    cc_conn_array = [] # the rapid plasticity encoding weights between role neurons
    c_act_array = [] # the role neuron activations
    cf_conn_array = [] # rapid plasticity encoding weights between word and role neurons
    f_act_array = [] # the word neuron activations
    cc_long_weights_list = [] # long-term knowledge between role neurons
    wc_long_weights_list = [] # long-term knowledge between word and role neurons
    morph_array = [] # morpheme neuron activations
    wm_conn_array = [] # the rapid plasticity encoding between word and morpheme neurons
    
    epoch = 1
    
    for period in range(n_period):
        if period%10==0:
            print('**'*10)
            
        for step in range(len(sentence)):
                
            model.input_activate(sentence[step])
            
            # Also we update the temp activation of all nodes from the previous loop
            model.update_temp_activation()
            
            # we update temp connection store based on any activation following directly after another activation
            model.update_connections()
            
            model.update_decay()
             
            role_neuron_dict, cc_conn_dict, word_neuron_dict, cf_conn_dict, morph_dict, morph_word_conn_dict = model.update_all_activation()
            
            cc_conn_array, c_act_array, cf_conn_array, f_act_array, cc_long_weights_list, wc_long_weights_list, morph_array, wm_conn_array = data_collection(   
                                                                                                                              cc_conn_array=cc_conn_array,
                                                                                                                              c_act_array=c_act_array,
                                                                                                                              cf_conn_array=cf_conn_array,
                                                                                                                              f_act_array=f_act_array,
                                                                                                                              n_role_neurons=model.n_role_neurons,
                                                                                                                              role_neuron_dict=role_neuron_dict,
                                                                                                                              conn_dict=cc_conn_dict,
                                                                                                                              word_neuron_dict=word_neuron_dict,
                                                                                                                              cf_conn_dict=cf_conn_dict,
                                                                                                                              n_word_neurons=model.n_word_neurons,
                                                                                                                              long_encoding=True,
                                                                                                                              cc_long_weights_list=cc_long_weights_list,
                                                                                                                              wc_long_weights_list=wc_long_weights_list,
                                                                                                                              morph_dict = morph_dict,
                                                                                                                              morph_array = morph_array,
                                                                                                                              wm_conn_array = wm_conn_array,
                                                                                                                              morph_word_conn_dict = morph_word_conn_dict)
                        
    
    
        print('Epoch = ' + str(epoch))
        print('Number of recorded time steps: ', str(np.array(cc_conn_array).shape[0]))
        print('**'*10)
        epoch+=1
    
    
    return role_neuron_dict,cc_conn_dict, word_neuron_dict, cf_conn_dict, model, np.array(cc_conn_array), np.array(c_act_array), np.array(cf_conn_array), np.array(f_act_array),np.array(cc_long_weights_list),np.array(wc_long_weights_list), np.array(morph_array), np.array(wm_conn_array)


    
def recall_feat_layer(model, activated_node, n_steps=500, initial_steps=5):
    '''Recall function which activates a node and then updates all activations then decays and repeats; hence it skips the update connection step.'''
    
    cc_conn_array = [] # the rapid plasticity encoding weights between role neurons
    c_act_array = [] # the role neuron activations
    cf_conn_array = [] # rapid plasticity encoding weights between word and role neurons
    f_act_array = [] # the word neuron activations
    cc_long_weights_list = [] # long-term knowledge between role neurons
    wc_long_weights_list = [] # long-term knowledge between word and role neurons
    morph_array = [] # morpheme neuron activations
    wm_conn_array = [] # the rapid plasticity encoding between word and morpheme neurons
    
    for n in range(model.n_role_neurons):
        model.role_neuron_dict[str(n)].recall=True # recall true changes the sigmoid function
        
    for n in range(model.n_word_neurons):
        model.word_neuron_dict[str(n)].recall=True # recall true changes the sigmoid function
    
    model.recall = True # only makes a difference if the 'input_node_connectivity is non-zero' causing the connection between input nodes to be non-zero      
    
    for n in range(n_steps):
        
        model.input_activate(activated_node[n])
         
        model.update_temp_activation()
        
        model.update_decay()
        
        role_neuron_dict, cc_conn_dict, word_neuron_dict, cf_conn_dict, morph_dict, morph_word_conn_dict = model.update_all_activation()
        
        cc_conn_array, c_act_array, cf_conn_array, f_act_array, cc_long_weights_list, wc_long_weights_list, morph_array, wm_conn_array = data_collection(        
                                                                                                                            cc_conn_array, 
                                                                                                                            c_act_array, 
                                                                                                                            cf_conn_array, 
                                                                                                                            f_act_array,
                                                                                                                            model.n_role_neurons, 
                                                                                                                            role_neuron_dict, 
                                                                                                                            cc_conn_dict, 
                                                                                                                            word_neuron_dict, 
                                                                                                                            cf_conn_dict,model.n_word_neurons,
                                                                                                                            long_encoding=True,
                                                                                                                            cc_long_weights_list=cc_long_weights_list,
                                                                                                                            wc_long_weights_list=wc_long_weights_list,
                                                                                                                            morph_dict = morph_dict,
                                                                                                                            morph_array = morph_array,
                                                                                                                            wm_conn_array = wm_conn_array,
                                                                                                                            morph_word_conn_dict = morph_word_conn_dict)

        
    return np.array(cc_conn_array), np.array(c_act_array), np.array(cf_conn_array), np.array(f_act_array), np.array(cc_long_weights_list), np.array(wc_long_weights_list), np.array(morph_array), np.array(wm_conn_array)




def plot_results(sentence, ceiling_cc, floor_cc, ceiling_cf, floor_cf, LT_cc_knowledge, LT_wc_knowledge, cc_history, cf_history, c_firing, f_firing, morph_act_hx=None, no_periods=1):
    '''Function to visualise firing rates and rapid synaptic plasticities.'''
    
    Wcc_track = []
    for n in range(len(LT_cc_knowledge)):
        previous_node = None
        for i in LT_cc_knowledge[n]:
            if previous_node != None:
                Wcc_track.append([previous_node,i])
            previous_node = i


    Wcf_track = []
    for n in range(len(LT_wc_knowledge)):
        previous_node = None
        for i in LT_wc_knowledge[n]:
            if previous_node != None:
                Wcf_track.append([previous_node,i])
            previous_node = i

    colors = ['xkcd:pale teal', 'xkcd:warm purple', 'xkcd:light forest green', 'xkcd:blue with a hint of purple', 'xkcd:light peach', 'xkcd:dusky purple', 'xkcd:pale mauve', 'xkcd:bright sky blue', 'xkcd:baby poop green', 'xkcd:brownish', 'xkcd:deep blue', 'xkcd:melon', 'xkcd:faded green', 'xkcd:cyan', 'xkcd:brown green', 'xkcd:purple blue', 'xkcd:greyish blue']
    colors = colors * 10
    
    if morph_act_hx is not None:
        fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(15,9.375), gridspec_kw={'width_ratios': [1, 2]})
        gs0 = axs[0, 0].get_gridspec()
        gs1 = axs[2, 0].get_gridspec()
        axbig0 = fig.add_subplot(gs0[:2, 0])
        axbig1 = fig.add_subplot(gs1[2:4, 0]) 
    else:
        fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(15,7.5), gridspec_kw={'width_ratios': [1, 2]})
        gs0 = axs[0, 0].get_gridspec()
        gs1 = axs[2, 0].get_gridspec()
        axbig0 = fig.add_subplot(gs0[:2, 0])
        axbig1 = fig.add_subplot(gs1[2:, 0])

    # remove the underlying axes
    for ax in axs[:, 0]:
        ax.remove()


    Wcc_track_handles = []
    for n in range(len(Wcc_track)):
        Wcc_track_handles.append(Line2D([0],[0],color = colors[n],label = str(Wcc_track[n])))
    axs[0,1].legend(handles = Wcc_track_handles, loc = 'upper right', fontsize = 'x-small',ncol=2)


    feat_handles=[]
    for n in range(cc_history.shape[1]):
        feat_handles.append(Line2D([0], [0], color = colors[n], label = str(n)))
    axs[1,1].legend(handles = feat_handles, loc = 'upper right', fontsize = 'x-small',ncol=2)


    Wcf_track_handles = []
    for n in range(len(Wcf_track)):
        Wcf_track_handles.append(Line2D([0],[0],color = colors[-(n+3)],label = str(Wcf_track[n])))
    axs[2,1].legend(handles = Wcf_track_handles, loc = 'upper right', fontsize = 'x-small',ncol=2)


    input_handles=[]
    for n in range(f_firing.shape[1]):
        input_handles.append(Line2D([0], [0], color = colors[-(n+3)], label = str(n)))
    axs[3,1].legend(handles = input_handles, loc = 'upper right', fontsize = 'x-small',ncol=2)
    
    if morph_act_hx is not None:
        morph_handles=[]
        for n in range(morph_act_hx.shape[1]):
            morph_handles.append(Line2D([0], [0], color = colors[-(n+3)], label = str(n)))
        axs[4,1].legend(handles = input_handles, loc = 'upper right', fontsize = 'x-small',ncol=2)


    j = 0 #start of graph
    n = j+(len(sentence)*no_periods-2) # end of graph
    # n = 110

    sn.heatmap(cc_history[n,:,:],square=False,cbar=False,ax=axbig0,annot=True,fmt='.1f',vmin=floor_cc, vmax=ceiling_cc)
    axbig0.set(ylabel='Final role-to-role connectivity')


    sn.heatmap(cf_history[n,:,:],square=False,cbar=False,ax=axbig1,annot=True,fmt='.1f',vmin=floor_cf, vmax=ceiling_cf)
    axbig1.set(ylabel='Final word-to-role connectivity')


    for i in range(len(Wcc_track)):
        sn.lineplot(x=range(n-j+1),y=cc_history[j:n+1,Wcc_track[i][0],Wcc_track[i][1]],ax=axs[0,1],color=colors[i])

    axs[0,1].set_xlim([0,n-j+1])
    axs[0,1].set_ylim(floor_cc-0.5, ceiling_cc+1)
    axs[0,1].set(ylabel='Role-to-role\nconnectivity')


    for i in range(cc_history.shape[1]):
        sn.lineplot(x=range(n-j+1),y=c_firing[j:n+1,i,0],ax=axs[1,1],color=colors[i])

    axs[1,1].set_xlim([0,n-j+1])
    axs[1,1].set_ylim(0,1.2)
    axs[1,1].set(ylabel='Role neuron\nfiring rates')


    for i in range(len(Wcf_track)):
        sn.lineplot(x=range(n-j+1),y=cf_history[j:n+1,Wcf_track[i][0],Wcf_track[i][1]],ax=axs[2,1],color=colors[-(i+3)])

    axs[2,1].set_xlim([0,n-j+1])
    axs[2,1].set_ylim(floor_cf-0.5,ceiling_cf+1)
    axs[2,1].set(ylabel='Word-to-role\nconnectivity')


    for i in range(f_firing.shape[1]):
        sn.lineplot(x=range(n-j+1),y=f_firing[j:n+1,i,0],ax=axs[3,1],color=colors[-(i+3)])

    axs[3,1].set_xlim([0,n-j+1])
    axs[3,1].set_ylim(0,1.2)
    axs[3,1].set(ylabel='Word neuron\nfiring rates', xlabel='Time steps')
    
    if morph_act_hx is not None:
        for i in range(morph_act_hx.shape[1]):
            sn.lineplot(x=range(n-j+1),y=morph_act_hx[j:n+1,i,0],ax=axs[4,1],color=colors[-(i+3)])
        axs[4,1].set_xlim([0,n-j+1])
        axs[4,1].set_ylim(0,1.2)
        axs[4,1].set(ylabel='Morpheme neuron\nfiring rates')


    fig.tight_layout()

    plt.show()



def spec_argmax(array):
    '''An Argmax function along axis 1, while checking for any neurons with equal max activation.'''
    
    final_array = np.zeros((array.shape[0],1))
    final_array[:] = np.NaN
    
    for n in range(array.shape[0]):
        
        counter = 0
        
        for j in range(array.shape[1]):

            if array[n,j] == np.max(array[n,:]):

                counter += 1

        if counter > 1:

            final_array[n,0] = -1

        else:

            final_array[n,0] = np.argmax(array[n,:])

    
    return final_array



def argmax_sentence(f_firing):
    '''Function to give the recalled output by taking the argmax of each time step and combining repeats of the same word between adjacent time steps.'''
    aa = spec_argmax(f_firing[4:,:,0])
    aa = aa.tolist()
    bb = [i[0] for i in groupby(aa)]
    return bb


def get_response_time(f_firing):
    '''Function to give the time step at which the final word of the sentence is recalled.'''
    aa = spec_argmax(f_firing[4:,:])        
    a,b = np.unique(aa, return_index = True)
    return np.max(b)+4
        
    
def long_term_encoding_plot(conn_hx_of_interest,connections_to_track):
    '''Function to plot the gradual encoding of long-term knowledge.'''

    new_conn_hx_of_interest = np.zeros((conn_hx_of_interest.shape[0],conn_hx_of_interest.shape[1]*conn_hx_of_interest.shape[2]))

    for n in range(conn_hx_of_interest.shape[0]):
        new_conn_hx_of_interest[n,:] = conn_hx_of_interest[n,:,:].flatten()

    interesting_conn = connections_to_track


    conn_index = []

    for n in range(len(interesting_conn)):
        conn_index.append(interesting_conn[n][0]*conn_hx_of_interest.shape[2]+interesting_conn[n][1])

    final_conn_hx = np.zeros((conn_hx_of_interest.shape[0],len(conn_index)))

    for n in range(len(conn_index)):
        final_conn_hx[:,n] = new_conn_hx_of_interest[:,conn_index[n]]

    final_conn_hx = np.transpose(final_conn_hx)

    a4_dims = (12*1.25, 2.5)
    fig, ax = plt.subplots(figsize=a4_dims)

    snsplt = sn.heatmap(final_conn_hx[:,:],ax=ax,cmap='rocket',vmin=0,vmax=1)

    snsplt.set_yticks([])
    snsplt.set_xticks([])
    snsplt.set(xlabel=None,ylabel=None)
