import numpy as np
from collections import namedtuple

def _do_jackknife_ttest(x, x0):
    '''
    Perform jackknifed ttest. 
    Null hypothesis is that mean(x) == x0

    x are single trial observations (np.ndarray) 
    x0 is null 
    
    create jackknifed sets of x to estimate mean and standard error,
    
    return alpha level of significance (i.e. 0.001, 0.01, 0.05, or n.s.)
    '''
    x = x.copy()
    x0 = x0
    idx = np.arange(0, len(x))
    jack_sets = _get_jack_sets(idx)

    x -= x0  # data will be centered at 0 if null hypothesis is true

    # generate jackknifed distribution of sample statistic
    obs = np.zeros(len(jack_sets))
    for i, j in enumerate(jack_sets):
        obs[i] = x[j].mean() 

    # calculate jackknife estimates of mean and standard error
    u = np.mean(obs)
    se = (((len(x) - 1) / len(x)) * np.sum((obs - u)**2)) ** (1 / 2) 

    if se > 0:
        z = u / se
    elif u == 0:
        z = 0
    else:
        # if se = 0 and u != 0, then should be significant
        z = 10

    result = namedtuple('jackknifed_ttest_result', 'statistic pvalue')
    # based on z-table
    if abs(z) > 3.3:
        return result(statistic=z, pvalue=0.001)
    elif abs(z) > 2.58:
        return result(statistic=z, pvalue=0.01)
    elif abs(z) > 1.96:
        return result(statistic=z, pvalue=0.05)
    else:
        return result(statistic=z, pvalue=np.nan)


def _get_jack_sets(idx):
    '''
    Given a set of indices, generate njacks random sets of unique observations
    '''

    jack_sets = []
    set_size = len(idx) - 1
    for j in range(len(idx)):
        roll_idx = np.roll(idx, j)
        jack_sets.append(roll_idx[:set_size])

    return jack_sets


# hierarchachal bootstrap, see: cite biorxiv paper
def get_bootstrapped_sample(variable, nboot=1000):
    '''
    This function performs a hierarchical bootstrap on the data present in 'variable'.
    This function assumes that the data in 'variable' is in the format of a 2D array where
    the rows represent the higher level (e.g. animal/recording site) and
    the number of columns represent repetitions within that level (e.g. neurons).
    '''
    bootstats = np.zeros(nboot)
    for i in np.arange(nboot):
        temp = []
        num_lev1 = np.shape(variable)[0]  # 10 animals
        num_lev2 = np.shape(variable)[1]  # 100 neurons
        rand_lev1 = np.random.choice(num_lev1,num_lev1)
        for j in rand_lev1:
            rand_lev2 = np.random.choice(num_lev2,num_lev2)

            # need to do something smart here, because not all animals have same number of neurons...


            temp.append(variable[j,rand_lev2])   # j is saying which animal, rand_lev2 are the neurons from this animal

        
        #Note that this is the step at which actual computation is performed. In all cases for these simulations
        #we are only interested in the mean. But as elaborated in the text, this method can be extended to 
        #several other metrics of interest. They would be computed here:
        bootstats[i] = np.mean(temp)
        
    return bootstats


def get_direct_prob(sample1, sample2):
    '''
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    '''
    joint_low_val = min([min(sample1),min(sample2)])
    joint_high_val = max([max(sample1),max(sample2)])
    
    p_joint_matrix = np.zeros((100,100))
    p_axis = np.linspace(joint_low_val,joint_high_val,num=100)
    edge_shift = (p_axis[2] - p_axis[1])/2
    p_axis_edges = p_axis - edge_shift
    p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

    #Calculate probabilities using histcounts for edges.

    p_sample1 = np.histogram(sample1,bins=p_axis_edges)[0]/np.size(sample1)
    p_sample2 = np.histogram(sample2,bins=p_axis_edges)[0]/np.size(sample2)

    #Now, calculate the joint probability matrix:

    for i in np.arange(np.shape(p_joint_matrix)[0]):
        for j in np.arange(np.shape(p_joint_matrix)[1]):
            p_joint_matrix[i,j] = p_sample1[i]*p_sample2[j]
            
    #Normalize the joint probability matrix:
    p_joint_matrix = p_joint_matrix/np.sum(p_joint_matrix)
    
    #Get the volume of the joint probability matrix in the upper triangle:
    p_test = np.sum(np.triu(p_joint_matrix))
    
    return p_test, p_joint_matrix