import random


def _f2p(fList):
    """
	Private method for calculating probabilities for
	each value in a given list fList
	"""
    pDict = {}
    n = len(fList)
    for val in fList:
        # calculate frequencies for each value
        pDict.setdefault(val, 0)
        pDict[val] += 1
    # return probabilities
    return dict((val, freq / n) for val, freq in pDict)


def _alpha(n):
    """
    The step size function to ensure convergence. The
    function decreases as the number of times a state
    has been visited increases (n). It means that the
    utility of policy pi for state s will converge to
    correct value.
    """
    return 60. / (59 + n)


def _getEstimates(transs, utils, currState, currActions=None):
    """
    Gets estimates according to current transition states,
    utility, current state and actions that can be executed
    in current state.

    Return (rewardEstimate, action) pairs in a dict
    """

    estimates = []
    for ac in (currActions or transs.get(currState, {})):
        freq = transs.get(currState, {}).get(ac, {})
		# Number of states.
        n = sum(val for val in freq.values())
        probs = dict((key, val / n) for key, val in freq.iteritems())
        estimates.append((sum(p * utils.get(s, 0) for s, p in probs.iteritems()), ac, ))
    return estimates


def adp_random_exploration(env, transs={}, utils={}, freqs={},
                           t=1, tStep=0.02, alpha=_alpha, maxItr=100000):
    """
	Active ADP (adaptive dynamic programming) learning
	algorithm which returns the best policy for a given
	environment env and experience dictionary exp
	
	The experience dictionary exp can be empty if 
	the agent has no experience with the environment
	but can also be full with values from
	previous trials

	The algorithm returns the number of iterations
	needed to reach a terminal state

	For reference look in page 834.

	@param env: Environment
	@param transs: A transition table (N_s'_sa) with outcome frequencies given state action pairs, initially zero.
	@param utils: Utilities table
	@param freqs: A table of frequencies (N_sa) for state-action pairs, initially zero.
	@param t:
	@param tStep:
	@param alpha: Step size function
	@param maxItr: Maximum iterations
	"""

    # Number of iterations.
    itr = 0
    isTerminal = False
    state = env.getStartingState()

    # Get possible actions with respect to current state.
    actions = env.getActions(state)
    rewardEstimate, bestAction = None, None
    if len(utils) > 0: # if this is not the first trial
        rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))

    while not isTerminal: # while not terminal
        utility = None
        if random.random() < 1. / t or bestAction is None:
            # If it is the first iteration or exploration event
            # then randomly choose an action. Taking a random action in 1/t instances.
            bestAction = random.choice(actions)

        # do the action with the best policy
        # or do some random exploration
        newState, reward, isTerminal = env.do(state, bestAction)

        # Set to zero if newState does not exist yet.
        freqs.setdefault(newState, 0)
        freqs[newState] += 1

        env.printState(state)
        print actions, bestAction, isTerminal
        env.printState(newState)
        print "----------"
        print "----------"
        print "----------"

        # update transition table. The first one returns dictionary of actions for specific state and the
        # second one a dictionary of possible states from specific action (best action).
        transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
        transs[state][bestAction][newState] += 1

        # Getting actions on new state?
        actions = env.getActions(newState)
        rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))

        # Update utility
        utils[state] = reward + _alpha(freqs.get(state, 0)) * rewardEstimate

        state = newState
        t, itr = t + tStep, itr + 1
        if itr >= maxItr:
            break
    return itr


def adp_random_exploration_state(env, transs={}, utils={}, freqs={},
                                 alpha=_alpha, maxItr=100000):
    """
	Active ADP learning algorithm which returns the best
	policy for a given environment env and experience
	dictionary exp
	
	The experience dictionary exp can be empty if 
	the agent has no experience with the environment
	but can also be full with values from
	previous trials

	The algorithm returns the number of iterations
	needed to reach a terminal state
	"""
    itr = 0
    isTerminal = False
    state = env.getStartingState()

    actions = env.getActions(state)
    rewardEstimate, bestAction = None, None
    if len(utils) > 0: # if this is not the first trial
        rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))

    while not isTerminal: # while not terminal
        t = len(freqs) or 1
        if random.random() < 1. / t or bestAction is None:
            # If it is the first iteration or exploration event
            # then randomly choose an action
            bestAction = random.choice(actions)

        # do the action with the best policy
        # or do some random exploration

        newState, reward, isTerminal = env.do(state, bestAction)
        freqs.setdefault(newState, 0)
        freqs[newState] += 1

        env.printState(state)
        print actions, bestAction, isTerminal
        env.printState(newState)
        print "----------"
        print "----------"
        print "----------"

        # update transition table
        transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
        transs[state][bestAction][newState] += 1

        actions = env.getActions(newState)
        rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))

        # Update utility
        utils[state] = reward + _alpha(freqs.get(state, 0)) * rewardEstimate

        state = newState
        itr = itr + 1
        if itr >= maxItr:
            break
    return itr

# Agent class.
class Agent():
    def __init__(self):
        self.clearExperience()

    def clearExperience(self):
        # Frequency table.
        self.nTable = {}

        # Transition table.
        self.transTable = {}

        # Utilities table.
        self.uTable = {}

    def getPolicy(self):
        policy = {}
        # For every state set appropriate action.
        for state in self.transTable:
            policy[state] = max(_getEstimates(self.transTable, self.uTable, state))[1]
        return policy

    def learn(self, env, alg=adp_random_exploration, numOfTrials=100):
        """
		Learn best policy given the environment, algorithm and number of trials.
		@param env:
		@param alg:
		@param numOfTrials:
		"""
        itrs = 0
        for trial in range(numOfTrials):
            itrs += alg(env, transs=self.transTable,
                        utils=self.uTable,
                        freqs=self.nTable)
        return self.getPolicy()

    def solve(self, env, policy):
        # solve environment with respect to policy
        actions, energy = [], 0

        # Set state to starting state of environment.
        state, prevState = env.getStartingState(), None
        isTerminalState = False
        while not isTerminalState:
        # Policy has best actions for given state.
            act = policy.get(state)
            if act is None:
                act = random.choice(env.getActions(state))
                # Execute selected action in current state.
            state, reward, isTerminalState = env.do(state, act)

            actions.append(act)
            energy += reward
            # We get a list of actions that were executed and sum of rewards that were given when agent entered certain state.
        return actions, energy

