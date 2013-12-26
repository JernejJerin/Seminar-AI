import random

def _f2p(fList):
	"""
	Private method for calculating probabilities for
	each value in a given list fList
	"""
	fDict = {}
	n = len(fList)
	for val in fList:
		# calculate frequencies for each value
		pDict.setdefault(val, 0)
		pDict[val] += 1
	# return probabilities
	return dict((val, freq/n) for val, freq in fDict)

def _alpha(n):
	return 60. / (59 + n)

def _getEstimates(transs, utils, currState, currActions=None):
	# Return (rewardEstimate, action) pairs in a dict
	estimates = []
	for ac in (currActions or transs.get(currState, {})):
		probs = transs.get(currState, {}).get(ac, {})
		n = sum(val for val in probs.values())
		probs = dict((key, val/n) for key, val in probs.iteritems())
		estimates.append((sum(p*utils.get(s, 0) for s, p in probs.iteritems()), ac, ))
	return estimates


def adp_random_exploration(env, transs={}, utils={}, freqs={}, 
						   t = 1, tStep=0.05, alpha=_alpha, maxItr=100000):
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
		utility = None
		if random.random() < 1./t or bestAction is None:
			# If it is the first iteration or exploration event
			# then randomly choose an action
			bestAction = random.choice(actions)
		
		# do the action with the best policy
		# or do some random exploration
		
		newState, reward, isTerminal = env.do(state, bestAction)
		
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
		t, itr = t+tStep, itr+1
		if itr >= maxItr:
			break
	return itr

class Agent():

	def __init__(self):
		self.clearExperience()

	def clearExperience(self):
		self.nTable = {}
		self.transTable = {}
		self.uTable = {}

	def getPolicy(self):
		policy = {}
		for state in self.transTable:
			policy[state] = max(_getEstimates(self.transTable, self.uTable, state))[1]
		return policy

	def learn(self, env, alg=adp_random_exploration, numOfTrials=100):
		itrs = 0
		for trial in range(numOfTrials):
			itrs += alg(env, transs=self.transTable, 
						utils=self.uTable, 
						freqs=self.nTable)
		return self.getPolicy()

	def solve(self, env, policy):
		# solve environment with respect to policy
		actions, energy = [], 0
		state, prevState = env.getStartingState(), None
		isTerminalState = False
		while not isTerminalState:
			act = policy.get(state)
			if act is None:
				act = random.choice(env.getActions(state))
			state, reward, isTerminalState = env.do(state, act)
			
			actions.append(act)
			energy += reward
		return actions, energy

