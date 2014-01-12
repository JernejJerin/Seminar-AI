import random
import environments as env
from math import log

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
	return 50. / (49 + n)

def _policy_iteration(transs, utils, policy, rewards):
	utils_i = {}

	changes = True
	while changes:
		for state in transs:
			if state not in rewards:
				continue
			estimates = max(_getEstimates(transs, utils, state))[0]
			utils[state] = rewards[state] + estimates
	
		changes = False
		for state in transs:
			estimates = []
			for ac in transs.get(state, {}):
				freq = transs.get(state, {}).get(ac, {})
				# Number of states.
				n = sum(val for val in freq.values())
				probs = dict((key, float(val) / n) for key, val in freq.iteritems())
	
				estimates.append((sum(p * utils.get(s, 0) for s, p in probs.iteritems()), ac, ))
	
			if not estimates:
				continue
			
			maxEst, maxAct = max(estimates)
	
			polEst = dict((act, est, ) for est, act in estimates)[policy.get(state, maxAct)]
	
			if maxEst > polEst or policy.get(state, None) is None:
				policy[state] = maxAct
				changes = True

def _getEstimates(transs, utils, currState, currActions=None):
	"""
	Gets estimates according to current transition states,
	utility, current state and actions that can be executed
	in current state.

	transs keys:
	currState => actions => newState
	
	For every possible action in currState
		- get frequencies newState|currState,action
		- count them: n
		- get probabilities: divide freqs with n
		- calculate estimate with bellman
	
	Return (rewardEstimate, action) pairs in a dict
	"""

	estimates = []
	for ac in (currActions or transs.get(currState, {})):
		freq = transs.get(currState, {}).get(ac, {})
		# Number of states.
		n = sum(val for val in freq.values())
		probs = dict((key, float(val) / n) for key, val in freq.iteritems())
		estimates.append((sum(p * utils.get(s, 0) for s, p in probs.iteritems()), ac, ))
	return estimates


def _getEstimatesOptimistic(transs, utils, currState, R_plus, N_e, currActions=None):
	"""
	Gets estimates for optimistic.

	Return (rewardEstimate, action) pairs in a dict
	"""

	estimates = []
	for ac in (currActions or transs.get(currState, {})):
		# We get N_s_a from transition table.
		freq = transs.get(currState, {}).get(ac, {})

		# Number of states.
		n = sum(val for val in freq.values())
		probs = dict((key, float(val) / n) for key, val in freq.iteritems())
		u = sum(p * utils.get(s, 0) for s, p in probs.iteritems())

		# This if function f from page 842.
		if n < N_e:
			estimates.append((R_plus, ac, ))
		else:
			estimates.append((u, ac, ))
	return estimates

def adp_random_exploration(env, transs={}, utils={}, freqs={}, policy={},
						   rewards={}, **kwargs):
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
	@param t: A parameter for choosing best action or random action.
	@param tStep: A step to increment parameter t.
	@param alpha: Step size function
	@param maxItr: Maximum iterations
	"""

	
	tStep = kwargs.get('tStep', 0.01)
	alpha = kwargs.get('alpha', _alpha)
	maxItr = kwargs.get('maxItr', 50)
	tFac = kwargs.get('tFac', 1.)
	t = kwargs.get('currItrs', 0)/5 if kwargs.get('remember', False) else 0
	minRnd = kwargs.get('minRnd', 0.0)
	
	itr = 0
	isTerminal = False
	state = env.getStartingState()

	# Start reward should be zero.
	reward = 0


	rewardSum = 0

	
	# Get possible actions with respect to current state.

	actions = env.getActions(state)
	_policy_iteration(transs, utils, policy, rewards)
	bestAction = policy.get(state, random.choice(actions))
	
	while not isTerminal: # while not terminal
		if random.random() < max(minRnd, 1. / (tFac*(t+1))) or bestAction is None:
			# If it is the first iteration or exploration event
			# then randomly choose an action. Taking a random action in 1/t instances.
			bestAction = random.choice(actions)
		
		# do the action with the best policy
		# or do some random exploration
		newState, reward, isTerminal = env.do(state, bestAction)
		rewards[newState] = reward
		rewardSum += reward
		
		# Set to zero if newState does not exist yet. For new state?
		freqs.setdefault(newState, 0)
		freqs[newState] += 1

		#env.printState(newState)
		
		# update transition table. The first one returns dictionary of actions for specific state and the
		# second one a dictionary of possible states from specific action (best action).
		transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
		transs[state][bestAction][newState] += 1


		actions = env.getActions(newState)
		for ac in actions:
			transs.setdefault(newState, {}).setdefault(ac, {})
		_policy_iteration(transs, utils, policy, rewards)
		
		bestAction = policy.get(newState, random.choice(actions))
		
		# Is this part from the book:
		# Having obtained a utility function U that is optimal for the learned model,
		# the agent can extract an optimal action by one-step look-ahead to maximize
		# the expected utility; alternatively, if it uses policy iteration, the
		# optimal policy is already available, so it should simply execute the
		# action the optimal policy recommends. Or should it?
		
		state = newState

		# A GLIE scheme must also eventually become greedy, so that the agent's actions
		# become optimal with respect to the learned (and hence the true) model. That is
		# why the parameter t needs to be incremented.
		t, itr = t + tStep, itr + 1
		if itr >= maxItr:
			break
	return itr, rewardSum

def adp_random_exploration_state(env, transs={}, utils={}, freqs={}, **kwargs):
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
	alpha = kwargs.get('alpha', _alpha)
	maxItr = kwargs.get('maxItr', 50)
	logFac = kwargs.get('logFac', 1.1)
	minRnd = kwargs.get('minRnd', 0.001)
	
	itr = 0
	isTerminal = False
	state = env.getStartingState()

	reward = 0
	actions = env.getActions(state)
	rewardEstimate, bestAction = None, None
	# Also not clear how this contributes to better performance.
	# if len(utils) > 0: # if this is not the first trial
	#	rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))

	while not isTerminal: # while not terminal
		t = float(len(freqs)) or 1.
		if random.random() < 1. / max(minRnd, (logFac*log(t+1))) or bestAction is None:
			# If it is the first iteration or exploration event
			# then randomly choose an action
			bestAction = random.choice(actions)

		# do the action with the best policy
		# or do some random exploration
		newState, new_reward, isTerminal = env.do(state, bestAction)

		rewards[newState] = new_reward

		# Not sure which frequency should we increment (new state or current state)?
		# When testing it works better if using new state!
		freqs.setdefault(newState, 0)
		freqs[newState] += 1

		# update transition table
		transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
		transs[state][bestAction][newState] += 1

		actions = env.getActions(state)
		rewardEstimate, bestAction = max(_getEstimates(transs, utils, state, actions))
		
		# Update utility
		utils[state] = reward + _alpha(freqs.get(state, 0)) * rewardEstimate

		new_actions = env.getActions(newState)
		rewardEstimate, bestAction = max(_getEstimates(transs, utils, newState, new_actions))

		actions = new_actions
		state = newState
		reward = new_reward

		itr += 1
		if itr >= maxItr:
			break
	return itr


def adp_optimistic_rewards(env, transs={}, utils={}, freqs={}, **kwargs):
	"""
	Active ADP (adaptive dynamic programming)

	@param env: Environment
	@param transs: A transition table (N_s'_sa) with outcome frequencies given state action pairs, initially zero.
	@param utils: Utilities table
	@param freqs: A table of frequencies (N_sa) for state-action pairs, initially zero.
	@param R_plus: An optimistic estimate of the best possible reward obtainable in any state.
	@param N_e: Limit of how many number of optimistic reward is given before true utility.
	@param alpha: Step size function
	@param maxItr: Maximum iterations
	"""
	R_plus = kwargs.get('R_plus', 5)
	N_e = kwargs.get('N_e', 12)
	alpha = kwargs.get('alpha', _alpha)
	maxItr = kwargs.get('maxItr', 10)

	itr = 0
	isTerminal = False
	state = env.getStartingState()

	# Start reward should be zero.
	reward = 0

	# Get possible actions with respect to current state.
	actions = env.getActions(state)
	rewardEstimate, bestAction = None, None
	if len(utils) > 0: # if this is not the first trial
		rewardEstimate, bestAction = max(_getEstimatesOptimistic(transs, utils, state, R_plus, N_e, actions))

	while not isTerminal: # while not terminal
		if bestAction is None:
			# If it is the first iteration or exploration event
			# then randomly choose an action. Taking a random action in 1/t instances.
			bestAction = random.choice(actions)

		# do the action with the best policy
		# or do some random exploration
		newState, new_reward, isTerminal = env.do(state, bestAction)

		# Set to zero if newState does not exist yet. For new state?
		freqs.setdefault(newState, 0)
		freqs[newState] += 1

		# update transition table. The first one returns dictionary of actions for specific state and the
		# second one a dictionary of possible states from specific action (best action).
		transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
		transs[state][bestAction][newState] += 1

		# We need to get actions on current state!
		actions = env.getActions(state)
		rewardEstimate, bestAction = max(_getEstimatesOptimistic(transs, utils, state, R_plus, N_e, actions))

		# Update utility: Bellman equation
		utils[state] = reward + _alpha(freqs.get(state, 0)) * rewardEstimate

		# Is this part from the book:
		# Having obtained a utility function U that is optimal for the learned model,
		# the agent can extract an optimal action by one-step look-ahead to maximize
		# the expected utility; alternatively, if it uses policy iteration, the
		# optimal policy is already available, so it should simply execute the
		# action the optimal policy recommends. Or should it?
		new_actions = env.getActions(newState)
		rewardEstimate, bestAction = max(_getEstimatesOptimistic(transs, utils, newState, R_plus, N_e, new_actions))

		actions = new_actions
		state = newState
		reward = new_reward

		itr += 1
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

		# Results
		self.results = []

		# Policy table
		self.policyTable = {}

		# Rewards table
		self.rewardsTable = {}

		# history
		self.history = []
		
	def getPolicy(self):
		policy = {}
		# For every state set appropriate action.
		for state in self.transTable:
			policy[state] = max(_getEstimates(self.transTable, self.uTable, state))[1]
		return policy

	def learn(self, env, alg=adp_random_exploration, numOfTrials=150, **kwargs):
		"""
		Learn best policy given the environment, algorithm and number of trials.
		@param env:
		@param alg:
		@param numOfTrials:
		"""
		
		
		itrs = 0
		self.clearExperience()
		for trial in range(numOfTrials):
			currItrs, reward = alg(env,
						transs=self.transTable,
						utils=self.uTable,
						freqs=self.nTable,
						currItrs=itrs,
						results=self.results,
						policy=self.policyTable,
						rewards=self.rewardsTable,
						**kwargs)
			itrs += currItrs

			self.history.append({
				'reward': reward,
				'steps': currItrs,
			})
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

			if energy < -1000:
				break
			# We get a list of actions that were executed and sum of rewards that were given when agent entered certain state.
		return actions, energy

"""
# lets test it on simple4
a = Agent()
a.learn(env.simple4, alg=adp_optimistic_rewards, numOfTrials=15000, **{'maxItr': 15,
			'R_plus': 7,
			'N_e': 5,})
env.simple4.printPolicy(a.getPolicy())

# get solution and print it for this simple example
solution = a.solve(env.simple4, a.getPolicy())
print "Solution steps: " + str(solution)

# print solution steps
state = env.simple4.getStartingState()
for move in solution[0]:
	env.simple4.printState(state)
	state, reward, is_terminal = env.simple4.do(state, move)
env.simple4.printState(state)
"""
