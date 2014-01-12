import random
import environmentMaze as env


def _alpha(n):
	"""
	The step size function to ensure convergence. The
	function decreases as the number of times a state
	has been visited increases (n). It means that the
	utility of policy pi for state s will converge to
	correct value.
	"""
	return 50. / (49 + n)


def _getEstimates(transs, utils, currState, R_plus=None, N_e=None, currActions=None):
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
		# We get N_s_a from transition table.
		freq = transs.get(currState, {}).get(ac, {})

		# Number of states.
		n = sum(val for val in freq.values())
		probs = dict((key, float(val) / n) for key, val in freq.iteritems())
		u = sum(p * utils.get(s, 0) for s, p in probs.iteritems())

		# This if function f from page 842. Otherwise we are doing normal estimation.
		# It means if the number of actions a that were executed in state s is not high enough,
		# it means we should set some optimistic reward to search more into that direction.
		if R_plus is not None and N_e is not None and n < N_e:
			estimates.append((R_plus, ac, ))
		else:
			estimates.append((u, ac, ))
	return estimates


def _policy_iteration(transs, utils, policy, rewards, R_plus=None, N_e=None, th=1):
	changes = True
	while changes:
		for state in transs:
			if state not in rewards:
				continue
			estimates = max(_getEstimates(transs, utils, state, R_plus, N_e))[0]
			utils[state] = rewards[state] + th * estimates

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
	rewardSum = 0

	# Get possible actions with respect to current state.
	actions = env.getActions(state)
	_policy_iteration(transs, utils, policy, rewards, th=alpha(itr))
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
		
		# update transition table. The first one returns dictionary of actions for specific state and the
		# second one a dictionary of possible states from specific action (best action).
		transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
		transs[state][bestAction][newState] += 1

		actions = env.getActions(newState)
		for ac in actions:
			transs.setdefault(newState, {}).setdefault(ac, {})
		_policy_iteration(transs, utils, policy, rewards, th=alpha(itr))
		
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


def adp_optimistic_rewards(env, transs={}, utils={}, freqs={}, policy={},
						   rewards = {}, **kwargs):
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
	rewardSum = 0

	# Get possible actions with respect to current state.
	actions = env.getActions(state)
	_policy_iteration(transs, utils, policy, rewards, R_plus=R_plus, N_e=N_e, th=alpha(itr))
	bestAction = policy.get(state, random.choice(actions))

	while not isTerminal: # while not terminal
		if bestAction is None:
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

		# update transition table. The first one returns dictionary of actions for specific state and the
		# second one a dictionary of possible states from specific action (best action).
		transs.setdefault(state, {}).setdefault(bestAction, {}).setdefault(newState, 0)
		transs[state][bestAction][newState] += 1

		# We need to get actions on new state.
		actions = env.getActions(newState)
		for ac in actions:
			transs.setdefault(newState, {}).setdefault(ac, {})
		_policy_iteration(transs, utils, policy, rewards, R_plus=R_plus, N_e=N_e, th=alpha(itr))

		#rewardEstimate, bestAction = max(_getEstimatesOptimistic(transs, utils, state, R_plus, N_e, actions))
		bestAction = policy.get(newState, random.choice(actions))
		state = newState

		itr += 1
		if itr >= maxItr:
			break
	return itr, rewardSum


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


# we are testing it on the example from the book
a = Agent()
a.learn(env.example_book, alg=adp_random_exploration, numOfTrials=100, **{'maxItr': 20,
			'tStep': 0.01,
			'remember': True,
			})

# the results are similar to the results on page 840
print a.getPolicy()
print a.uTable
env.example_book.print_policy(a.getPolicy())

# lets try it also on the reward algorithm
a.clearExperience()
a.learn(env.example_book, alg=adp_optimistic_rewards, numOfTrials=100, **{'maxItr': 20,
			'R_plus': 1,
			'N_e': 8,
			})

print a.getPolicy()
print a.uTable
env.example_book.print_policy(a.getPolicy())

