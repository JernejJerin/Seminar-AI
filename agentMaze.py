__author__ = 'Ziga Stopinsek & Jernej Jerin'
import random
import environmentMaze as em


def _alpha(n):
	"""
	The step size function to ensure convergence. The
	function decreases as the number of times a state
	has been visited increases (n). It means that the
	utility of policy pi for state s will converge to
	correct value.
	"""
	return 60. / (59 + n)


def padp_random_exploration(env, policy, trans_table={}, util_table={}, freq_table={}, reward_table={},
	prob_table={}, t=1, tStep=0.02, alpha=_alpha, maxItr=100000
):
	"""
	Passive ADP (adaptive dynamic programming) learning
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
	@param policy:
	@param trans_table: A transition table (N_s'_sa) with outcome frequencies given state action pairs, initially zero.
	@param utils_table: Utilities table
	@param freq_table: A table of frequencies (N_sa) for state-action pairs, initially zero.
	@param t:
	@param tStep:
	@param alpha: Step size function
	@param maxItr: Maximum iterations
	"""

	# Number of iterations.
	itr = 0

	# Start at initial state.
	is_terminal = False
	current_state = env.getStartingState()
	current_reward = 0
	previous_state = None
	previous_action = None

	while not is_terminal:
		# We have not reached the terminal state.
		# Set reward for current state.
		if reward_table.get(current_state) is None:
			reward_table.setdefault(current_state, current_reward)
			util_table.setdefault(current_state, current_reward)

		if previous_state is not None:
			# Set to zero if new state and subsequent action does not exist yet.
			freq_table.setdefault(previous_state, {}).setdefault(previous_action, 0)
			freq_table[previous_state][previous_action] += 1

			# Update transition table. The first one returns dictionary of actions for specific state and the
			# second one a dictionary of possible states from specific action (best action).
			trans_table.setdefault(previous_state, {}).setdefault(previous_action, {}).setdefault(current_state, 0)
			trans_table[previous_state][previous_action][current_state] += 1

			# Update probability table for each outcome state from previous state and action.
			for outcome in trans_table.get(previous_state).get(previous_action):
				if trans_table[previous_state][previous_action][outcome] != 0:
					prob = trans_table[previous_state][previous_action][outcome] / freq_table[previous_state][previous_action]
				prob_table.setdefault(previous_state, {}).setdefault(previous_action, {})[current_state] = prob

		# Policy evaluation.
		actions = env.getActions(current_state)
		reward_estimate = -9999
		for state in util_table:
			for action in actions:
				reward = sum([prob * util_table[next_state] for (next_state, prob) in prob_table.setdefault(state, {}).
					setdefault(action, {}).iteritems() if prob is not None])
				if reward > reward_estimate:
					reward_estimate = reward
			util_table[state] = reward_table[state] + _alpha(freq_table.setdefault(state, {}).setdefault(policy.get(state[0]), 0)) * reward_estimate

		env.print_state(current_state)
		print "----------"
		print "----------"
		print "----------"

		# Execute the action specified in the policy for the given state.
		previous_state = current_state
		previous_action = policy.get(current_state[0])
		current_state, current_reward, is_terminal = env.do(current_state, policy.get(current_state[0]))

		t, itr = t + tStep, itr + 1
		if itr >= maxItr:
			break
	return itr


def _get_estimates(trans_table, util_table, previous_state, current_actions=None):
	"""
	Gets estimates according to current transition states,
	utility, current state and actions that can be executed
	in current state.

	Return (rewardEstimate, action) pairs in a dict
	"""
	estimates = []
	for action in (current_actions or trans_table.get(previous_state, {})):
		freq = trans_table.get(previous_state, {}).get(action, {})

		# Number of states.
		n = sum(val for val in freq.values())
		probs = dict((key, float(val) / n) for key, val in freq.iteritems())
		estimates.append((sum(p * util_table.get(s, 0) for s, p in probs.iteritems()), action, ))
	return estimates


# Agent class.
class Agent():
	def __init__(self):
		"""
		Initialize our agent with dictionaries.

		@return:
		"""
		# Frequency table.
		self.freq_table = {}

		# Transition table.
		self.trans_table = {}

		# Utilities table.
		self.util_table = {}

		# Reward table.
		self.reward_table = {}

		# Probability table.
		self.prob_table = {}

	def learn_utilities(self, env, policy, alg=padp_random_exploration, num_of_trials=100):
		"""
		Learn utilities of the given environment, policy algorithm and number of trials.
		@param env:
		@param policy:
		@param alg:
		@param num_o_trials:
		"""
		for trial in range(num_of_trials):
			alg(
				env, policy, trans_table=self.trans_table,
				util_table=self.util_table,
				freq_table=self.freq_table,
				reward_table=self.reward_table,
				prob_table=self.prob_table
			)
			#print self.util_table
		return self.util_table

# Create test agent.
agent = Agent()
print agent.learn_utilities(em.example_book, em.example_book_policy)

