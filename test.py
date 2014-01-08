
import environments as e
import agents as a
import operator

ALGORITHMS = (
#	a.adp_optimistic_rewards,
	a.adp_random_exploration,
	a.adp_random_exploration_unknown,
#	a.adp_random_exploration_state,
)
MAX_ITERATIONS = (
	20, 30, 40
)
NUM_OF_TRIALS = 500
NUM_OF_TESTS = 50
ENVIRONMENTS = (
	e.simple4,
	e.simple5,
	# e.boxworld1
)

results = {}

success = {}

def test():

	for alg in ALGORITHMS:
		for env in ENVIRONMENTS:
			for maxIter in MAX_ITERATIONS:
				rewards = []
				for tst in range(NUM_OF_TESTS):
					agent = a.Agent()
					agent.learn(env, alg=alg, numOfTrials=NUM_OF_TRIALS, maxItr=maxIter)
					test_id = (alg.func_name, env.name, maxIter, NUM_OF_TRIALS, tst, )
				
					result = agent.solve(env, agent.getPolicy())
					rewards.append(1 if result[1] > 0 else 0)

					
				success[test_id] = float(sum(rewards)) / NUM_OF_TESTS
				results[test_id] = result
					
	
	from pprint import pprint
	pprint(success)

if __name__ == '__main__':
	
		test()

