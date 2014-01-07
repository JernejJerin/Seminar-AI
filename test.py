
import environments as e
import agents as a
import operator

ALGORITHMS = (
	a.adp_optimistic_rewards,
	a.adp_random_exploration,
	a.adp_random_exploration_state,
)
MAX_ITERATIONS = (
	5, 10, 20, 50, 70, 90, 120, 150, 200, 300,
)
NUM_OF_TRIALS = 500

ENVIRONMENTS = (
	e.simple4,
)

results = {}

def test():
	for alg in ALGORITHMS:
		for env in ENVIRONMENTS:
			for maxIter in MAX_ITERATIONS:
				agent = a.Agent()

				agent.learn(env, alg=alg, numOfTrials=NUM_OF_TRIALS, maxIter=maxIter)
				test_id = (alg.func_name, env.name, maxIter, NUM_OF_TRIALS, )
				results[test_id] = agent.solve(env, agent.getPolicy())[1]
	
	from pprint import pprint
	pprint(results)

if __name__ == '__main__':
	test()
