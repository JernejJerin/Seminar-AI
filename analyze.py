import environments as e
import agents as a
import operator

ALGORITHMS = (
	(a.adp_optimistic_rewards, {
		e.CELJE.name: {
			'maxItr': 12,
			'R_plus': 7,
			'N_e': 4,
		},
		e.MARIBOR.name: {
			'maxItr': 20,
			'R_plus': 2,
			'N_e': 5,
		},
		e.LJUBLJANA.name: {
			'maxItr': 20,
			'R_plus': 2,
			'N_e': 5,
		}
	}),
	(a.adp_random_exploration, {
		e.CELJE.name: {
			'maxItr': 20,
			'tStep': 0.005,
			'remember': True,
		},
		e.MARIBOR.name: {
			'maxItr': 20,
			'tStep': 0.2,
			'tFac': 0.9, 
			'remember': False,
		},
		e.LJUBLJANA.name: {
			'maxItr': 20,
			'tStep': 0.005,
			'remember': True,
		}
	}),
)

STOP_AFTER_ONE = True
MAX_TESTS = 100

NUM_OF_TRIALS = 150

ENVIRONMENTS = (
	e.CELJE,
	e.MARIBOR,
	e.LJUBLJANA,
)

results = {}

def test():
	for alg, params in ALGORITHMS:
		for env in ENVIRONMENTS:
			reward = -1
			steps = 0
			while reward <= 0 and steps < (1 if STOP_AFTER_ONE else MAX_TESTS):
				steps += 1
				agent = a.Agent()
				agent.learn(env, alg=alg, numOfTrials=NUM_OF_TRIALS, **params[env.name])
				test_id = (alg, env)
				utilities, reward = agent.solve(env, agent.getPolicy())
			print env.name, alg.func_name, reward, steps

if __name__ == '__main__':
		test()
