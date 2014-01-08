import environments as e
import agents as a
import operator

ALGORITHMS = (
#	(a.adp_optimistic_rewards, {
#		
#	}),
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
	(a.adp_random_exploration_unknown, {
		e.CELJE.name: {
			'maxItr': 20,
			'preferUnknown': 0.8,
		},
		e.MARIBOR.name: {
			'maxItr': 20,
			'preferUnknown': 0.8,
		},
		e.LJUBLJANA.name: {
			'maxItr': 20,
			'preferUnknown': 0.8,
		}	
	}),
	(a.adp_random_exploration_state, {
		e.CELJE.name: {
			'maxItr': 20,
			'logFac': 1.1,
		},
		e.MARIBOR.name: {
			'maxItr': 20,
			'logFac': 1.1,
		},
		e.LJUBLJANA.name: {
			'maxItr': 20,
			'logFac': 1.1,
		}		
	}),
)

STOP_AFTER_ONE = True
MAX_TESTS = 10

NUM_OF_TRIALS = 2000

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
