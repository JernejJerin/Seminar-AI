import environments as e
import agents as a
import operator
import graph

ALGORITHMS = (
	(a.adp_optimistic_rewards, {
		e.CELJE.name: {
			'maxItr': 20,
			'R_plus': 1,
			'N_e': 1,
		},
		e.MARIBOR.name: {
			'maxItr': 20,
			'R_plus': 1,
			'N_e': 1,
		},
		e.LJUBLJANA.name: {
			'maxItr': 20,
			'R_plus': 1,
			'N_e': 1,
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

NUM_OF_TRIALS = {
	e.CELJE.name: 150, #500,
	e.MARIBOR.name: 100, #2000,
	e.LJUBLJANA.name: 100, #5000,
}

ENVIRONMENTS = (
	e.CELJE,
	e.MARIBOR,
	e.LJUBLJANA,
)

graphs_funcs = {
	'reward': [
		None,
	],
	'steps': [
		None,
		lambda x, h: x if h['win'] else 0,
	],
	'length': [
		None,
		lambda x, h: x if h['win'] else 0,
	],
	'energy': [
		None,
	],
}

def test():
	for env in ENVIRONMENTS:
		agents = {}
		for alg, params in ALGORITHMS:
			reward = -1
			steps = 0
			while reward <= 0 and steps < (1 if STOP_AFTER_ONE else MAX_TESTS):
				steps += 1
				agent = a.Agent()
				agent.learn(env, alg=alg, numOfTrials=NUM_OF_TRIALS[env.name], **params[env.name])
				test_id = (alg, env)
				utilities, reward = agent.solve(env, agent.getPolicy())
				agents[alg.func_name] = agent
			print env.name, alg.func_name, reward, steps
			
		for field, funcs in graphs_funcs.iteritems():
			for f, fun in enumerate(funcs):
				filename = "%s-%s-%d.png" % (env.name, field, f, )
				title="Scalability (%s) " % (field, )
				graph.plot_agents(agents, field, show=False, fname=filename, title=title)
				
			

if __name__ == '__main__':
		test()
