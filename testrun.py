import environments as env
import agents as ag

# we are testing it on the very simple environment Celje
a = ag.Agent()
a.learn(env.CELJE, alg=ag.adp_random_exploration, numOfTrials=150, **{'maxItr': 20,
			'tStep': 0.005,
			'remember': True,
			})

# outputs policy in graphics
print "Policy for all the possible positions of box: "
env.CELJE.printPolicy(a.getPolicy())

# get solution and print it for this simple example
solution = a.solve(env.CELJE, a.getPolicy())
print "Solution steps: " + str(solution)

# print solution steps in graphics
state = env.CELJE.getStartingState()
for move in solution[0]:
	env.CELJE.printState(state)
	state, reward, is_terminal = env.CELJE.do(state, move)
env.CELJE.printState(state)

# lets try it also on the GLIE scheme of R+ optimistic rewards
a.clearExperience()
a.learn(env.CELJE, alg=ag.adp_optimistic_rewards, numOfTrials=150, **{'maxItr': 20,
			'R_plus': 1,
			'N_e': 1,
			})

# outputs policy in graphics
print "Policy for all the possible positions of box: "
env.CELJE.printPolicy(a.getPolicy())

# get solution and print it for this simple example
solution = a.solve(env.CELJE, a.getPolicy())
print "Solution steps: " + str(solution)

# print solution steps in graphics
state = env.CELJE.getStartingState()
for move in solution[0]:
	env.CELJE.printState(state)
	state, reward, is_terminal = env.CELJE.do(state, move)
env.CELJE.printState(state)