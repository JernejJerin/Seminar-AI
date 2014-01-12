
import numpy as np
import matplotlib.pyplot as plt
import random

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
FIELDS = {
	'reward': 'Reward',
	'steps': 'Number of steps',
	'length': 'Solution length',
	'energy': 'Reward'
}

def plot_agents(agents, field,
				title='Scalability',
				show=True,
				func=None,
				fname=False):
	
	i = random.randint(0, len(COLORS) - 1)
	plt.close()
	plt.title(title)
	plt.xlabel('Number of trials')
	plt.ylabel(FIELDS[field])

	if not func:
		func = lambda x, other: x
	
	for name, agnt in agents.iteritems():
		x = np.arange(1, len(agnt.history)+1, 1)
		y = np.array([func(h[field], h) for h in agnt.history])

		plt.plot(x,y, label=name, color=COLORS[i % len(COLORS)])
		i += 1
	if fname:
		plt.savefig(fname)
	if show:
		plt.show()

