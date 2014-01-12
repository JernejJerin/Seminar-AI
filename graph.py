
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_agents(agents, field,
				title='Scalability',
				show=True,
				fname=False):
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	i = random.randint(0, len(colors) - 1)
	plt.title(title)
	plt.xlabel('number of trials')
	plt.ylabel(field)
	for name, agnt in agents.iteritems():
		x = np.arange(1, len(agnt.history)+1, 1)
		y = np.array([h[field] for h in agnt.history])
		from pprint import pprint
		pprint(y)
		plt.plot(x,y, label=name, color=colors[i % len(colors)])
		i += 1
	if show:
		plt.show()
	if fname:
		plt.savefig(fname)
