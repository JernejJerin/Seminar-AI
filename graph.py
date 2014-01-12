
import numpy as np
import matplotlib.pyplot as plt


def plot_agents(agents, field):

	for a in agents:
		x = np.arange(1, len(a.history)+1, 1)
		y = np.array([h[field] for h in a.history])

		plt.plot(x,y)
	plt.show()
