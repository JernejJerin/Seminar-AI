class Stats:

	def __init__(self):
		self.table = {}

	def saveReward(self, algorithm, trial, reward):
		self.table[algorithm].setdefault(trial, []).append(reward)

	def plotRewardViaTime(self):
		pass
