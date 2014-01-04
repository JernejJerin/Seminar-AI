__author__ = 'Ziga Stopinsek & Jernej Jerin'
import random

class Environment():
	def __init__(self):
		raise NotImplementedError("__init__")

	def getStartingState(self):
		raise NotImplementedError("getStartingState")

	def do(self, state, action):
		raise NotImplementedError("do")

	def getActions(self, state):
		raise NotImplementedError("getActions")


class Maze(Environment):
	def __init__(
		self, size_or_image=(5, 5),
		agent_pos=(0, 0),
		end_plus_pos_list=[],
		end_minus_pos_list=[],
		stone_pos_list=[],
		steps_limit=1000
	):
		"""
		Initialization of simple maze environment.

		@param size_or_image:
		@param agent_pos:
		@param end_plus_pos_list:
		@param end_minus_pos_list:
		@param stone_pos_list:
		@param steps_limit:
		"""
		if type(size_or_image) in [str, unicode]:
			# If contains image of environment.
			self._init_from_image(size_or_image.split("\n"))
		elif len(size_or_image) == 2 and type(size_or_image[0]) == int and type(size_or_image[1]) == int:
			self.size = size_or_image
			self.stone_pos_set = sorted(set(stone_pos_list))

			# Start position is only our position.
			# Position is define for dynamic objects, in this case only our agent.
			self.start_pos = agent_pos

			# Terminal position or positive/negative positions to which the agent is
			# moved and gets positive/negative reward.
			self.end_plus_pos_set = sorted(set(end_plus_pos_list))
			self.end_minus_pos_set = sorted(set(end_minus_pos_list))
		elif hasattr(size_or_image, "__iter__"):
			self._init_from_image(size_or_image)

		# Using notation of first rows then columns.
		self.possible_actions = [
			(1, 0),  # up
			(-1, 0),  # down
			(0, -1),  # left
			(0, 1),  # right
		]
		self.possible_actions_dict = {
			"up": self.possible_actions[0],
			"down": self.possible_actions[1],
			"left": self.possible_actions[2],
			"right": self.possible_actions[3],
		}

		# Number of steps or actions executed and the limit.
		self.steps = 0
		self.steps_limit = steps_limit

	def _init_from_image(self, image):
		"""
		Initialize environment from image.

		@param image:
		@return:
		"""
		agent_pos = [None, None]
		self.stone_pos_set = set()
		self.end_plus_pos_set = set()
		self.end_minus_pos_set = set()

		def _a(i, j):
			agent_pos[0] = i
			agent_pos[1] = j

		def _p(i, j):
			self.end_plus_pos_set.add((i, j, ))

		def _n(i, j):
			self.end_minus_pos_set.add((i, j, ))

		def _s(i, j):
			self.stone_pos_set.add((i, j, ))

		fun = {
			"A": _a,
			"+": _p,
			"-": _n,
			"#": _s,
		}

		self.size = (len(image), len(image[0]), )
		for j, column in enumerate(image):
			for i, char in enumerate(column):
				if char in fun:
					fun[char](self.size[0] - j - 1, i)

		self.stone_pos_set = sorted(self.stone_pos_set)
		self.end_plus_pos_set = sorted(self.end_plus_pos_set)
		self.end_minus_pos_set = sorted(self.end_minus_pos_set)
		self.start_pos = tuple(agent_pos)

	def print_state(self, state):
		"""
		Print current state of environment.

		@param state:
		@return:
		"""
		agent_pos = state

		# Over rows.
		for i in range(self.size[0]):
			line = ""

			# Over columns.
			for j in range(self.size[1]):
				pt = (self.size[0] - i - 1, j)
				if pt == agent_pos:
					line += "A"
				elif pt in self.end_plus_pos_set:
					line += "+"
				elif pt in self.end_minus_pos_set:
					line += "-"
				elif pt in self.stone_pos_set:
					line += "#"
				else:
					line += " "
			print line

	def getStartingState(self):
		"""
		Get start position of environment.

		@return:
		"""
		return self.start_pos

	def do(self, state, action):
		"""
		Execute action on given state.

		@param state:
		@param action:
		@return:
		"""
		self.steps += 1

		# Select appropriate action given notation.
		if action in self.possible_actions_dict:
			action = self.possible_actions_dict[action]

		agent_pos = state[0]

		# Get new position with respect to action. We have already checked whether this action is possible.
		new_pos = self._add(agent_pos, action)
		reward = -0.04  # For evey additional move the agent gets negative points.
		is_terminal_state = False

		# Check if we are in one of the positive positions.
		if new_pos in self.end_plus_pos_set:
			reward = 1
			is_terminal_state = True

		# Check if we are in one of the negative positions.
		elif new_pos in self.end_minus_pos_set:
			reward = -1
			is_terminal_state = True

		# Are we over the limit step?
		if self.steps > self.steps_limit:
			reward = -1000
			is_terminal_state = True

		if is_terminal_state:
			self.steps = 0

		# First position is new position of a player.
		return new_pos, reward, is_terminal_state

	def _add(self, absolute, relative):
		"""
		Absolute = agent position, relative = executed action (i.e. (0, +\1))
		Checking if agent moves out of the environment is done outside.
		The 'intended' outcome occurs with probability 0.8, but with probability
		0.2 the agent moves at right angles to the intended direction.
		A collision with a wall results in no movement.

		@param absolute:
		@param relative:
		@return:
		"""
		new_pos = None
		rand = random.random()

		if rand > 0.2:
			# With probability of 0.8.
			new_pos = absolute[0] + relative[0], absolute[1] + relative[1]
		else:
			if relative[0] == 1:
				if 0.1 < rand <= 0.2:
					new_pos = absolute[0], absolute[1] + 1
				else:
					new_pos = absolute[0], absolute[1] - 1
			else:
				if 0.1 < rand <= 0.2:
					new_pos = absolute[0] + 1, absolute[1]
				else:
					new_pos = absolute[0] - 1, absolute[1]

		# Are we in stone?
		if new_pos in self.stone_pos_set:
			new_pos = absolute[0], absolute[1]
		return new_pos

	def getActions(self, state):
		"""
		Get all possible actions given current state.

		@param state:
		@return:
		"""
		agent_pos = state[0]
		actions = []

		for action in self.possible_actions:
			new_pos = self.add(agent_pos, action)

			# Check for wall.
			if new_pos not in self.stone_pos_set:
				continue

			# Everything is OK!
			actions.append(action)

		return actions

# The environment is mathematically defined.
example1 = Maze(
	size_or_image=(5, 5),
	agent_pos=(1, 1),
	end_plus_pos_list=[(3, 2)],
	end_minus_pos_list=[(3, 3)],
	stone_pos_list=[(i, j) for i in range(5) for j in range(5) if i == 0 or i == 4 or j == 0 or j == 4]
)

# The environment is graphically defined.
example2 = Maze([
	"######",
	"#A   #",
	"#    #",
	"#    #",
	"#   +#",
	"######"
])

# Example from the book.
example_book = Maze([
	"######",
	"#   +#",
	"# # -#",
	"#A   #",
	"######"
])

# Optimal policy for each state of agent for the example from the book.
example_book_policy = {
	(1, 1): (1, 0),
	(1, 2): (0, -1),
	(1, 3): (0, -1),
	(1, 4): (0, -1),
	(2, 1): (1, 0),
	(2, 3): (1, 0),
	(3, 1): (0, 1),
	(3, 2): (0, 1),
	(3, 3): (0, 1)
}