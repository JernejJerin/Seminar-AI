class Environment():
	def __init__(self):
		raise NotImplementedError("__init__")

	def getStartingState(self):
		raise NotImplementedError("getStartingState")

	def do(self, state, action):
		raise NotImplementedError("do")

	def getActions(self, state):
		raise NotImplementedError("getActions")


class Sokoban(Environment):
	def __init__(self, size_or_image=(5, 5),
				 agentPos=(0, 0),
				 boxPosList=[],
				 endPosList=[],
				 stonePosList=[],
				 steps_limit=1000):

		if (type(size_or_image) in [str, unicode]):
			# If contains image of environment.
			self._init_from_image(size_or_image.split("\n"))
		elif len(size_or_image) == 2 and type(size_or_image[0]) == int and type(size_or_image[1]) == int:
			self.size = size_or_image
			self.stonePosSet = set(stonePosList) # optimizes search

			# Start position is our position + positions of boxes.
			# Position is define for dynamic objects, in this case agent and boxes.
			self.startPos = (agentPos, ) + tuple(sorted(boxPosList))

			# Terminal positions or positions to which the boxes need to be moved.
			self.endPosSet = set(endPosList)
		elif hasattr(size_or_image, "__iter__"): # check if iterable
			self._init_from_image(size_or_image)

		# Using notation of first columns then rows.
		self.possibleActions = [
			(0, +1), # up
			(0, -1), # down
			(-1, 0), # left
			(+1, 0), # right
		]
		self.possibleActionsDict = {
			"up": self.possibleActions[0],
			"down": self.possibleActions[1],
			"left": self.possibleActions[2],
			"right": self.possibleActions[3],
		}

		m = self.size[0] - 1
		n = self.size[1] - 1

		# define environment corners for terminal state detection
		self.envCorners = {(0, 0, ), (0, n, ), (m, 0, ), (m, n, )}

		# Number of steps or actions executed and the limit.
		self.steps = 0
		self.steps_limit = steps_limit

	def _init_from_image(self, image):
		boxPosList = [] # TODO: sort?
		agentPos = [None, None, ]
		self.stonePosSet = set()
		self.endPosSet = set()
		def _a(i, j):
			agentPos[0] = i
			agentPos[1] = j
		def _b(i, j):
			boxPosList.append((i, j, ))
		def _s(i, j):
			self.stonePosSet.add((i, j, ))
		def _e(i, j):
			self.endPosSet.add((i, j, ))
		fun = {
			"A": _a,
			"B": _b,
			"*": _e,
			"#": _s,
		}
		self.size = (len(image[0]), len(image), )
		for j, row in enumerate(image):
			for i, char in enumerate(row):
				if char in fun:
					fun[char](i, self.size[1] - j - 1)

		self.startPos = (tuple(agentPos), ) + tuple(sorted(boxPosList))
		
	def printState(self, state):
		# In the beginning state is self.startPos!
		agentPos = state[0]
		boxPosSet = set(state[1:]) # set optimizes search

		# Print columns before last row.
		print "|" + "".join("-" for i in range(self.size[0])) + "|"

		# Over columns.
		for j in range(self.size[0]):
			line = "|"

			# Over rows.
			for i in range(self.size[1]):
				# Change column and row order.
				pt = (i, self.size[1] - 1 - j)
				if pt == agentPos:
					line += "A"
				elif pt in boxPosSet:
					line += "B"
				elif pt in self.endPosSet:
					line += "*"
				elif pt in self.stonePosSet:
					line += "#"
				else:
					line += " "
			print line + "|"

		# Print columns after last row.
		print "|" + "".join("-" for i in range(self.size[0])) + "|"


	def getStartingState(self):
		return self.startPos

	def do(self, state, action):
		self.steps += 1
		# Select appropriate action given notation.
		if action in self.possibleActionsDict:
			action = self.possibleActionsDict[action]

		agentPos = state[0]
		boxPosSet = set(state[1:]) # set optimizes search
		newPos = self._add(agentPos, action) # get new position with respect to action. We have already checked whether this action is possible.
		boxList = [] # new box positions
		reward = -10 # agent reward. For evey additional move the agent gets negative points.
		isTerminalState = False
		newBoxPos = None

		# After player move check for new positions of boxes.
		if newPos in boxPosSet:
			for boxPos in boxPosSet:
				# check if new position moves a box
				if boxPos == newPos:
					newBoxPos = self._add(boxPos, action)

					# Reward for moving a box.
					reward = 10
					if newBoxPos in self.endPosSet:
						# If new position is in end position then we give greater reward.
						reward = 30
					boxList.append(newBoxPos)
				else:
					boxList.append(boxPos)
		else:
			boxList = tuple(boxPosSet)

		# Finding deadlocks. We will first try without this.
		# _deadlock_detection(newBoxPos, boxList)

		boxInEndPosCount = sum(box in self.endPosSet for box in boxList)

		# check if we are finished
		if boxInEndPosCount == len(self.endPosSet):
			reward = 1000
			isTerminalState = True

		if self.steps > self.steps_limit:
			reward = -1000
			isTerminalState = True
		# First position is new position of a player.
		return ((newPos, ) + tuple(sorted(boxList)),
				reward,
				isTerminalState, )

	# Inside borders of box. Rename to location for clarity.
	def _in_borders(self, loc):
		return 0 <= loc[0] < self.size[0] and 0 <= loc[1] < self.size[1]

	def _in_corner(self, box):
		"""
		Check if box is in corner. There can be more than four corners
		in environment! That is why this function checks corners only
		for single box in next order (NE = up-right, SE = right-down, SW = down-left,
		NW = left-up). This can be helpful as in if box is in corner but
		not in position, then the game cannot be solved.
		"""
		return box in self.envCorners or \
			   self._add(box, self.possibleActions[0]) in self.stonePosSet and self._add(box, self.possibleActions[3]) in self.stonePosSet or \
			   self._add(box, self.possibleActions[3]) in self.stonePosSet and self._add(box, self.possibleActions[1]) in self.stonePosSet or \
			   self._add(box, self.possibleActions[1]) in self.stonePosSet and self._add(box, self.possibleActions[2]) in self.stonePosSet or \
			   self._add(box, self.possibleActions[2]) in self.stonePosSet and self._add(box, self.possibleActions[0]) in self.stonePosSet

	def _on_edge(self, loc):
		"""
		Check if location loc=(x,y) is on environment edge
		"""

		# Columns and rows count starts at 0!
		return sum((loc[i] in {0, self.size[i]-1}) for i in range(2)) > 0

	def _add(self, absolute, relative):
		# Absolute = agent position or box position, relative = executed action (i.e. (0, +1))
		# Checking if agent moves out of the environment is done outside.
		return (absolute[0] + relative[0], absolute[1] + relative[1])

	def _deadlock_detection(self, newBoxPos, boxList):
		"""
		Detecting deadlock positions in sokoban. Currently not used!
		"""
		# IN THIS SECTION WE SHOULD FIRST TRY TO FIND DEADLOCKS!
		# There are three papers:
		# http://www.lamsade.dauphine.fr/~cazenave/papers/sokoban.pdf
		# http://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/ai98_soko.pdf
		# http://weetu.net/Timo-Virkkala-Solving-Sokoban-Masters-Thesis.pdf
		# I am inclining towards last (master thesis), where on page 44 there is a description
		# of simple deadlock detection in O(1) time.

		# Check for situations where box cannot be moved any more to any of the possible end positions.
		# First we start with corner position.
		if newBoxPos is not None and newBoxPos not in self.endPosSet and self._in_corner(newBoxPos):
			reward = -1000
			isTerminalState = True

		# How many boxes are in end position and on edge of environment?
		boxOnEdgeCount = sum(self._on_edge(loc) for loc in boxList)
		# Check if the number of boxes on edges
		# exceeds number of ends on edge
		if boxOnEdgeCount > sum(self._on_edge(loc) for loc in self.endPosSet):
			# When a box is on the edge it cannot be moved to the center
			# and that is why the game is over
			# TODO: Jernej, check if this is a fact
			# This is true. You cannot move the box away from edge but you can move it along the edge.
			# And this of course only applies if you are not at the corner. Then you are stuck.
			# TODO: But we should check for every edge (four edges) if the number of
			# TODO: end positions for each edge does not exceed the number of boxes in each edge.
			reward = -1000
			isTerminalState = True

	def getActions(self, state):
		agentPos = state[0]
		boxPosSet = set(state[1:]) # set optimizes search

		# check if possible
		actions = []

		# Possible actions (0, +1), (0, -1), etc.
		for diff in self.possibleActions:
			newPos = self._add(agentPos, diff)
			if not self._in_borders(newPos) or newPos in self.stonePosSet:
				continue
			if newPos in boxPosSet: # we are moving a box
				newBoxPos = self._add(newPos, diff)

				# Remove position of the moved box. The position matches the new position of agent.
				otherBoxes = boxPosSet - {newPos, } # get other boxes
				if newBoxPos in otherBoxes:
					# you cannot move two boxes in same time
					continue
				if not self._in_borders(newBoxPos) or newBoxPos in self.stonePosSet:
					continue
			# everything is OK
			actions.append(diff) # TODO: check if absolute actions are better
		# Returns all possible actions for current state of environment.
		return actions


simple1 = Sokoban((5,5), 
				  agentPos=(1,1),
				  boxPosList=[(1,2)],
				  endPosList=[(2,1)],
				  stonePosList=[(2, 2)])

simple2 = Sokoban((10, 10), 
				  agentPos=(1,1),
				  boxPosList=[(1,2)],
				  endPosList=[(2,1)],
				  stonePosList=[(2, 2)])

# Check for corner detection.
simple3 = Sokoban((5, 5),
				  agentPos=(0,0),
				  boxPosList=[(3,3)],
				  endPosList=[(4,1)],
				  stonePosList=[(2, 2), (2, 3), (3, 2)])

simple4 = Sokoban([
	"######",
	"#A   #",
	"# B  #",
	"#    # ",
	"#*  ##",
	"######"
])
