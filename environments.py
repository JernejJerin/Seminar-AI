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
	def __init__(self, size=(5, 5), 
				 agentPos=(0, 0), 
				 boxPosList=[], 
				 endPosList=[], 
				 holePosList=[],
				 stonePosList=[]):
		self.size = size
		self.stonePosSet = set(stonePosList) # optimizes search

        # Start position is our position + positions of boxes.
        # Position is define for dynamic objects, in this case agent and boxes.
		self.startPos = (agentPos, ) + tuple(boxPosList)

        # Terminal positions or positions to which the boxes need to be moved.
		self.endPosSet = set(endPosList)

        # Hole?
		self.holePosSet = set(holePosList)

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

        # Count for end position.
		self.endOnEdgeCount = sum(self._on_edge(loc) for loc in self.endPosSet)

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
				elif pt in self.holePosSet:
					line += "o"
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
        # Select appropriate action given notation.
		if action in self.possibleActionsDict:
			action = self.possibleActionsDict[action]

		agentPos = state[0]
		boxPosSet = set(state[1:]) # set optimizes search
		newPos = self._add(agentPos, action) # get new position with respect to action
		boxList = [] # new box positions
		reward = -10 # agent reward
		isTerminalState = False

		# After player move check positions for boxes.
		for boxPos in boxPosSet:
			newBoxPos = boxPos
			# check if new position moves a box
			if boxPos == newPos:
                # Move box.
				newBoxPos = self._add(boxPos, action)
				reward = 30
			boxList.append(newBoxPos)

        # How many boxes are in end position and on edge of environment?
		boxInEndPosCount = sum(box in self.endPosSet for box in boxList)
		boxOnEdgeCount = sum(self._on_edge(loc) for loc in boxList)
		# Check if the number of boxes on edges
		# exceeds number of ends on edge
		if boxOnEdgeCount > self.endOnEdgeCount:
			# When a box is on the edge it cannot be moved to the center
			# and that is why the game is over
			# TODO: Jernej, check if this is a fact
            # This is true. You cannot move the box away from edge but you can move it along the edge.
            # And this of course only applies if you are not at the corner. Then you are stuck.
            # TODO: But we should check for every edge (four edges) if the number of
            # TODO: end positions for each edge does not exceed the number of boxes in each edge.
			reward = -1000
			isTerminalState = True

		# check if we are finished
		if boxInEndPosCount == len(self.endPosSet):
			reward = 1000
			isTerminalState = True
        # First position is new position of a player.
		return ((newPos, ) + tuple(boxList), 
				reward, 
				isTerminalState, )

    # Inside borders of box. Rename to location for clarity.
	def _in_borders(self, loc):
		return 0 <= loc[0] < self.size[0] and 0 <= loc[1] < self.size[1]

	def _on_edge(self, loc):
		"""
		Check if location loc=(x,y) is on environment edge
		"""

        # Columns and rows count starts at 0!
		return sum((loc[i] in {0, self.size[i]-1}) for i in range(2)) > 0

	def _add(self, absolute, relative):
        # Absolute = agent position or box position, relative = executed action (i.e. (0, +1))
        # Checking if agent moves out of the box is done outside.
		return (absolute[0] + relative[0], absolute[1] + relative[1])

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
				  holePosList=[], 
				  stonePosList=[(2, 2)])

simple2 = Sokoban((10, 10), 
				  agentPos=(1,1), 
				  boxPosList=[(1,2)], 
				  endPosList=[(2,1)], 
				  holePosList=[], 
				  stonePosList=[(2, 2)])
