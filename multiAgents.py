# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def manhattanDistance(position1, position2):
        "Calculates the Manhattan distance between 2 points"
        xy1 = position1
        xy2 = position2
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        closestX = 9999
        closestY = 9999
        pointDistance = -1
        lowestDistance = -1

        # find the food pellet closest to the current Pacman position
        food = gameState.getFood()
        for rowIndex in xrange(0, food.width):
            for colIndex in xrange(0, food.height):
                if food[rowIndex][colIndex]:
                    pointDistance = manhattanDistance([rowIndex, colIndex], gameState.getPacmanPosition())
                    if lowestDistance == -1 or lowestDistance > pointDistance:
                        lowestDistance = pointDistance
                        closestX = rowIndex
                        closestY = colIndex

        self.closestFood = tuple([closestX, closestY])


        # find the index at which the "Stop" move is placed
        stopIndex = -1
        if("Stop" in legalMoves):
            stopIndex = legalMoves.index("Stop")

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        negativeScores = sum(score < 0 for score in scores)
        # Give a low score to Stop index unless there is a ghost nearby
        if scores[stopIndex] > 0 and negativeScores <= (len(scores) - 1):
            scores[stopIndex] *= -1

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)


        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print "newPos=", newPos
        # print "newScaredTimes="
        # print newScaredTimes
        # print "newFood="
        # print newFood[0]
        # print "successorGameState.getScore()=", successorGameState.getScore()
        # print newFood.count()
        "*** YOUR CODE HERE ***"
        # closestX = 9999
        # closestY = 9999
        # pointDistance = -1
        # lowestDistance = -1
        # print newFood.width
        # print newFood.height


        walls = successorGameState.getWalls()
        newPosX = newPos[0]
        newPosY = newPos[1]
        ghostPositions = []
        for ghostState in newGhostStates:
            ghostPositions.append(ghostState.getPosition())

        # calculate the number of walls bordering the new position
        numWalls = 0
        if walls[newPosX + 1][newPosY]:
            numWalls += 1
        if walls[newPosX - 1][newPosY]:
            numWalls += 1
        if walls[newPosX][newPosY + 1]:
            numWalls += 1
        if walls[newPosX][newPosY - 1]:
            numWalls += 1


        # for rowIndex in xrange(0, newFood.width):
        #     for colIndex in xrange(0, newFood.height):
        #         if newFood[rowIndex][colIndex]:
        #             pointDistance = manhattanDistance([rowIndex, colIndex], newPos)
        #             if lowestDistance == -1 or lowestDistance > pointDistance:
        #                 lowestDistance = pointDistance
        #                 closestX = rowIndex
        #                 closestY = colIndex

        # print "lowestDistance=", lowestDistance
        # print "successorGameState.getScore()=", successorGameState.getScore()
        # print "point = ", closestX, ", ", closestY

        # return lowestDistance

        # calcuate the manhattan distance till the closest food
        distance = manhattanDistance(self.closestFood, newPos)

        # calculate the sum of manhattan distances from the new position till each of the ghosts
        ghostDistance = 0
        for ghostPosition in ghostPositions:
            individualGhostDistance = manhattanDistance(ghostPosition, newPos)
            #if the ghost is at a new position, return a large negative value
            if(individualGhostDistance == 0):
                return -50
            ghostDistance += individualGhostDistance

        # if the new position is at the nearest food pellet, return a large positive value
        if distance == 0:
            return 9999

        """
        calculate the final score by:
        1. Adding reciprocal of the total distance till the nearest food pellet
        2. Adding the sum of manhattan distance from the new position till all the ghosts
        3. Subtracting the number of walls bordering the position
        """
        distance = (50 / float(distance)) + (ghostDistance / 50) - numWalls

        if(newPos in ghostPositions):
            ghostScared = sum(scared > 0 for scared in newScaredTimes)
            # if new position has a ghost and the ghost is not scared, return large negative value
            if not ghostScared > 0:
                distance *= -1

        return distance

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def utility(self, successorState):
        # this function calls the evaluation function
        return self.evaluationFunction(successorState)

    def terminalTest(self, successorState, gameState, depth):
        # this function tests if the game has ended or the depth has been reached
        if depth <= 0 or successorState.isWin() or successorState.isLose():
            return True
        return False

    def maxValue(self, successorState, gameState, depth):
        # this function plays Max for the given depth
        value = -sys.maxint - 1
        legalMoves = successorState.getLegalActions(0)
        if not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        states = [successorState.generateSuccessor(0, action) for action in legalMoves]
        # let Min play for the given depth
        for state in states:
            value = max(value, self.minValue(state, gameState, depth, 1))
        return value

    def minValue(self, successorState, gameState, depth, ghostNumber):
        # this function plays Min for the given depth
        # Check if the terminal state has been reached
        if self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        value = sys.maxint
        legalMoves = successorState.getLegalActions(ghostNumber)
        if  not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)

        states = [successorState.generateSuccessor(ghostNumber, action) for action in legalMoves]
        # if ghost number is not final ghost, the next ghost will play as Min
        if ghostNumber < (successorState.getNumAgents() - 1):
            for state in states:
                value = min(value, self.minValue(state, gameState, depth, ghostNumber + 1))
        else:
            depth -= 1
            # Check if the terminal state has been reached
            if self.terminalTest(successorState, gameState, depth):
                # if the terminal state has been reached, execute last ghost action and return
                for state in states:
                    value = min(value, self.utility(state))
            else:
                # if the terminal state has not been reached, let Max play for next depth
                for state in states:
                    value = min(value, self.maxValue(state, gameState, depth))

        return value

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        legalMoves = gameState.getLegalActions(0)
        # calculates the move Max has to play
        scores = [self.minValue(gameState.generateSuccessor(0, action), gameState, self.depth, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def utility(self, successorState):
        # this function calls the evaluation function
        return self.evaluationFunction(successorState)

    def terminalTest(self, successorState, gameState, depth):
        # this function tests if the game has ended or the depth has been reached
        if depth <= 0 or successorState.isWin() or successorState.isLose():
            return True
        return False

    def maxValue(self, successorState, gameState, depth, alpha, beta):
        # this function plays Max for the given depth
        value = -sys.maxint - 1
        legalMoves = successorState.getLegalActions(0)
        if not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        # states = [successorState.generateSuccessor(0, action) for action in legalMoves]
        # let Min play for the given depth
        # for state in states:
        for i in range(0, len(legalMoves)):
            state = self.getNextState(successorState, legalMoves, i, 0)
            value = max(value, self.minValue(state, gameState, depth, 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def minValue(self, successorState, gameState, depth, ghostNumber, alpha, beta):
        # this function plays Min for the given depth
        # Check if the terminal state has been reached
        if self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        value = sys.maxint
        legalMoves = successorState.getLegalActions(ghostNumber)
        if not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)

        # states = [successorState.generateSuccessor(ghostNumber, action) for action in legalMoves]
        # if ghost number is not final ghost, the next ghost will play as Min
        if ghostNumber < (successorState.getNumAgents() - 1):
            # for state in states:
            for i in range(0, len(legalMoves)):
                state = self.getNextState(successorState, legalMoves, i, ghostNumber)
                value = min(value, self.minValue(state, gameState, depth, ghostNumber + 1, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
        else:
            depth -= 1
            # Check if the terminal state has been reached
            if self.terminalTest(successorState, gameState, depth):
                # if the terminal state has been reached, execute last ghost action and return
                # for state in states:
                for i in range(0, len(legalMoves)):
                    state = self.getNextState(successorState, legalMoves, i, ghostNumber)
                    value = min(value, self.utility(state))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
            else:
                # if the terminal state has not been reached, let Max play for next depth
                # for state in states:
                for i in range(0, len(legalMoves)):
                    state = self.getNextState(successorState, legalMoves, i, ghostNumber)
                    value = min(value, self.maxValue(state, gameState, depth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)

        return value

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # scores = list()
        # scores.append(self.maxValue(gameState, gameState, self.depth, -sys.maxint - 1, sys.maxint))


        alpha = -sys.maxint - 1
        value = -sys.maxint - 1
        beta = sys.maxint
        depth = self.depth
        scores = []
        legalMoves = gameState.getLegalActions(0)
        for i in range(0, len(legalMoves)):
            # print "alpha = ", alpha
            # print "beta = ", beta
            state = self.getNextState(gameState, legalMoves, i, 0)
            score = self.minValue(state, gameState, depth, 1, alpha, beta)
            value = max(value, score)
            scores.append(score)
            if value > beta:
                break
            alpha = max(alpha, value)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # print "score = ", scores


        # legalMoves = gameState.getLegalActions(0)
        # # calculates the move Max has to play
        # scores = [self.minValue(gameState.generateSuccessor(0, action), gameState, self.depth, 1, -sys.maxint - 1, sys.maxint) for action in legalMoves]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def getNextState(self, successorState, legalMoves, i, agentNumber):
        return successorState.generateSuccessor(agentNumber, legalMoves[i])


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def utility(self, successorState):
        # this function calls the evaluation function
        return self.evaluationFunction(successorState)

    def terminalTest(self, successorState, gameState, depth):
        # this function tests if the game has ended or the depth has been reached
        if depth <= 0 or successorState.isWin() or successorState.isLose():
            return True
        return False

    def maxValue(self, successorState, gameState, depth):
        # this function plays Max for the given depth
        value = -sys.maxint - 1
        legalMoves = successorState.getLegalActions(0)
        if not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        states = [successorState.generateSuccessor(0, action) for action in legalMoves]
        # let Min play for the given depth
        for state in states:
            value = max(value, self.minValue(state, gameState, depth, 1))
        return value

    def minValue(self, successorState, gameState, depth, ghostNumber):
        # this function plays Min for the given depth
        # Check if the terminal state has been reached
        if self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)
        value = 0.0
        legalMoves = successorState.getLegalActions(ghostNumber)
        if not legalMoves or self.terminalTest(successorState, gameState, depth):
            return self.utility(successorState)

        states = [successorState.generateSuccessor(ghostNumber, action) for action in legalMoves]
        # if ghost number is not final ghost, the next ghost will play as Min
        if ghostNumber < (successorState.getNumAgents() - 1):
            for state in states:
                value += self.minValue(state, gameState, depth, ghostNumber + 1)
        else:
            depth -= 1
            # Check if the terminal state has been reached
            if self.terminalTest(successorState, gameState, depth):
                # if the terminal state has been reached, execute last ghost action and return
                for state in states:
                    value += self.utility(state)
            else:
                # if the terminal state has not been reached, let Max play for next depth
                for state in states:
                    value += self.maxValue(state, gameState, depth)

        return value / len(legalMoves)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        legalMoves = gameState.getLegalActions(0)
        # calculates the move Max has to play
        scores = [self.minValue(gameState.generateSuccessor(0, action), gameState, self.depth, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

