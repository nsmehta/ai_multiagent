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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
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
        print 'successorGameState = ', successorGameState
        newPos = successorGameState.getPacmanPosition()
        print 'newPos = ', newPos
        newFood = successorGameState.getFood()
        print 'newFood = ', newFood
        newGhostStates = successorGameState.getGhostStates()
        print 'newGhostStates = ', newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print 'newScaredTimes = ', newScaredTimes

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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

    def calculateFoodPalletsLeft(self, gameState):
        return gameState.getNumFood()
        # return 0

    def utility(self, sucessorState):
        return self.evaluationFunction(sucessorState)

    def terminalTest(self, sucessorState, gameState):
        if self.depth < 0:
            return True
        return False
        # return gameState.getNumFood() <= 0
        # return self.calculateFoodPalletsLeft(gameState) <= 0

    def maxValue(self, sucessorState, gameState):
        self.depth -= 1
        if self.terminalTest(sucessorState, gameState):
            return self.utility(sucessorState)
        value = -sys.maxint - 1
        legalMoves = gameState.getLegalActions(0)
        states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        for state in states:
            value = max(value, self.minValue(state, gameState))

        return value

    def minValue(self, sucessorState, gameState):
        if self.terminalTest(sucessorState, gameState):
            return self.utility(sucessorState)
        value = sys.maxint
        states = list()
        for i in range(1, gameState.getNumAgents()):
            legalMoves = gameState.getLegalActions(i)
            states += [gameState.generateSuccessor(i, action) for action in legalMoves]
        for state in states:
            value = min(value, self.maxValue(state, gameState))

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
        # print 'numAgents = ', gameState.getNumAgents()
        # print 'legalActions0 = ', gameState.getLegalActions(0)
        # print 'legalActions1 = ', gameState.getLegalActions(1)
        # print 'legalActions2 = ', gameState.getLegalActions(2)
        # print 'legalActions3 = ', gameState.getLegalActions(3)
        # print 'gameState = ', gameState.generateSuccessor(0, 'West')
        # print 'self.depth = ', self.depth
        # print 'getNumFood = ', gameState.getNumFood()
        # print 'self.evaluationFunction = ', self.evaluationFunction
        legalMoves = gameState.getLegalActions(0)
        scores = [self.minValue(gameState.generateSuccessor(0, action), gameState) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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

