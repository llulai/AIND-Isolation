"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
from sample_players import *
from tournament import Agent
from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = Agent(game_agent.MinimaxPlayer(score_fn=open_move_score), "MM_Open")
        self.player2 = Agent(game_agent.MinimaxPlayer(score_fn=improved_score), "MM_Improved")
        self.game = isolation.Board(self.player1, self.player2)

        self.game.play()




if __name__ == '__main__':
    unittest.main()
