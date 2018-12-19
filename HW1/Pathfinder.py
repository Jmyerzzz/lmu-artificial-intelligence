'''
The Pathfinder class is responsible for finding a solution (i.e., a
sequence of actions) that takes the agent from the initial state to all
of the goals with optimal cost.

This task is done in the solve method, as parameterized
by a maze pathfinding problem, and is aided by the SearchTreeNode DS.

Jackson Myers
'''
import unittest
from queue import PriorityQueue
from MazeProblem import MazeProblem
from SearchTreeNode import SearchTreeNode

def total_cost(current, child):
    return current.totalCost + child

def heuristic(current, goals):
    distances = []
    for goal in goals:
        distances.append(abs(current[0]-goal[0])+abs(current[1]-goal[1]))
    return min(distances)

def is_goal(state, goals):
    return goals.count(state) > 0

def get_actions(current, path_root):
    actions = []
    while current.parent is not path_root.parent:
        actions.insert(0, current.action)
        current = current.parent
        if current.parent is None:
            break
    return actions

def solve(problem, initial, goals):
    frontier = PriorityQueue()
    root = SearchTreeNode(initial, None, None, 0, heuristic(initial, goals))
    path_root = root
    frontier.put(root)
    closed_list = {}
    actions = []

    while not frontier.empty():
        current = frontier.get()
        if is_goal(current.state, goals):
            goals.remove(current.state)
            actions.extend(get_actions(current, path_root))
            path_root = current
            if not goals:
                return actions
            frontier.queue.clear()
            closed_list.clear()
        closed_list[current.state] = 1
        for node in problem.transitions(current.state):
            if node[2] not in closed_list:
                child = SearchTreeNode(node[2], node[0], current, total_cost(current, node[1]), heuristic(node[2], goals))
                frontier.put(child)
    return None


class PathfinderTests(unittest.TestCase):

    def test_maze1(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.M.X",
                "X.X.X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (1, 3)
        goals   = [(5, 3)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 8)

    def test_maze2(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.M.X",
                "X.X.X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (1, 3)
        goals   = [(3, 3),(5, 3)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 12)

    def test_maze3(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.MMX",
                "X...M.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 1)
        goals   = [(5, 3), (1, 3), (1, 1)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 12)

    def test_maze4(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.XXX",
                "X...X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 1)
        goals   = [(5, 3), (1, 3), (1, 1)]
        soln = solve(problem, initial, goals)
        self.assertTrue(soln == None)

    def test_maze5(self):
        maze = ["XXXXXXX",
                "X...X.X",
                "X.XXXMX",
                "X.MM.MX",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (1, 3)
        goals   = [(3,1), (5,1), (4,3)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 22)

    def test_maze6(self):
        maze = ["XXXXXXXXXXX",
                "X..MMM.MX.X",
                "X.X.XXX.X.X",
                "X...XXX...X",
                "X....MX.XXX",
                "X.M.XXX...X",
                "X...M...X.X",
                "XXXXXXXXXXX"]
        problem = MazeProblem(maze)
        initial = (9, 6)
        goals   = [(6, 1), (3, 5), (9, 1)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 31)

    def test_maze7(self):
        maze = ["XXXXXXXXXXX",
                "X.X..MX...X",
                "X.XMM.MM.XX",
                "XM..XMM.X.X",
                "X.X....MX.X",
                "X..MX.X...X",
                "X.M..X....X",
                "XXXXXXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 2)
        goals   = [(9, 1), (9, 6), (1, 1), (4, 6), (5, 5)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 55)

    def test_maze8(self):
        maze = ["XXXXXXXXXXX",
                "X.X..MX...X",
                "X.XMM.MM.XX",
                "XM..XMM.X.X",
                "X.X....MX.X",
                "X..MX.X..XX",
                "X.M..X..X.X",
                "XXXXXXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 2)
        goals   = [(9, 1), (9, 6), (1, 1), (4, 6), (5, 5)]
        soln = solve(problem, initial, goals)
        self.assertTrue(soln == None)


if __name__ == '__main__':
    unittest.main()
