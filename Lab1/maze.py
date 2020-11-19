import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display, get_ipython

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    ENTER_GOAL_REWARD = 1000
    LEAVE_GOAL_REWARD = -2000
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_REWARD = -100
    TIME_REWARD = -100

    def __init__(self, maze, mino_still=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.mino_still = mino_still
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        # Coordinates for player (p), minotaur (m) and time t
        # for t in range(self.T):
        for i_p in range(self.maze.shape[0]):
            for j_p in range(self.maze.shape[1]):
                for i_m in range(self.maze.shape[0]):
                    for j_m in range(self.maze.shape[1]):
                        if self.maze[i_p, j_p] != 1:
                            states[s] = (i_p, j_p, i_m, j_m)
                            map[(i_p, j_p, i_m, j_m)] = s
                            s += 1
        return states, map

    def __next_states_minotaur(self, player_pos, mino_pos):
        i_p, j_p = player_pos
        i_m, j_m = mino_pos
        possible_actions = []

        if i_m > 0:
            possible_actions.append(self.actions[self.MOVE_UP])
        if i_m < self.maze.shape[0] - 1:
            possible_actions.append(self.actions[self.MOVE_DOWN])
        if j_m > 0:
            possible_actions.append(self.actions[self.MOVE_LEFT])
        if j_m < self.maze.shape[1] - 1:
            possible_actions.append(self.actions[self.MOVE_RIGHT])

        if self.mino_still:
            possible_actions.append((0, 0))

        next_states = []
        for a in possible_actions:
            i_m_next = i_m + a[0]
            j_m_next = j_m + a[1]
            next_states.append(self.map[(i_p, j_p, i_m_next, j_m_next)])

        return next_states

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            Move minotaur to all possible directions and pair the sequence of next states with transition probabilities.

            :return list of possible next states and list of transition probabilities
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)
        # Based on the impossiblity check return the next state.
        next_player_pos = (self.states[state][0], self.states[state][1]) if hitting_maze_walls else (row, col)
        mino_pos = (self.states[state][2], self.states[state][3])
        next_states = self.__next_states_minotaur(next_player_pos, mino_pos)
        return next_states, [1.0 / len(next_states) for _ in next_states]  # Equal prob for each state

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        # num_states_no_t = int(self.n_states / self.T)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            s_tup = self.states[s]
            s_tup = (s_tup[0], s_tup[1], s_tup[2], s_tup[3])
            s = self.map[s_tup]
            for a in range(self.n_actions):
                next_states, next_probs = self.__move(s, a)
                # Stochastic transitions of minotaur, get list of next_states and list of next_probs
                for next_outcome_idx in range(len(next_probs)):
                    next_s = next_states[next_outcome_idx]
                    s_tup_next = self.states[next_s]
                    s_tup_next = (s_tup_next[0], s_tup_next[1], s_tup_next[2], s_tup_next[3])
                    next_s = self.map[s_tup_next]

                    next_p = next_probs[next_outcome_idx]
                    transition_probabilities[next_s, s, a] = next_p
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        # if weights is None:
        for s in range(self.n_states):
            i_p, j_p, _, _ = self.states[s]
            for a in range(self.n_actions):
                next_states, next_probs = self.__move(s, a)
                i_p_next, j_p_next, _, _ = self.states[next_states[0]]  # All next states contain same player position
                player_moved = not ((i_p, j_p) == (i_p_next, j_p_next))
                # Reward for hitting a wall
                if not player_moved and a != self.STAY:
                    rewards[s, a] = self.IMPOSSIBLE_REWARD
                # Reward for reaching the exit
                elif not player_moved and self.maze[i_p][j_p] == 2:
                    rewards[s, a] = self.GOAL_REWARD
                elif player_moved and self.maze[i_p][j_p] == 2:
                    rewards[s, a] = self.LEAVE_GOAL_REWARD  # Do not leave goal and enter again to gain repeated rewards
                # Reward for taking a step to an empty cell that is not the exit
                # Average reward for all possible stochastic transitions (we could end up in same position as minotaur
                else:
                    for next_outcome_idx in range(len(next_probs)):
                        # print(next_states)
                        # print(next_states[next_outcome_idx])
                        i_p_next, j_p_next, i_m_next, j_m_next = self.states[next_states[next_outcome_idx]]
                        if (i_p_next, j_p_next) == (i_m_next, j_m_next):
                            outcome_reward = self.MINOTAUR_REWARD
                        else:
                            outcome_reward = self.ENTER_GOAL_REWARD if self.maze[i_p_next][
                                                                           j_p_next] == 2 else self.STEP_REWARD

                        rewards[s, a] += outcome_reward * next_probs[next_outcome_idx]

                    '''
                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]] < 0:
                        row, col = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s, a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s, a]
                        # The average reward
                        rewards[s, a] = 0.5 * r1 + 0.5 * r2
                    '''
        # If the weights are described by a weight matrix
        '''
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]
        '''

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                # next_s = self.__move(s, policy[s, t])
                next_states, next_probs = self.__move(s, policy[s, t])
                next_s = np.random.choice(next_states, p=next_probs)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s

                # Exit if dead or goal
                i_p, j_p, i_m, j_m = self.states[next_s]
                if (i_p, j_p) == (i_m, j_m) or self.maze[i_p][j_p] == 2:
                    break

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_states, next_probs = self.__move(s, policy[s])
            next_s = np.random.choice(next_states, p=next_probs)
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while True:  # s != next_s:
                # Update state
                s = next_s
                t += 1
                # Move to next state given the policy and the current state
                next_states, next_probs = self.__move(s, policy[s])
                next_s = np.random.choice(next_states, p=next_probs)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                i_p, j_p, i_m, j_m = self.states[next_s]
                if (i_p, j_p) == (i_m, j_m) or self.maze[i_p][j_p] == 2:
                    break

        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        print("DP, t:", t)
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    # num_states_no_t = int(n_states / env.T)
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 2000:
        # Increment by one the numbers of iteration
        n += 1
        print("VI Iteration:", n)
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze, texts=None):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_ORANGE}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))
    # texts = np.zeros(maze.shape)

    # Create a table to color
    grid = plt.table(cellText=texts,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


'''
def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color at each frame
    for i in range(len(path)):
        i_p, j_p, i_m, j_m = path[i]
        player_tuple = (i_p, j_p)
        i_p_prev, j_p_prev, i_m_prev, j_m_prev = path[i - 1]
        player_prev_tuple = (i_p_prev, j_p_prev)
        grid.get_celld()[player_tuple].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[player_tuple].get_text().set_text('Player')
        if i > 0:
            if path[i] == path[i - 1]:
                grid.get_celld()[player_tuple].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[player_tuple].get_text().set_text('Player is out')
            else:
                grid.get_celld()[player_prev_tuple].set_facecolor(col_map[maze[player_prev_tuple]])
                grid.get_celld()[player_prev_tuple].get_text().set_text('')
        display.display(fig)
        # plt.imshow(fig)
        plt.show()
        # display.clear_output(wait=True)
        time.sleep(1)
'''
