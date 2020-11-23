import numpy as np
import matplotlib.pyplot as plt

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class City:
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
    STEP_REWARD = 0
    BANK_REWARD = 1
    IMPOSSIBLE_REWARD = -1000
    POLICE_REWARD = -10

    def __init__(self, city, start_state):
        """ Constructor of the environment City.
        """
        self.city = city
        self.actions = self.__actions()
        self.state_to_pos, self.pos_to_state = self.__states()
        self.start_state = self.pos_to_state[start_state]
        self.n_actions = len(self.actions)
        self.n_states = len(self.state_to_pos)
        # self.transition_probabilities = self.__transitions()
        # self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        state_to_pos = dict()
        pos_to_state = dict()
        s = 0
        # Coordinates for player (p), officer/police (m) and time t
        # for t in range(self.T):
        for i_p in range(self.city.shape[0]):
            for j_p in range(self.city.shape[1]):
                for i_o in range(self.city.shape[0]):
                    for j_o in range(self.city.shape[1]):
                        state_to_pos[s] = (i_p, j_p, i_o, j_o)
                        pos_to_state[(i_p, j_p, i_o, j_o)] = s
                        s += 1
        return state_to_pos, pos_to_state

    def __next_state_police(self, player_pos_next, police_pos):

        i_p_next, j_p_next = player_pos_next
        i_o, j_o = police_pos
        possible_actions = []

        if i_o > 0:
            possible_actions.append(self.MOVE_UP)
        if i_o < self.city.shape[0] - 1:
            possible_actions.append(self.MOVE_DOWN)
        if j_o > 0:
            possible_actions.append(self.MOVE_LEFT)
        if j_o < self.city.shape[1] - 1:
            possible_actions.append(self.MOVE_RIGHT)

        a = self.actions[np.random.choice(possible_actions)]
        i_o_next = i_o + a[0]
        j_o_next = j_o + a[1]

        next_state = self.pos_to_state[(i_p_next, j_p_next, i_o_next, j_o_next)]

        return next_state

    def __move(self, state, action):
        i_p, j_p, i_o, j_o = self.state_to_pos[state]
        player_pos = (i_p, j_p)
        police_pos = (i_o, j_o)

        # Compute the future position given current (state, action)
        row = i_p + self.actions[action][0]
        col = j_p + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.city.shape[0]) or \
                             (col == -1) or (col == self.city.shape[1])
        # Based on the impossibility check return the next state.
        next_player_pos = player_pos if hitting_maze_walls else (row, col)
        next_state = self.__next_state_police(next_player_pos, police_pos)
        return next_state

    '''
    def __transitions(self):

        # Initialize the transition probabilities tensor (S,S,A)
        # num_states_no_t = int(self.n_states / self.T)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states, next_probs = self.__move(s, a)
                # Stochastic transitions of police, get list of next_states and list of next_probs
                for next_outcome_idx in range(len(next_probs)):
                    next_s = next_states[next_outcome_idx]
                    next_p = next_probs[next_outcome_idx]
                    transition_probabilities[next_s, s, a] = next_p
        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            i_p, j_p, i_o, j_o = self.state_to_pos[s]
            # resetting = (i_p, j_p) == (i_o, j_o)  # Reset
            for a in range(self.n_actions):
                next_states, next_probs = self.__move(s, a)
                for next_outcome_idx in range(len(next_probs)):
                    # Find next possible state and see if player moved
                    i_p_next, j_p_next, i_m_next, j_m_next = self.state_to_pos[next_states[next_outcome_idx]]
                    player_moved = not ((i_p, j_p) == (i_p_next, j_p_next))

                    if not player_moved and a != self.STAY:
                        outcome_reward = self.IMPOSSIBLE_REWARD
                    elif (i_p_next, j_p_next) == (i_m_next, j_m_next):
                        outcome_reward = self.POLICE_REWARD
                    elif self.city[i_p_next][j_p_next] == 2:
                        outcome_reward = self.BANK_REWARD
                    else:
                        outcome_reward = self.STEP_REWARD

                    rewards[s, a] += outcome_reward * next_probs[next_outcome_idx]

        return rewards
    '''

    def observe_reward(self, s, a, next_s):
        i_p, j_p, i_o, j_o = self.state_to_pos[s]
        i_p_next, j_p_next, i_m_next, j_m_next = self.state_to_pos[next_s]
        player_moved = not ((i_p, j_p) == (i_p_next, j_p_next))

        if not player_moved and a != self.STAY:
            return self.IMPOSSIBLE_REWARD
        elif (i_p_next, j_p_next) == (i_m_next, j_m_next):
            return self.POLICE_REWARD
        elif self.city[i_p_next][j_p_next] == 2:
            return self.BANK_REWARD
        else:
            return self.STEP_REWARD

    def simulate_step(self, s, a):

        next_s = self.__move(s, a)
        observed_reward = self.observe_reward(s, a, next_s)

        return observed_reward, next_s


def draw_city(city, texts=None):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_ORANGE}

    # Give a color to each cell
    rows, cols = city.shape
    colored_maze = [[col_map[city[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = city.shape
    colored_maze = [[col_map[city[j, i]] for i in range(cols)] for j in range(rows)]

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
