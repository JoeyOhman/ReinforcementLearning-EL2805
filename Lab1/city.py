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
    BANK_REWARD = 10
    IMPOSSIBLE_REWARD = -1000
    POLICE_REWARD = -50

    def __init__(self, city, start_state):
        """ Constructor of the environment City.
        """
        self.city = city
        self.actions = self.__actions()
        self.state_to_pos, self.pos_to_state = self.__states()
        self.start_state = self.pos_to_state[start_state]
        self.n_actions = len(self.actions)
        self.n_states = len(self.state_to_pos)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

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

    def __next_states_police(self, player_pos_next, police_pos, player_pos):
        # Only called when player_pos != police_pos
        # assert player_pos != police_pos

        i_p, j_p = player_pos
        i_o, j_o = police_pos
        i_p_next, j_p_next = player_pos_next
        possible_actions = []

        if i_o > 0 and i_p <= i_o:
            possible_actions.append(self.actions[self.MOVE_UP])
        if i_o < self.city.shape[0] - 1 and i_p >= i_o:
            possible_actions.append(self.actions[self.MOVE_DOWN])
        if j_o > 0 and j_p <= j_o:
            possible_actions.append(self.actions[self.MOVE_LEFT])
        if j_o < self.city.shape[1] - 1 and j_p >= j_o:
            possible_actions.append(self.actions[self.MOVE_RIGHT])

        next_states = []
        resets = []
        for a in possible_actions:
            i_o_next = i_o + a[0]
            j_o_next = j_o + a[1]
            if player_pos_next == (i_o_next, j_o_next):
                next_state = self.start_state
                reset = True
            else:
                next_state = self.pos_to_state[(i_p_next, j_p_next, i_o_next, j_o_next)]
                reset = False

            next_states.append(next_state)
            resets.append(reset)

        return next_states, resets

    def __move(self, state, action):
        i_p, j_p, i_o, j_o = self.state_to_pos[state]
        player_pos = (i_p, j_p)
        police_pos = (i_o, j_o)

        # If player-police collision, reset to starting state
        # if player_pos == police_pos:
        # return [self.start_state], [1.0]

        # Compute the future position given current (state, action)
        row = i_p + self.actions[action][0]
        col = j_p + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.city.shape[0]) or \
                             (col == -1) or (col == self.city.shape[1])
        # Based on the impossibility check return the next state.
        next_player_pos = player_pos if hitting_maze_walls else (row, col)
        next_states, resets = self.__next_states_police(next_player_pos, police_pos, player_pos)
        return next_states, [1.0 / len(next_states) for _ in next_states], resets  # Equal prob for each state

    def __transitions(self):

        # Initialize the transition probabilities tensor (S,S,A)
        # num_states_no_t = int(self.n_states / self.T)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states, next_probs, _ = self.__move(s, a)
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
                next_states, next_probs, resets = self.__move(s, a)
                # TODO: Find out here if the state got reset? Compare states to see if large jumps or return boolean from move?
                for next_outcome_idx in range(len(next_probs)):
                    # Find next possible state and see if player moved
                    i_p_next, j_p_next, i_m_next, j_m_next = self.state_to_pos[next_states[next_outcome_idx]]
                    player_moved = not ((i_p, j_p) == (i_p_next, j_p_next))

                    if not player_moved and a != self.STAY and not resets[next_outcome_idx]:
                        outcome_reward = self.IMPOSSIBLE_REWARD
                    elif resets[next_outcome_idx]:  # (i_p_next, j_p_next) == (i_m_next, j_m_next):
                        outcome_reward = self.POLICE_REWARD
                    elif self.city[i_p_next][j_p_next] == 2:
                        outcome_reward = self.BANK_REWARD
                    else:
                        outcome_reward = self.STEP_REWARD

                    rewards[s, a] += outcome_reward * next_probs[next_outcome_idx]

        return rewards

    def simulate(self, start, policy, num_steps=100):
        path = list()

        # Initialize current state, next state and time
        t = 1
        s = self.pos_to_state[start]
        # Add the starting position in the maze to the path
        path.append(start)
        # Move to next state given the policy and the current state
        next_states, next_probs, _ = self.__move(s, policy[s])
        next_s = np.random.choice(next_states, p=next_probs)
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.state_to_pos[next_s])
        # Loop while state is not the goal state
        while t < num_steps:  # s != next_s:
            # Update state
            s = next_s
            t += 1
            # Move to next state given the policy and the current state
            next_states, next_probs, _ = self.__move(s, policy[s])
            next_s = np.random.choice(next_states, p=next_probs)
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.state_to_pos[next_s])
            # Update time and state for next iteration
            # t += 1
            i_p, j_p, i_o, j_o = self.state_to_pos[next_s]
            # if (i_p, j_p) == (i_o, j_o) or self.city[i_p][j_p] == 2:
                # break

        return path


def value_iteration(env, gamma, epsilon):
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
    while np.linalg.norm(V - BV) >= tol and n < 20000:
        # Increment by one the numbers of iteration
        n += 1
        if n % 100 == 0:
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
    print("Converged in", n, "iterations")
    # Return the obtained policy
    return V, policy


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
