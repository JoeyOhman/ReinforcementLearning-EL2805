import time

import numpy as np
import matplotlib.pyplot as plt

from maze import Maze, draw_maze, value_iteration, dynamic_programming

HEIGHT = 7
WIDTH = 8
PLAYER_START = (0, 0)
MINO_START = (HEIGHT - 1, WIDTH - 3)
START_STATE_TUPLE = (0, 0, HEIGHT - 1, WIDTH - 3)

actions_symbols = {
    0: "▪",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}


def create_maze(mino_still):
    maze_arr = np.zeros((HEIGHT, WIDTH))
    maze_arr[-1, 4] = 1
    maze_arr[1, -3] = 1
    maze_arr[3, -3] = 1
    for i in range(4):
        maze_arr[i, 2] = 1
    for j in range(3):
        maze_arr[2, 5 + j] = 1
    for j in range(6):
        maze_arr[-2, 1 + j] = 1

    maze_arr[MINO_START] = 2

    return Maze(maze_arr, mino_still)


def get_action_for_each_player_pos(maze, s, policy, t=None):
    texts = []
    _, _, i_m, j_m = s
    for i in range(maze.maze.shape[0]):
        texts.append([])
        for j in range(maze.maze.shape[1]):
            if maze.maze[i, j] != 1:
                new_s = maze.map[(i, j, i_m, j_m)]
                if t is None:
                    a = policy[new_s]
                else:
                    a = policy[new_s, t]
                texts[i].append(actions_symbols[a])
            else:
                texts[i].append("")

    return np.array(texts)


def plot_maze(maze, s, policy, t=None):
    i_p, j_p, i_m, j_m = s
    player_tuple = (i_p, j_p)
    mino_tuple = (i_m, j_m)
    # i_p_prev, j_p_prev, i_m_prev, j_m_prev = path[i - 1]
    # player_prev_tuple = (i_p_prev, j_p_prev)
    maze_cpy = np.copy(maze.maze)
    maze_cpy[player_tuple] = 3
    maze_cpy[MINO_START] = 2  # Goal
    maze_cpy[mino_tuple] = -1
    plt.clf()
    # plt.close()
    texts = get_action_for_each_player_pos(maze, s, policy, t)
    draw_maze(maze_cpy, texts)
    plt.show()
    time.sleep(0.7)


def simulate_prob(maze, policy, method, numSims):
    numWins = 0
    for i in range(numSims):
        path = maze.simulate(START_STATE_TUPLE, policy, method)
        last_state = path[-1]
        if (last_state[0], last_state[1]) == MINO_START \
                and not (last_state[0] == last_state[2] and last_state[1] == last_state[3]):
            numWins += 1

    return numWins / numSims


def plot_escape_probs(maze, T):
    max_probs = []
    for t in range(T + 1):
        V, policy = dynamic_programming(maze, t)
        max_prob = simulate_prob(maze, policy, "DynProg", 1000)
        max_probs.append(max_prob)
        print(max_prob)

    plt.plot(range(0, T + 1), max_probs)
    plt.xlabel("T")
    plt.ylabel("Maximal escape probability")
    plt.title("Max escape probability, minotaur can stand still")
    plt.show()


def solve_dp(maze):
    T = 20
    V, policy = dynamic_programming(maze, T)
    return V, policy


def solve_vi(maze):
    # Discount Factor
    gamma = 29.0 / 30.0
    # gamma = 0.999
    # Accuracy threshold
    epsilon = 0.0001
    V, policy = value_iteration(maze, gamma, epsilon)
    winRate = simulate_prob(maze, policy, "ValIter", 10000)
    print("Maximum Success Probability:", winRate)
    return V, policy


def plot_simulation(maze):
    V, policy = solve_dp(maze)
    path = maze.simulate(START_STATE_TUPLE, policy, "DynProg")
    for t, p in enumerate(path):
        plot_maze(maze, p, policy, t)


if __name__ == '__main__':
    maze = create_maze(True)
    # plot_simulation(maze)
    # plot_escape_probs(maze, 20)
    solve_vi(maze)
