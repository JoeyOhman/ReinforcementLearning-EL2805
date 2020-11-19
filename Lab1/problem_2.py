import time

import numpy as np
import matplotlib.pyplot as plt

from Lab1.city import City, draw_city, value_iteration

HEIGHT = 3
WIDTH = 6
PLAYER_START = (0, 0)
POLICE_START = (1, 2)
START_STATE_TUPLE = (0, 0, 1, 2)

actions_symbols = {
    0: "▪",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}


def create_city():
    city = np.zeros((HEIGHT, WIDTH))
    city[0, 0] = 2
    city[0, -1] = 2
    city[-1, 0] = 2
    city[-1, -1] = 2

    return City(city, START_STATE_TUPLE)


def get_action_for_each_player_pos(city, s, policy):
    texts = []
    _, _, i_o, j_o = s
    for i in range(city.city.shape[0]):
        texts.append([])
        for j in range(city.city.shape[1]):
            if (i, j) != (i_o, j_o):
                new_s = city.pos_to_state[(i, j, i_o, j_o)]
                a = policy[new_s]
                texts[i].append(actions_symbols[a])
            else:
                texts[i].append("")

    return np.array(texts)


def plot_city(city, s, policy):
    i_p, j_p, i_o, j_o = s
    player_tuple = (i_p, j_p)
    police_tuple = (i_o, j_o)
    # i_p_prev, j_p_prev, i_m_prev, j_m_prev = path[i - 1]
    # player_prev_tuple = (i_p_prev, j_p_prev)
    city_copy = np.copy(city.city)
    city_copy[player_tuple] = 3
    city_copy[police_tuple] = -1  # Goal
    plt.clf()
    # plt.close()
    texts = get_action_for_each_player_pos(city, s, policy)
    # texts = None
    draw_city(city_copy, texts)
    plt.show()
    time.sleep(0.7)


def solve_vi(city, gamma=0.98):
    # Discount Factor
    # gamma = 0.98
    # gamma = 0.999
    # Accuracy threshold
    epsilon = 0.0001
    V, policy = value_iteration(city, gamma, epsilon)
    start_state = city.pos_to_state[START_STATE_TUPLE]
    start_state_value = V[start_state]
    print("Initial state value:", start_state_value)
    return V, policy, start_state_value


def func_of_discount(city):
    vals = []
    gamma_range = np.arange(0.90, 1.01, 0.01)
    for gamma in gamma_range:
        V, policy, start_val = solve_vi(city, gamma)
        vals.append(start_val)

    plt.plot(gamma_range, vals)
    plt.title("Initial state value for different discount factors")
    plt.xlabel("gamma")
    plt.ylabel("Value")
    plt.show()


if __name__ == '__main__':
    city = create_city()
    # plot_city(city, START_STATE_TUPLE, None)
    V, policy, start_val = solve_vi(city, 0.99)
    path = city.simulate(START_STATE_TUPLE, policy)
    for p in path:
        plot_city(city, p, policy)
    # func_of_discount(city)
