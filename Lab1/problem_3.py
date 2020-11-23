import time

import numpy as np
import matplotlib.pyplot as plt

from Lab1.city_reloaded import City, draw_city

HEIGHT = 4
WIDTH = 4
PLAYER_START = (0, 0)
POLICE_START = (HEIGHT - 1, WIDTH - 1)
START_STATE_TUPLE = (PLAYER_START[0], PLAYER_START[1], POLICE_START[0], POLICE_START[1])
LAMBDA = 0.8
NUM_ACTIONS = 5

actions_symbols = {
    0: "▪",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓"
}


def create_city():
    city = np.zeros((HEIGHT, WIDTH))
    city[1, 1] = 2

    return City(city, START_STATE_TUPLE)


def plot_city(city, s, policy):
    i_p, j_p, i_o, j_o = city.state_to_pos[s]
    player_tuple = (i_p, j_p)
    police_tuple = (i_o, j_o)
    city_copy = np.copy(city.city)
    city_copy[player_tuple] = 3
    city_copy[police_tuple] = -1  # Goal
    plt.clf()
    # plt.close()
    # texts = get_action_for_each_player_pos(city, s, policy)
    texts = None
    draw_city(city_copy, texts)
    plt.show()
    time.sleep(0.7)


def random_action():
    return np.random.randint(0, NUM_ACTIONS)


def plot_V(vs, title):
    xrange = np.arange(0, len(vs) / 10, 0.1)
    plt.plot(xrange, vs)
    plt.xlabel("iterations / $10^3$")
    plt.ylabel("V(init_s)")
    plt.title(title)
    plt.show()


def Q_simulations(city, steps):
    init_s = city.pos_to_state[START_STATE_TUPLE]
    s = init_s
    # plot_city(city, s, None)
    Q = np.zeros((city.n_states, city.n_actions))
    Q_visits = np.zeros((city.n_states, city.n_actions))
    init_s_V = []
    save_state_interval = 100

    for i in range(steps):
        # Simulate with behavior policy
        a = np.random.randint(0, NUM_ACTIONS)
        reward, s_next = city.simulate_step(s, a)

        # Update result list
        if i % save_state_interval == 0:
            init_s_V.append(np.max(Q[init_s]))
            print(init_s_V[-1])

        # Learn by using observations (s, a, r, s_next)
        Q_visits[s, a] += 1
        lr = 1 / np.power(Q_visits[s, a], 2.0 / 3.0)
        Q[s, a] = Q[s, a] + lr * (reward + LAMBDA * np.max(Q[s_next]) - Q[s, a])

        s = s_next
        # plot_city(city, s, None)

    plot_V(init_s_V, "Q-Learning")


def SARSA_select_action(city, Q, s, epsilon):
    i_p, j_p, i_o, j_o = city.state_to_pos[s]
    possible_actions = []
    if i_p > 0:
        possible_actions.append(city.MOVE_UP)
    if i_p < HEIGHT-1:
        possible_actions.append(city.MOVE_DOWN)
    if j_p > 0:
        possible_actions.append(city.MOVE_LEFT)
    if j_p < WIDTH-1:
        possible_actions.append(city.MOVE_RIGHT)

    if np.random.uniform(0.0, 1.0) < epsilon:
        return np.random.choice(possible_actions)
        # return np.random.randint(0, NUM_ACTIONS)
    else:
        return np.argmax(Q[s])


def SARSA_simulations(city, steps, epsilon=0.1):
    init_s = city.pos_to_state[START_STATE_TUPLE]
    state_to_record = city.pos_to_state[(START_STATE_TUPLE[0], START_STATE_TUPLE[1], 2, 1)]
    s = init_s
    # plot_city(city, s, None)
    Q = np.zeros((city.n_states, city.n_actions))
    Q_visits = np.zeros((city.n_states, city.n_actions))
    recorded_V = []
    save_state_interval = 100

    a = SARSA_select_action(city, Q, s, epsilon)

    for i in range(steps):
        # Simulate with behavior policy
        reward, s_next = city.simulate_step(s, a)
        a_next = SARSA_select_action(city, Q, s_next, epsilon)

        # Update result list
        if i % save_state_interval == 0:
            recorded_V.append(np.max(Q[state_to_record]))
            print(recorded_V[-1])

        # Learn by using observations (s, a, r, s_next, a_next)
        Q_visits[s, a] += 1
        lr = 1 / np.power(Q_visits[s, a], 2.0 / 3.0)
        Q[s, a] = Q[s, a] + lr * (reward + LAMBDA * Q[s_next, a_next] - Q[s, a])

        s = s_next
        a = a_next
        # plot_city(city, s, None)

    plot_V(recorded_V, "SARSA, epsilon=" + str(epsilon))


if __name__ == '__main__':
    city = create_city()
    expected_iterations_Q = 10000000
    # Q_simulations(city, expected_iterations_Q)
    SARSA_simulations(city, int(expected_iterations_Q / 10), 0.05)
