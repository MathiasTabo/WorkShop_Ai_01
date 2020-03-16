

import gym
import numpy as np

env = gym.???("MountainCar-v0") # cree l evironement avec les methodes de gym


LEARNING_RATE = 0.1
DISCOUNT = 0.95


EPISODES = ???  # metez le nombre de party que vous volez effectuer. 
SHOW = 1000


epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print(env.observation_space)


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE



# cette ligne juste en dessou permer de charger une q_table enregistré
# q_table = np.load("qtable.npy")

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

    

print(DISCRETE_OS_SIZE)
print(discrete_os_win_size)
print(q_table)


for episode in range(???): # Trouver le nombre de partie a faire
    discrete_state = get_discrete_state(env.reset())

    done = False
    # print("SCORE = " + str(max_reward) + "in party = " +  str(episode))
    max_reward = 0
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.???) # metez une action alleatoire

            
        new_state, reward, done, info = env.step(action) # donnez l'action a l'environement

        new_discrete_state = get_discrete_state(new_state)

        if (episode % ??? == 0): # mettez le nombre ou vous voulez voir la partie (toutes les 1000 parties (SHOW = 1000))
            env.render()
            
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            print("win in episode: " + str(episode))
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value


            
# pour sauvgarder une qtabe faite

# np.save("qtable.npy", q_table)

env.close()
