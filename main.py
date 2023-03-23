import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")


LEARNING_RATE = 0.1  # how fast we learn
EPSILON = 0.05  # how much we explore
DISCOUNT = 0.95  # how much we care about future rewards
EPISODES = 500  # how many episodes we want to run
SHOW_EVERY = 50  # how often we want to see the results
DISC_STEPS = 6   # how many steps we want to divide the space into
STATE_COUNT = len(env.observation_space.high)  # how many states we have
DISC_STATE_COUNT = DISC_STEPS*STATE_COUNT

LOWER_OBSERVATIONS = env.observation_space.low
LOWER_OBSERVATIONS[1] = -4
LOWER_OBSERVATIONS[3] = -4
UPPER_OBSERVATIONS = env.observation_space.high
UPPER_OBSERVATIONS[1] = 4
UPPER_OBSERVATIONS[3] = 4

temp_array = [DISC_STEPS]*len(env.observation_space.high)

discrete_step_array = [0]*len(temp_array)

for i in range(len(temp_array)):

    curr_step = (env.observation_space.high -
                 env.observation_space.low)/temp_array[i]

    if curr_step[i] != float("inf"):
        discrete_step_array[i] = curr_step[i]
    else:
        discrete_step_array[i] = ((4*4)/temp_array[i])


def init_disc_state_table(disc_state_table, lower_observations, discrete_step_array):

    temp_target_count = [0]*STATE_COUNT
    temp_count = [1]*STATE_COUNT

    disc_state_table[0] = lower_observations

    for i in range(STATE_COUNT):
        temp_target_count[i] = ((2**STATE_COUNT)/2) + i*(0.5)

    for i in range(DISC_STATE_COUNT):

        for j in range(STATE_COUNT):
            temp_count[j] += 1

            if temp_count[j] == temp_target_count[j]:
                disc_state_table[i][j] += discrete_step_array[j]
                temp_count[j] = 1


def init_q_table_v2():
    return np.random.uniform(low=0, high=1, size=(
        DISC_STEPS, DISC_STEPS, DISC_STEPS, DISC_STEPS, env.action_space.n))


def get_descrete_state_v2(state):

    cartPositionBin = np.linspace(
        LOWER_OBSERVATIONS[0], UPPER_OBSERVATIONS[0], DISC_STEPS)
    cartVelocityBin = np.linspace(
        LOWER_OBSERVATIONS[1], UPPER_OBSERVATIONS[1], DISC_STEPS)
    poleAngleBin = np.linspace(
        LOWER_OBSERVATIONS[2], UPPER_OBSERVATIONS[2], DISC_STEPS)
    poleAngleVelocityBin = np.linspace(
        LOWER_OBSERVATIONS[3], UPPER_OBSERVATIONS[3], DISC_STEPS)

    indexPosition = np.maximum(np.digitize(state[0], cartPositionBin)-1, 0)
    indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin)-1, 0)
    indexAngle = np.maximum(np.digitize(state[2], poleAngleBin)-1, 0)
    indexAngularVelocity = np.maximum(
        np.digitize(state[3], poleAngleVelocityBin)-1, 0)

    return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])


def select_action(state):

    randomNumber = np.random.random()

    if randomNumber < EPSILON:
        return np.random.choice(env.action_space.n)

    else:
        return np.argmax(q_table[get_descrete_state_v2(state)])


total_episode_rewards = []


def plot_optimal_policy(q_table):
    policy = np.argmax(q_table, axis=4)

    plt.imshow(policy[:, :, 5, 0], cmap='coolwarm')
    plt.colorbar()
    plt.title('Optimal Policy')
    plt.xlabel('Cart Position')
    plt.ylabel('Cart Velocity')
    plt.show()


def solve():
    t = 0
    MAX_STEPS = 500
    PosWithinRange = False
    angWithinRange = False
    for episode in range(EPISODES):
        (curr_state, _) = env.reset()
        # check if the states are within an acceptable range
        if curr_state[0] < 2.4 and curr_state[0] > -2.4:
            PosWithinRange = True
        else:
            PosWithinRange = False
        if curr_state[2] < 0.2095 and curr_state[2] > -0.2095:
            angWithinRange = True
        else:
            angWithinRange = False

        print("Simulating episode {}".format(episode))

        episode_rewards = []

        is_terminal = False
        while not is_terminal and t < MAX_STEPS and PosWithinRange and angWithinRange:

            disc_curr_state = get_descrete_state_v2(curr_state)

            action = select_action(curr_state)

            (next_state, reward, is_terminal, is_truncated, _) = env.step(action)

            episode_rewards.append(reward)

            next_state = list(next_state)

            next_state_index = get_descrete_state_v2(next_state)

            q_next_state = np.max(q_table[next_state_index])

            if not (is_terminal or is_truncated):
                error = reward+DISCOUNT*q_next_state - \
                    q_table[disc_curr_state+(action,)]
                q_table[disc_curr_state+(action,)] += LEARNING_RATE*error
            else:
                error = reward-q_table[disc_curr_state+(action,)]
                q_table[disc_curr_state+(action,)] += LEARNING_RATE*error

            curr_state = next_state

        print("Sum of rewards {}".format(np.sum(episode_rewards)))
        total_episode_rewards.append(np.sum(episode_rewards))
    print("Average reward over 500 episodes: ", np.mean(total_episode_rewards))
    return total_episode_rewards, q_table


q_table = init_q_table_v2()
total_episode_rewards, q_table = solve()
plt.plot(total_episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of Q-learning Agent')
# Call this function after training the Q-learning agent
plt.show()
plot_optimal_policy(q_table)

env.close()
