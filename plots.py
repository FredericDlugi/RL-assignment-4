import os.path

import numpy as np
import matplotlib.pyplot as plt

members = ['franek', 'frederic', 'moritz', 'robin']



def plot_task1():
    # Rewards (Learning Curves?)
    for i,member in enumerate(members):
        plt.subplot(2,2,i+1)
        plt.yscale('symlog')
        data = np.load(os.path.join('data', 'task_1', member + '.npz'))
        plt.plot(data['all_rewards'])
        plt.title(member)
        plt.xlabel('episode')
        plt.ylabel('reward')


    plt.suptitle('Rewards')
    plt.show()

    # Losses
    for i, member in enumerate(members):
        plt.subplot(2, 2, i + 1)
        data = np.load(os.path.join('data', 'task_1', member + '.npz'))
        plt.plot(data['losses'])
        plt.title(member)
        plt.xlabel('frame')
        plt.ylabel('loss')

    plt.show()

    # Q Values
    for i, member in enumerate(members):
        plt.subplot(2, 2, i + 1)
        data = np.load(os.path.join('data', 'task_1', member + '.npz'))
        plt.plot(data['est_Q_values_running_network'], color='r')
        plt.plot(data['est_Q_values_target_network'], color='b')
        plt.legend(['running-network', 'target-network'])
        plt.ylabel('Expected Return')
        plt.xlabel('Frame')


    plt.suptitle('Q Values')
    plt.suptitle('Q Values')
    plt.show()


if __name__ == '__main__':
    plot_task1()