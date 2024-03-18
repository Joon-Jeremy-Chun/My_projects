# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:54:27 2024

@author: joonc
"""

import numpy as np

# Define the state space
states = ['rainy', 'sunny']

# Define the transition matrix (probability of transitioning from one state to another)
transition_matrix = np.array([[0.7, 0.3],  # Transition from rainy to rainy and sunny
                              [0.4, 0.6]]) # Transition from sunny to rainy and sunny

# Define the initial state distribution (probability of starting in each state)
initial_distribution = np.array([0.5, 0.5])  # Equal probability of starting in rainy or sunny

# Simulate the Markov chain for a certain number of time steps
def simulate_markov_chain(transition_matrix, initial_distribution, num_steps):
    current_state = np.random.choice(len(states), p=initial_distribution)
    chain = [states[current_state]]
    for _ in range(num_steps - 1):
        current_state = np.random.choice(len(states), p=transition_matrix[current_state])
        chain.append(states[current_state])
    return chain

# Example: Simulate the Markov chain for 100 time steps
num_steps = 100
chain = simulate_markov_chain(transition_matrix, initial_distribution, num_steps)
print("Generated Markov chain:", chain)