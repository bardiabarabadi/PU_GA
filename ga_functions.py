import numpy as np
from helper import fitness_calc


def population_fitness_calc(pop, probs):
    fitnesses = np.zeros([pop.shape[0], 1])
    for i in range(pop.shape[0]):
        fitnesses[i] = fitness_calc(pop[i], probs)
    return fitnesses.T


def select_mating_pool(population, fitness, num_parents_mating):
    strength = np.argsort(fitness)
    sorted_parents = population[strength, :]
    selected_parents = sorted_parents[0, 0:num_parents_mating, :]
    return selected_parents


def crossover(parents, offspring_size):
    offspring = np.empty(shape=offspring_size)
    mating_policy = np.random.choice(a=[0, 1], size=offspring_size, p=[0.5, 0.5])
    for child in range(offspring_size[0]):
        dad = parents[child % parents.shape[0]]
        mom = parents[(child + 1) % parents.shape[0]]
        dad = np.multiply(dad, mating_policy[child % parents.shape[0]])
        mom = np.multiply(mom, 1 - mating_policy[child % parents.shape[0]])
        offspring[child] = mom + dad
    return offspring


def mutation(children, chance, maxambig):
    random_mutation = np.random.choice(a=[0, 1, -1], size=children.shape, p=[1 - chance, chance / 2, chance / 2])
    children = children + random_mutation
    children = np.clip(children, -maxambig, maxambig)
    return children
