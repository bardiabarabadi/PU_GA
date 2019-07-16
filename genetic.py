from helper import *
from simulate import *
import random
from ga_functions import *

if __name__ == '__main__':
    random.seed(time.clock())
    input_size = (20, 20)
    maxambig = 1
    wrapped, unwrapped = simulate(shape=[input_size[0], input_size[1], 2], maxambig=maxambig, seed=100)
    ground_truth = (wrapped - unwrapped) / (2 * np.pi)
    probs = probs_gen(wrapped)
    ground_truth_fitness = fitness_calc(ground_truth, probs)
    # fig = plot_figure(ground_truth, ground_truth, wrapped,label=False)
    # plt.show(fig)
    # print(fitness_calc(probs, np.zeros(shape=ground_truth.size)))
    # print(fitness_calc(probs, ground_truth))
    number_of_genes = input_size[0] * input_size[1]
    solution_per_population = 100
    population_size = (solution_per_population, number_of_genes)
    population = np.random.uniform(low=-maxambig, high=maxambig, size=population_size).astype('int32')

    num_generations = 3300
    num_parents_mating = 5
    chance = 0.01

    for generation in range(num_generations):
        # Measuring the fitness of each chromosome in the population.
        fitness = population_fitness_calc(population, probs)
        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(population, fitness, num_parents_mating)
        if (generation % 100 == 0):
            print ('Best fitness in gen ' + str(generation) + ' is: ' + str(
                fitness_calc(parents[0], probs)) + ' \ Best fitness: ' + str(ground_truth_fitness))
        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                        offspring_size=(population_size[0], number_of_genes))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover, chance, maxambig)
        # Creating the new population based on the parents and offspring.
        # population[0:parents.shape[0], :] = parents
        # population[parents.shape[0]:, :] = offspring_mutation
        population = offspring_mutation

    sol = parents[0]
    fig = plot_figure(sol, ground_truth, wrapped,label=False)
    plt.show(fig)