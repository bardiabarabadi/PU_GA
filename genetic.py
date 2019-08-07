from helper import *
from simulate import *
import random
from ga_functions import *

from matplotlib import cm

if __name__ == '__main__':

    # Make data.
    X = np.arange(-2, 2, 0.25)
    Y = np.arange(-2, 2, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = 2 * (np.sin(R) + 1)

    random.seed(time.clock())
    global input_size
    input_size = (16, 16)
    maxambig = 1
    # wrapped, unwrapped = simulate(shape=[input_size[0], input_size[1], 2], maxambig=1, seed=100)
    unwrapped = Z
    wrapped = wrap(Z)

    ground_truth = (wrapped - unwrapped) / (2 * np.pi)
    probs = probs_gen(wrapped)
    ground_truth_fitness = fitness_calc(ground_truth, probs)
    # fig = plot_figure(ground_truth, ground_truth, wrapped,label=False)
    # plt.show(fig)
    # print(fitness_calc(probs, np.zeros(shape=ground_truth.size)))
    # print(fitness_calc(probs, ground_truth))
    number_of_genes = input_size[0] * input_size[1]
    solution_per_population = 5000
    population_size = (solution_per_population, number_of_genes)
    population = np.zeros(population_size).astype('int32')
    # population = np.random.uniform(low=-maxambig - 1, high=maxambig + 1, size=population_size).astype('int32')

    num_generations = 50000
    num_parents_mating = 5
    chance = 6 / 256

    for generation in range(num_generations):
        if generation > 3000:
            num_parents_mating = 2
            chance = 0.008

        # Measuring the fitness of each chromosome in the population.
        fitness = population_fitness_calc(population, probs)
        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                        offspring_size=(population_size[0] - num_parents_mating, number_of_genes))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover, chance)
        # Creating the new population based on the parents and offspring.
        # population[0:parents.shape[0], :] = parents
        # population[parents.shape[0]:, :] = offspring_mutation
        population = np.concatenate((offspring_mutation, parents), axis=0)

        sol = parents[0]

        if generation % 100 == 0:
            plt.close('all')
            fitnesses = [fitness_calc(parents[x], probs) for x in range(5)]
            avg_10_fitness = np.mean(fitnesses)
            print('Avg 10 best fitness in gen ' + str(generation) + ' is: ' + str(
                avg_10_fitness) + ' \ Best fitness: ' + str(ground_truth_fitness))

            fig = plot_figure(sol, ground_truth, wrapped, label=False)
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            # plt.show(fig)
            os.mkdir('./figs' + str(num_generations) + '_' + str(chance) + '_' + str(
                solution_per_population))
            figname = './figs' + str(num_generations) + '_' + str(chance) + '_' + str(
                solution_per_population) + '/' + str(generation / 100) + '.jpg'
            fig.set_size_inches((15, 10), forward=False)
            fig.savefig(figname, dpi=200)

            if fitness_calc(sol,probs)==0:
                break
