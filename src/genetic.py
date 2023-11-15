import copy
from operator import itemgetter
import random
from typing import Callable, Optional, Tuple

MINIMUM_DIRECTION = 0
MAXIMUM_DIRECTION = 1


class GeneticAlgorithm:
    """
    Genetic algorithm (GA) class for search and optimization problems.

    :param direction: Optimization direction, minimum (0) or maximum (1).
    :param genes: Elements from which the individuals are formed from.
    :param population_size: Number of individuals in the population.
    :param num_generations: Number of generations.
    :param elite_size: Number of elite individuals.
    :param mutation_rate: Possibility of the mutation to happen.
    :param fitness_function: Function that computes the fitness of the individual.
    :param kwargs: Optional key-word arguments for the fitness function.

    """

    def __init__(
        self,
        direction: int,
        genes: list,
        population_size: int,
        num_generations: int,
        elite_size: int,
        mutation_rate: float,
        fitness_function: Callable[[list, Optional[dict]], float],
    ) -> None:
        self.direction = direction
        self.genes = genes
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function

        self.abort = False
        self.children = []
        self.mating_pool = []
        self.population = []
        self.sorted_fit = []
        self.fitness_progress = []

    def _print_progress(self, generation: int):
        """
        Print the progress of the GA computation.

        :param generation: Current generation.

        """
        prog = str(generation).zfill(len(str(self.num_generations)))
        txt1 = f"Generation {prog}/{self.num_generations}"
        txt2 = f"Fitness {self.fitness_progress[-1]}"
        text = f"GA: {txt1}, {txt2}\r"
        print(text, end="", flush=True)

    def _update_progress(self):
        """Update the progress of the fitness computation."""
        best_fitness = self.sorted_fit[0][1]
        self.fitness_progress.append(best_fitness)

    def _initialize_population(self):
        """Create a population of randomly chosen individuals."""
        self.population = []
        gene_count = len(self.genes)
        for _ in range(self.population_size):
            individual = random.sample(self.genes, gene_count)
            self.population.append(individual)

    def _compute_fitness(self):
        """Compute the fitness of the individuals in the population."""
        fit = []
        for index in range(self.population_size):
            individual = self.population[index]
            fitness = self.fitness_function(self, individual, index)
            fit.append((index, fitness))

        # Sort the fitness list in ascending or in descending order.
        reverse = False
        if self.direction == MINIMUM_DIRECTION:
            reverse = False
        elif self.direction == MAXIMUM_DIRECTION:
            reverse = True
        self.sorted_fit = []
        self.sorted_fit = sorted(fit, key=itemgetter(1), reverse=reverse)

    def _selection(self):
        """Select a part of the population to the mating pool."""
        selected_indices = []

        data = []

        for i, item in enumerate(self.sorted_fit):
            fitness = item[1]
            cumulative_sum = fitness if i == 0 else data[i - 1][1] + fitness
            percent = 0.0
            data.append([fitness, cumulative_sum, percent])

        total_fitness_sum = data[-1][1]

        for i in range(len(data)):
            data[i][2] = 100 * data[i][1] / total_fitness_sum

        for i in range(self.elite_size):
            selected_indices.append(self.sorted_fit[i][0])

        for i in range(self.population_size - self.elite_size):
            value = 100 * random.random()
            for j in range(self.population_size):
                if value <= data[j][2]:
                    selected_indices.append(self.sorted_fit[j][0])
                    break
        self.mating_pool = []
        for i in range(len(selected_indices)):
            individual = self.population[selected_indices[i]]
            self.mating_pool.append(individual)

    def _breed(self, parent1: list, parent2: list) -> list:
        """Breed two parent solutions to get a child solution"""
        child = []
        child_p1 = []
        child_p2 = []

        # Two randomly picked crossover points.
        point1 = int(random.random() * len(parent1))
        point2 = int(random.random() * len(parent1))
        start_point = min(point1, point2)
        end_point = max(point1, point2)

        for i in range(start_point, end_point):
            child_p1.append(parent1[i])

        child_p2 = [item for item in parent2 if item not in child_p1]

        # Child solution is a combination of the properties of the
        # parent solutions.
        child = child_p1 + child_p2
        return child

    def _crossover(self):
        """Combine genetic information of two parents to form a new child."""
        self.children = []

        for i in range(self.elite_size):
            individual = self.mating_pool[i]
            self.children.append(individual)

        pool_size = len(self.mating_pool)
        pool = random.sample(self.mating_pool, pool_size)
        for i in range(pool_size - self.elite_size):
            parent1 = pool[i]
            parent2 = pool[pool_size - i - 1]
            child = self._breed(parent1, parent2)
            self.children.append(child)

    def _mutate(self, individual: list) -> list:
        """Swap the genes of the individual."""
        gene_count = len(individual)
        for i in range(gene_count):
            if random.random() < self.mutation_rate:
                j = int(random.random() * gene_count)
                gene1 = individual[i]
                gene2 = individual[j]
                individual[i] = gene2
                individual[j] = gene1
        return individual

    def _mutation(self):
        """Mutate the population in order to maintain genetic diversity."""
        mutated_population = []

        for i in range(self.population_size):
            individual = self.children[i]
            mutated_individual = self._mutate(individual)
            mutated_population.append(mutated_individual)

        self.population = copy.deepcopy(mutated_population)

    def run(self) -> Tuple[list, float]:
        """
        Run the genetic algorithm.

        :returns: The best solution and the best fitness value.
        :rtype: tuple.

        """
        print("GA: Initializing")
        self._initialize_population()
        self._compute_fitness()

        try:
            for i in range(self.num_generations):
                self._selection()
                self._crossover()
                self._mutation()
                self._compute_fitness()
                self._update_progress()
                self._print_progress(generation=i + 1)
        except KeyboardInterrupt:
            self.abort = True

        print("\nGA: Aborted") if self.abort else print("\nGA: Ready")

        best_solution = self.population[self.sorted_fit[0][0]]
        best_fitness = self.sorted_fit[0][1]

        return (best_solution, best_fitness)

    def get_progress(self) -> list:
        """
        Get the progression of the fitness value.

        :returns: The best fitness value of each generation.
        :rtype: list.

        """
        return self.fitness_progress
