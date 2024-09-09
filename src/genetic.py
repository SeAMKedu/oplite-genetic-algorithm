import copy
import random
from enum import Enum
from operator import itemgetter
from typing import Callable, Tuple


class Direction(Enum):
    MINIMUM = 0
    MAXIMUM = 1


class CrossoverOperator(Enum):
    ONE_POINT = 1
    MULTI_POINT = 2
    UNIFORM = 3
    PMX = 4


class SelectionOperator(Enum):
    PROPORTIONATE = 1
    RANKING = 2
    TOURNAMENT = 3


class MutationType(Enum):
    BIT_FLIP = 1
    SWAP = 2
    SCRAMBLE = 3
    INVERSION = 4


class GeneticAlgorithm:
    """
    Genetic algorithm (GA) for search and optimization problems.

    :param direction: Optimization direction, minimum (0) or maximum (1).
    :param genes: Elements from which the chromosomes are formed from.
    :param population_size: Number of chromosomes in the population.
    :param num_generations: Number of generations.
    :param elite_size: Number of elite chromosomes.
    :param mutation_rate: Possibility of the mutation to happen.
    :param fitness_function: Function that computes the fitness of the individual.
    :param selection_opr: Selection operator.
    :param crossover_opr: Crossover operator.
    :param mutation_opr: Mutation operator.
    :param kwargs: Optional key-word arguments for the fitness function.

    """

    def __init__(
            self,
            direction: Direction,
            genes: list,
            population_size: int,
            num_generations: int,
            elite_size: int,
            mutation_rate: float,
            fitness_function: Callable,
            selection_opr: SelectionOperator,
            crossover_opr: CrossoverOperator,
            mutation_opr: MutationType,
            **kwargs
        ) -> None:
        self.direction = direction
        self.genes = genes
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.selection_opr = selection_opr
        self.crossover_opr = crossover_opr
        self.mutation_opr = mutation_opr
        self.kwargs = kwargs

        self.num_genes = len(genes)
        self.abort = False
        self.children = []
        self.mating_pool = []
        self.population = []
        self.fitnesses = []
        self.fitness_progress = []

    def _initialize_population(self):
        """Create a population of randomly chosen chromosomes."""
        self.population = []
        for _ in range(self.population_size):
            chromosome = random.sample(self.genes, self.num_genes)
            self.population.append(chromosome)

    def _compute_fitness(self):
        """Compute the fitness of the chromosomes in the population."""
        self.fitnesses = []
        fit = []
        for index in range(self.population_size):
            chromosome = self.population[index]
            fitness = self.fitness_function(self, chromosome, index)
            fit.append((index, fitness, chromosome))
        # Sort list by fitness value.
        reverse = False if self.direction == Direction.MINIMUM.value else True
        self.fitnesses = sorted(fit, key=itemgetter(1), reverse=reverse)

    #------------------------------------------------------------------
    # Selection
    #------------------------------------------------------------------
    def _proportionate_selection(self) -> int:
        # TODO: implemantation
        return 0

    def _ranking_selection(self) -> int:
        # TODO: implemantation
        return 0

    def _tournament_selection(self, k: int = 3) -> int:
        """Select K chromosomes and choose the best to be as a parent."""
        if self.direction == Direction.MINIMUM.value:
            best_fitness = 1_000_000_000
        elif self.direction == Direction.MAXIMUM.value:
            best_fitness = 0
        best_index = 0
        for _ in range(k):
            i = random.randint(0, self.population_size-1)
            index, fitness, chromosome = self.fitnesses[i]
            if self.direction == Direction.MINIMUM.value:
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_index = index
            if self.direction == Direction.MAXIMUM.value:
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_index = index
        return best_index

    def _select(self):
        """Select parents to get offspring for the next generation."""
        indices = []
        # Indices of the elite parents.
        for i in range(self.elite_size):
            index, _, _ = self.fitnesses[i]
            indices.append(index)
        # Indices of the other parents.
        for i in range(self.elite_size, self.population_size):
            index = None
            if self.selection_opr == SelectionOperator.PROPORTIONATE.value:
                index = self._proportionate_selection()
            elif self.selection_opr == SelectionOperator.RANKING.value:
                index = self._ranking_selection()
            elif self.selection_opr == SelectionOperator.TOURNAMENT.value:
                index = self._tournament_selection()
            indices.append(index)
        # Populate the mating pool.
        del self.mating_pool[:]
        for index in indices:
            chromosome = self.population[index]
            self.mating_pool.append(chromosome)

    #------------------------------------------------------------------
    # Crossover
    #------------------------------------------------------------------
    def _one_point_crossover(self, parent1: list, parent2: list):
        """Swap the genes of the parents after a random crossover point."""
        index = random.randint(0, len(parent1)-1)
        child1 = parent1[0:index] + parent2[index:]
        child2 = parent2[0:index] + parent1[index:]
        return (child1, child2)

    def _multi_point_crossover(self, parent1: list, parent2: list):
        """Swap the alternating segments of the parents."""
        index1 = random.randint(1, len(parent1)-1)
        index2 = random.randint(1, len(parent2)-1)
        a = min(index1, index2)
        b = max(index1, index2)
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if i < a or i > b:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return (child1, child2)

    def _uniform_crossover(self, parent1: list, parent2: list):
        """Choose gene by gene from which parent the child gets it from."""
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            value = random.random()
            if value < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return (child1, child2)

    def _pmx_crossover(self, parent1: list, parent2: list):
        """Partially matched crossover (PMX)."""
        # Select two random crossover points, (point1 < point2).
        point1 = random.randint(1, int(len(parent1)/2))
        point2 = random.randint(point1 + 1, len(parent2) - 1)
        # Exchange the genes between the crossover points.
        genes1 = parent1[point1:point2]
        genes2 = parent2[point1:point2]
        child1 = parent1[0:point1] + genes2 + parent1[point2:]
        child2 = parent2[0:point1] + genes1 + parent2[point2:]
        # Exchange the doubles.
        for point, gene in enumerate(child1):
            if point1 <= point < point2: # leave exchanged genes intact
                continue
            if not gene in genes2:
                continue
            index = genes2.index(gene)
            exchange_gene = 0
            while True:
                exchange_gene = genes1[index]
                if not exchange_gene in genes2:
                    break
                index = genes2.index(exchange_gene)
            child1[point] = exchange_gene
            for _point, _gene in enumerate(child2):
                if point1 <= _point < point2:
                    continue
                if _gene == exchange_gene:
                    child2[_point] = gene
        return child1, child2

    def _crossover(self):
        """Transfer the properties of the parents to the offspring."""
        del self.children[:]
        # Elitism.
        for i in range(self.elite_size):
            chromosome = self.mating_pool[i]
            self.children.append(chromosome)
        # Rest of the children.
        for i in range(self.elite_size, self.population_size, 2):
            parent1 = self.mating_pool[i]
            parent2 = self.mating_pool[i+1]
            if self.crossover_opr == CrossoverOperator.ONE_POINT.value:
                child1, child2 = self._one_point_crossover(parent1, parent2)
            elif self.crossover_opr == CrossoverOperator.MULTI_POINT.value:
                child1, child2 = self._multi_point_crossover(parent1, parent2)
            elif self.crossover_opr == CrossoverOperator.UNIFORM.value:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            elif self.crossover_opr == CrossoverOperator.PMX.value:
                child1, child2 = self._pmx_crossover(parent1, parent2)
            self.children.append(child1)
            self.children.append(child2)

    #------------------------------------------------------------------
    # Mutation
    #------------------------------------------------------------------
    def _bit_flip_mutation(self, chromosome: list) -> list:
        """On binary representation, change a random 0 to 1 or vice versa."""
        chrom = copy.copy(chromosome)
        index = random.randint(0, len(chrom)-1)
        value = chrom[index]
        chrom[index] = 1 if value == 0 else 0
        return chrom

    def _swap_mutation(self, chromosome: list) -> list:
        """Swap two random genes on the chromosome."""
        chrom = copy.copy(chromosome)
        index1 = random.randint(0, len(chrom)-1)
        index2 = random.randint(0, len(chrom)-1)
        value1 = chrom[index1]
        value2 = chrom[index2]
        chrom[index1] = value2
        chrom[index2] = value1
        return chrom

    def _scramble_mutation(self, chromosome: list) -> list:
        """Scramble a random subset of genes on the chromosome."""
        chrom = copy.copy(chromosome)
        index = random.randint(0, len(chrom)-1)
        size = random.randint(1, len(chrom)-index)
        subset = chrom[index:index+size]
        scrambled = random.sample(subset, len(subset))
        chrom = chrom[0:index] + scrambled + chrom[index+size:]
        return chrom

    def _inversion_mutation(self, chromosome: list) -> list:
        """Invert a random subset of genes on the chromosome."""
        chrom = copy.copy(chromosome)
        index = random.randint(0, len(chrom)-1)
        size = random.randint(1, len(chrom)-index)
        subset = chrom[index:index+size]
        subset.reverse()
        chrom = chrom[0:index] + subset + chrom[index+size:]
        return chrom

    def _mutate(self):
        """Mutate the population to preserve its diversity."""
        pop = []
        for i in range(self.population_size):
            chromosome = self.children[i]
            if random.random() < self.mutation_rate:
                if self.mutation_opr == MutationType.BIT_FLIP.value:
                    chromosome = self._bit_flip_mutation(chromosome)
                elif self.mutation_opr == MutationType.SWAP.value:
                    chromosome = self._swap_mutation(chromosome)
                elif self.mutation_opr == MutationType.SCRAMBLE.value:
                    chromosome = self._scramble_mutation(chromosome)
                elif self.mutation_opr == MutationType.INVERSION.value:
                    chromosome = self._inversion_mutation(chromosome)
            pop.append(chromosome)
        self.population = copy.deepcopy(pop)

    def run(self) -> Tuple[list, float]:
        """
        Run the genetic algorithm.

        :returns: The best solution and the best fitness value.
        :rtype: tuple.

        """
        print("GA: Initializing...")
        self._initialize_population()
        self._compute_fitness()
        print("GA: Running...")
        try:
            for i in range(self.num_generations):
                self._select()
                self._crossover()
                self._mutate()
                self._compute_fitness()
                gen = i + 1
                idx, fit, sol = self.fitnesses[0]
                print(f"GA: Generation {gen}: Fitness: {fit}")
        except KeyboardInterrupt:
            self.abort = True

        print("GA: Aborted") if self.abort else print("GA: Ready")

        best_solution = self.population[self.fitnesses[0][0]]
        best_fitness = self.fitnesses[0][1]

        return (best_solution, best_fitness)
