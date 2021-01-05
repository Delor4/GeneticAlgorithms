import math
import random
import time

import matplotlib.pyplot as plt
from drawnow import drawnow

MATRIX_A = [
    [-3, 3, 3, -4, 2, -1, 3, 1],
    [1, -3, 0, 2, 3, 1, 3, -4],
    [-1, 4, -1, -4, 1, -2, 4, -2],
    [-3, 0, 1, 0, 3, 0, 2, 1],
    [-3, -4, 1, -2, 4, 1, 3, 4],
    [-3, 4, 2, 4, -4, -4, 0, -3],
    [-4, 1, 1, 1, 3, 2, 1, -1],
    [-3, 3, 3, -4, 1, -1, 3, 1],
]

MATRIX_B = [-34, -19, -26, -21, -25, -19, -18, -33, ]

TIME_OF_LIFE = 50


def calculate_goal_func(gens):
    b = [sum([a * gen for gen, a in zip(gens, line)]) for line in MATRIX_A]
    abs_b_minus_br = [abs(br - b) for b, br in zip(b, MATRIX_B)]
    return sum(abs_b_minus_br)


class Individual:
    def __init__(self, gens, tol=TIME_OF_LIFE):
        self.gens = gens
        self.fitness = calculate_goal_func(gens)
        self.tol = tol

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return "({}, fit: {:.3f})".format(self.gens, self.fitness)

    def __len__(self):
        return len(self.gens)


class GeneticSystem:
    """
    Genetic algorithms
    by Sebastian Kucharczyk
    """
    GENERATIONS = 100000
    SHOW_DEBUG = False
    GENS = 8
    GEN_MIN_VAL = -5
    GEN_MAX_VAL = 5
    INDIVIDUALS = 100
    WORST = 14  # int(0.05 * INDIVIDUALS)
    CROSSED = 84  # int(0.90 * (INDIVIDUALS - WORST))
    MUTATED_GENS = 0  # int(0.001 * INDIVIDUALS)

    SIMULATED_ANNEALING_ITERATIONS = 10

    # expected result: [4,	2	,-3,	2,	-1,	3,	-3,	3,]
    def __init__(self):
        random.seed()
        self.population = self.make_population()
        self.sort_pop()

    def make_population(self, individuals=INDIVIDUALS):
        """
        Create initial population.
        """
        return [self.make_random_individual() for _ in range(individuals)]

    @staticmethod
    def randomize_gen(min_val=GEN_MIN_VAL, max_val=GEN_MAX_VAL):
        """
        Return randomized gen.
        """
        return random.randrange(min_val * 100, max_val * 100) / 100

    def make_random_individual(self, gens=GENS):
        """
        Return random individual.
        """
        return Individual([self.randomize_gen() for _ in range(gens)])

    def sort_pop(self):
        """
        Sort population by fitness factor.
        """
        self.population.sort(key=lambda ind: ind['fitness'])

    def selection(self, removed=WORST):
        """
        Remove individuals with worst fitness factor. (need sorted population)
        """
        if removed > 0:
            self.population = self.population[:-removed]

    def populate(self):
        """
        Populate collection with random individuals. (up to population size)
        """
        # self.population.extend([self._make_ind(self.population[i]['gens'])
        #   for _ in range(self.INDIVIDUALS - len(self.population))])
        while len(self.population) < self.INDIVIDUALS:
            self.population.append(self.make_random_individual())
        # self.population.extend(self.population[self.INDIVIDUALS - len(self.population)])

    def cross_sp_ind(self, i1, i2):
        """
        Single-point crossover. Fixed crossover point (in half of genome).
        """
        half = int(len(i1['gens']) / 2)

        gen1 = i1['gens'][:half]
        gen1.extend(i2['gens'][half:])

        gen2 = i2['gens'][:half]
        gen2.extend(i1['gens'][half:])

        return Individual(gen1), Individual(gen2)

    def cross_rp_ind(self, i1, i2):
        """
        Single-point crossover. Crossover point selected randomly.
        """
        half = int(random.randrange(1, self.GENS - 1))

        gen1 = i1['gens'][:half]
        gen1.extend(i2['gens'][half:])

        gen2 = i2['gens'][:half]
        gen2.extend(i1['gens'][half:])

        return Individual(gen1), Individual(gen2)

    def cross_dp_ind(self, i1, i2):
        """
        Swap random gen between two individuals.
        """
        i_gen = int(random.randrange(0, self.GENS))

        tmp = i1['gens'][i_gen]
        i1['gens'][i_gen] = i2['gens'][i_gen]
        i2['gens'][i_gen] = tmp

        return Individual(i1['gens']), Individual(i2['gens'])

    def cross_pop(self):
        """
        Crossover individuals in population.
        Max. CROSSED individuals grouped in pairs.
        """
        for _i in range(0, self.CROSSED & ~1, 2):
            self.population[_i], self.population[_i + 1] = self.cross_dp_ind(self.population[_i],
                                                                             self.population[_i + 1])

    def check_tol(self):
        """
            Reduce time of life and remove old individuals.
        """
        n_pop = []
        for _i in self.population:
            _i['tol'] = _i['tol'] - 1
            if _i['tol'] >= 0:
                n_pop.append(_i)

        self.population = n_pop

    def mutate_ind(self, ind):
        """
        Mutate random gen in given individual.
        """
        ind['gens'][random.randrange(len(ind['gens']))] = self.randomize_gen()
        return Individual(ind['gens'])

    def mutate_pop(self, mutated_gens=MUTATED_GENS):
        """
        Mutate individuals in population.
        """
        for _ in range(mutated_gens):
            index = random.randrange(len(self.population))
            self.population[index] = self.mutate_ind(self.population[index])

    @staticmethod
    def show_ind(msg, ind):
        """
        Helper method. For debug.
        """
        print(msg, ind)

    def debug(self, msg):
        """
        Helper method. For debug. :)
        """
        if self.SHOW_DEBUG:
            print(msg)
            for _i in range(len(self.population)):
                self.show_ind(_i, self.population[_i])

    def calculate_new_generation(self):
        """
        Calculate new iteration of genetic algorithm.
        """
        self.selection()
        # self.check_tol()
        self.debug("after remove")
        self.cross_pop()
        self.debug("After crossing:")
        self.mutate_pop()
        self.debug("After mutate:")
        self.populate()
        self.debug("After add missing:")
        self.sort_pop()
        # self.simulated_annealing(self.population[0]['fitness']/10)
        self.debug("After sort:")

    def best_fitness(self):
        """
        Returns best fitness factor from sorted population.
        """
        return self.population[0]['fitness']

    def overall_fitness(self):
        """
        Returns accumulated fitness factor of all individuals in population.
        """
        s = sum([_i['fitness'] for _i in self.population])
        return s

    def simulated_annealing_condition(self, f1, f2, c):
        out = math.exp(-(f1 - f2) / c)
        if random.random() < out:
            return True
        return False

    def simulated_annealing(self, temp, iter=SIMULATED_ANNEALING_ITERATIONS):
        for i in range(iter):
            p = self.population[0]
            p_gens = p['gens'].copy()
            index = random.randrange(len(p_gens))
            p_gens[index] += self.randomize_gen() / 100
            p_prim = Individual(p_gens)
            if p['fitness'] < p_prim['fitness']:
                self.population[0] = p_prim
                continue
            else:
                if self.simulated_annealing_condition(p['fitness'], p_prim['fitness'], temp):
                    self.population[0] = p_prim


class Fig:
    """
    Class to store and show statistics.
    """
    MAX_TO_SHOW = 2000
    REDRAW_STEP = 250

    def __init__(self):
        self.plt = plt
        plt.ion()  # enable interactivity
        self.fig = plt.figure()  # make a figure

        self.x = []
        self.y = []
        self.y2 = []

        self.y_sum = 0

        self.x_min = 0
        self.y_min = 999999

        self.actual_generation = 0
        self.actual_ind = None

    def store_results(self, generation, best_ind):
        best_fit = best_ind['fitness']
        self.actual_generation = generation
        self.actual_ind = best_ind

        self.x.append(generation)
        self.y.append(best_fit)

        self.y_sum += best_fit
        self.y2.append(self.y_sum / generation)

    def check_newbest(self):
        """
        Save and show individual if actual fitness factor is better than the past.
        """
        best_fit = self.y[-1]
        if self.y_min > best_fit:
            self.y_min = best_fit
            self.x_min = self.actual_generation
            print("New best: ", self.actual_ind)

    def get_avg(self):
        return self.y2[-1]

    def show_fig(self):

        if self.actual_generation % self.REDRAW_STEP == 0:
            self.x = self.x[-self.MAX_TO_SHOW:]
            self.y = self.y[-self.MAX_TO_SHOW:]
            self.y2 = self.y2[-self.MAX_TO_SHOW:]
            drawnow(self.draw_fig)

    def draw_fig(self):
        plt.plot(self.x, self.y, '', self.x, self.y2)
        plt.title('Alg. genetyczny')
        plt.xlabel("Pokolenie")
        plt.ylabel("Wsp. dostosowania")
        plt.legend(['wsp. dost', 'średnia'])


if __name__ == "__main__":
    pop = GeneticSystem()
    print("Population:", pop.INDIVIDUALS,
          ', removed on selection: ', pop.WORST,
          ', crossed: ', pop.CROSSED,
          ', mutated: ', pop.MUTATED_GENS,
          )
    stored_time = time.time()
    fig = Fig()
    print("Adjustment factor on start: {}".format(pop.best_fitness()))
    for i in range(1, pop.GENERATIONS):
        pop.calculate_new_generation()
        fig.store_results(i, pop.population[0])
        if i % 10 ** int(math.log(i, 10)) == 0 or stored_time > (time.time() * 60):
            stored_time = time.time()
            print("Adjustment factor on {} generation: {:.2f}, all: {:.2f}".format(i,
                                                                                   pop.best_fitness(),
                                                                                   pop.overall_fitness()),
                  pop.population[0]['gens'],
                  "avg.: {:.3f}".format(fig.get_avg())
                  )

        if pop.best_fitness() < 0.5:
            print("Find: ", pop.population[0], 'after:', i, "iterations")
            break

        fig.check_newbest()
        fig.show_fig()
