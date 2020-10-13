import math
import random

import matplotlib.pyplot as plt
from drawnow import drawnow


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
    INDIVIDUALS = 1000
    WORST = int(0.1 * INDIVIDUALS)
    CROSSED = int(0.99 * (INDIVIDUALS - WORST))
    MUTATED_GENS = int(0.01 * INDIVIDUALS)
    MATRIX_A = [
        [-3,  3,  3, -4,  2, -1, 3,  1],
        [ 1, -3,  0,  2,  3,  1, 3, -4],
        [-1,  4, -1, -4,  1, -2, 4, -2],
        [-3,  0,  1,  0,  3,  0, 2,  1],
        [-3, -4,  1, -2,  4,  1, 3,  4],
        [-3,  4,  2,  4, -4, -4, 0, -3],
        [-4,  1,  1,  1,  3,  2, 1, -1],
        [-3,  3,  3, -4,  1, -1, 3,  1],
    ]
    MATRIX_B = [-34, -19, -26, -21, -25, -19, -18, -33, ]

    # result: [4,	2	,-3,	2,	-1,	3,	-3,	3,]
    def __init__(self):
        random.seed()
        self.population = self.make_population()
        self.sort_pop()
        # print(len(self.population))

    def calculate_goal_func(self, ind):
        b = [sum([a * gen for gen, a in zip(ind, line)]) for line in self.MATRIX_A]
        abs_b_minus_br = [abs(br - b) for b, br in zip(b, self.MATRIX_B)]
        return sum(abs_b_minus_br)

    def make_population(self, individuals=INDIVIDUALS):
        return [self.make_random_individual() for _ in range(individuals)]

    def _make_ind(self, gens):
        return {
            'gens': gens,
            'fitness': self.calculate_goal_func(gens)
        }

    @staticmethod
    def randomize_gen(min_val=GEN_MIN_VAL, max_val=GEN_MAX_VAL):
        return int(random.randrange(min_val * 10, max_val * 10) / 10)

    def make_random_individual(self, gens=GENS):
        return self._make_ind([self.randomize_gen() for _ in range(gens)])

    def sort_pop(self):
        self.population.sort(key=lambda ind: ind['fitness'])

    def remove_worst(self, removed=WORST):
        if removed > 0:
            self.population = self.population[:-removed]

    def add_missing(self):
        # self.population.extend([self._make_ind(self.population[i]['gens'])
        #   for _ in range(self.INDIVIDUALS - len(self.population))])
        while len(self.population) < self.INDIVIDUALS:
            self.population.append(self.make_random_individual())
        # self.population.extend(self.population[self.INDIVIDUALS - len(self.population)])

    def cross_ind(self, i1, i2):
        half = int(len(i1['gens']) / 2)

        gen1 = i1['gens'][:half]
        gen1.extend(i2['gens'][half:])

        gen2 = i2['gens'][:half]
        gen2.extend(i1['gens'][half:])

        return self._make_ind(gen1), self._make_ind(gen2)

    def cross_pop(self):
        for _i in range(0, self.CROSSED & ~1, 2):
            self.population[_i], self.population[_i + 1] = self.cross_ind(self.population[_i], self.population[_i + 1])

    def mutate_ind(self, ind):
        ind['gens'][random.randrange(len(ind))] = self.randomize_gen()
        return self._make_ind(ind['gens'])

    def mutate_pop(self, mutated_gens=MUTATED_GENS):
        for _ in range(mutated_gens):
            index = random.randrange(len(self.population))
            self.population[index] = self.mutate_ind(self.population[index])

    @staticmethod
    def show_ind(msg, ind):
        print(msg, ind)

    def debug(self, msg):
        if self.SHOW_DEBUG:
            print(msg)
            for _i in range(len(self.population)):
                self.show_ind(_i, self.population[_i])

    def calculate_new_generation(self):
        self.remove_worst()
        self.debug("after remove")
        self.cross_pop()
        self.debug("After crossing:")
        self.mutate_pop()
        self.debug("After mutate:")
        self.add_missing()
        self.debug("After add missing:")
        self.sort_pop()
        self.debug("After sort:")

    def best_fitness(self):
        return self.population[0]['fitness']

    def overall_fitness(self):
        s = sum([_i['fitness'] for _i in self.population])
        return s


class Fig:
    MAX_TO_SHOW = 2000

    def __init__(self):
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
        best_fit = self.y[-1]
        if self.y_min > best_fit:
            self.y_min = best_fit
            self.x_min = self.actual_generation
            print("New best: ", self.actual_ind)

    def get_avg(self):
        return self.y2[-1]

    def show_fig(self):

        if self.actual_generation % 100 == 0:
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
    fig = Fig()
    print("Adjustment factor on start: {}".format(pop.best_fitness()))
    for i in range(1, pop.GENERATIONS):
        pop.calculate_new_generation()
        fig.store_results(i, pop.population[0])
        if i % 10 ** int(math.log(i, 10)) == 0:
            print("Adjustment factor on {} generation: {:.2f}, all: {:.2f}".format(i,
                                                                                   pop.best_fitness(),
                                                                                   pop.overall_fitness()),
                  pop.population[0]['gens'],
                  "avg.:", fig.get_avg()
                  )
        if pop.best_fitness() < 0.5:
            print("Find: ", pop.population[0], 'after:', i)
            break

        fig.check_newbest()
        fig.show_fig()
