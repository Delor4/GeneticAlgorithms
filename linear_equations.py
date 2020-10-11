import math
import random

import matplotlib.pyplot as plt
from drawnow import drawnow


class GeneticSystem:
    """
    Genetic algorithms
    by Sebastian Kucharczyk
    """
    GEN_MIN_VAL = -10
    GEN_MAX_VAL = 10
    # POPULATION_COUNT # set to even number
    INDIVIDUALS = 90
    GENS = 10
    WORST = 1
    MUTATED_GENS = 8
    MATRIX_A = [
        [-1, 3, -9, 5, 0, 2, 3, 9, 3, -7],
        [-9, -4, 0, 4, -3, 2, 5, -2, -5, -6],
        [-4, 3, 6, 5, -1, -6, 6, -8, -4, -2],
        [-5, -5, 0, -3, 1, 0, 3, 7, 2, 6],
        [-1, -5, -4, -3, 5, -2, 5, 9, 6, 4],
        [-8, -3, 9, -3, 2, 4, -6, -9, 0, -7],
        [6, -8, 9, 2, 1, 1, 6, 8, 1, 9],
        [2, 1, -8, -8, -1, -3, 3, 9, 6, 7, ],
        [4, -9, 3, -3, 6, -4, 8, -1, 3, 1],
        [7, 2, 7, -5, 0, -5, 2, 8, 5, 6],
        [7, 2, 7, -5, 0, -5, 2, 8, 5, 6],
        [2, 1, 5, 2, -4, 3, 1, 1]
    ]
    MATRIX_B = [- 8, - 5, 5, - 7, - 9, 0, 0, - 6, 4, - 5, - 5, ]

    # result: -3,20	2,58	3,01	-1,05	1,01	0,54	4,63	-3,34
    #1	8	1	-6	9	9	7	0	-8	1
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
        return [self.make_random_individual() for i in range(individuals)]

    def _make_ind(self, gens):
        return {
            'gens': gens,
            'fitness': self.calculate_goal_func(gens)
        }

    @staticmethod
    def randomize_gen(min_val=GEN_MIN_VAL, max_val=GEN_MAX_VAL):
        return int(random.randrange(min_val * 10, max_val * 10) / 10)

    def make_random_individual(self, gens=GENS):
        return self._make_ind([self.randomize_gen() for i in range(gens)])

    def sort_pop(self):
        try:
            self.population.sort(key=lambda ind: ind['fitness'])
        except:
            print(self.population)

    def remove_worst(self, removed=WORST):
        self.population = self.population[:-removed]

    def add_missing(self):
        # self.population.extend([self._make_ind(self.population[i]['gens']) for i in range(self.INDIVIDUALS - len(self.population))])
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
        for i in range(0, len(self.population) & ~1, 2):
            self.population[i], self.population[i + 1] = self.cross_ind(self.population[i], self.population[i + 1])

    def mutate_ind(self, ind):
        ind['gens'][random.randrange(len(ind))] = self.randomize_gen()
        return self._make_ind(ind['gens'])

    def mutate_pop(self, mutated_gens=MUTATED_GENS):
        for i in range(mutated_gens):
            index = random.randrange(len(self.population))
            self.population[index] = self.mutate_ind(self.population[index])

    def show_ind(self, msg, ind):
        print(msg, ind)

    def debug(self, msg):
        # print(msg)
        # for i in range(len(self.population)):
        #   self.show_ind(i, self.population[i])
        None

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
        s = sum([i['fitness'] for i in self.population])
        return s


def make_fig():
    plt.plot(x, y, '', x, y2)
    plt.title('Alg. genetyczny')
    plt.xlabel("Pokolenie")
    plt.ylabel("Wsp. dostosowania")
    # plt.annotate('początek', xy=(0, 0), xytext=(0.5, -0.5),
    # arrowprops=dict(facecolor='black', shrink=0.05),
    # )
    # plt.annotate('koniec', xy=(x[-1], y[-1]), xytext=(x[-1]*.8, 0.5),
    # arrowprops=dict(facecolor='black', shrink=0.05),
    # )
    plt.legend(['f(x)=wsp. dost', 'f(x)=avg'])


if __name__ == "__main__":
    pop = GeneticSystem()
    plt.ion()  # enable interactivity
    fig = plt.figure()  # make a figure
    x = []
    y = []
    y_sum = 0
    y2 = []
    print("Adjustment factor on start: {}".format(pop.best_fitness()))
    for i in range(1, 100000):
        pop.calculate_new_generation()
        if (i % 10 ** int(math.log(i, 10)) == 0):
            print("Adjustment factor on {} generation: {:.2f}, all: {:.2f}".format(i,
                                                                                   pop.best_fitness(),
                                                                                   pop.overall_fitness()),
                  pop.population[0]['gens'])
        if pop.best_fitness() < 0.5:
            break
        x.append(i)
        y.append(pop.best_fitness())
        y_sum += pop.best_fitness()
        y2.append(y_sum / i)
        if i % 100 == 0:
            x = x[-2000:]
            y = y[-2000:]
            y2 = y2[-2000:]
            drawnow(make_fig)
