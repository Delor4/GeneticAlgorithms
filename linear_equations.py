# Algorytmy genetyczne
# Sebastian Kucharczyk
import random


class GeneticSystem:
    GEN_MIN_VAL = -5
    GEN_MAX_VAL = 5
    # POPULATION_COUNT # set to even number
    INDIVIDUALS = 50
    GENS = 8
    WORST = 4
    MUTATED_GENS = 2
    MATRIX_A = [
        [2, 1, -1, -3, -1, 1, 4, 4],
        [4, 1, 2, -1, 2, 1, 2, 2],
        [3, 3, -3, 4, -1, 2, 1, 1],
        [3, 2, 2, -1, 1, 3, -1, -1],
        [5, 5, -3, 5, 4, 2, 3, 3],
        [4, -2, 2, -2, -3, 3, 3, 3],
        [3, 1, 0, -3, 3, -5, 3, 3],
        [2, 1, 5, 2, -4, 3, 1, 1],
    ]
    MATRIX_B = [1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(self):
        random.seed()
        self.population = self.make_population()
        self.sort_pop()
        print(len(self.population))

    def calculate_goal_func(self, ind):
        b = [sum([a * gen for gen, a in zip(ind, line)]) for line in self.MATRIX_A]
        abs_b_minus_br = [abs(br - b) for b, br in zip(b, self.MATRIX_B)]
        return sum(abs_b_minus_br)

    def make_population(self, individuals=INDIVIDUALS):
        return [self.make_individual() for i in range(individuals)]

    def _make_ind(self, gens):
        return {
            'gens': gens,
            'adj_factor': self.calculate_goal_func(gens)
        }

    @staticmethod
    def randomize_gen(min_val=GEN_MIN_VAL, max_val=GEN_MAX_VAL):
        return random.randrange(min_val * 100, max_val * 100) / 100

    def make_individual(self, gens=GENS):
        return self._make_ind([self.randomize_gen() for i in range(gens)])

    def sort_pop(self):
        self.population.sort(key=lambda ind: ind['adj_factor'])

    def remove_worst(self, removed=WORST):
        self.population = self.population[:-removed]

    def add_missing(self):
        while len(self.population) < self.INDIVIDUALS:
            self.population.append(self.make_individual())

    def cross_ind(self, i1, i2):
        half = int(len(i1['gens']) / 2)

        gen1 = i1['gens'][:half]
        gen1.extend(i2['gens'][half:])

        gen2 = i2['gens'][:half]
        gen2.extend(i1['gens'][half:])

        return self._make_ind(gen1), self._make_ind(gen2)

    def cross_pop(self):
        new_pop = []
        for i in range(0, len(self.population), 2):
            i1, i2 = self.cross_ind(self.population[i], self.population[i + 1])
            new_pop.extend([i1, i2])

        self.population = new_pop

    def mutate_ind(self, ind):

        ind['gens'][random.randrange(len(ind))] = self.randomize_gen()
        return self._make_ind(ind['gens'])

    def mutate_pop(self, mutated_gens=MUTATED_GENS):
        for i in range(mutated_gens):
            index = random.randrange(len(self.population))
            self.population[index] = self.mutate_ind(self.population[index])

    def calculate_new_population(self):
        self.remove_worst()
        self.cross_pop()
        self.mutate_pop()
        self.add_missing()
        self.sort_pop()


if __name__ == "__main__":
    pop = GeneticSystem()
    print("Adjustment factor on start: {}".format(pop.population[0]['adj_factor']))
    for i in range(1500):
        pop.calculate_new_population()
        print("Adjustment factor on {} generation: {}".format(i, pop.population[0]['adj_factor']))
