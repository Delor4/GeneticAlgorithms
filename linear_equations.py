# Algorytmy genetyczne
# Sebastian Kucharczyk
import random

class GeneticSystem:
    GEN_MIN_VAL = -5
    GEN_MAX_VAL = 5
    # POPULATION_COUNT
    INDIVIDUALS = 20
    GENS = 8
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

    def calculate_goal_func(self, ind):
        mat_a = [sum([wsp * gen for gen, wsp in zip(ind, line)]) for line in self.MATRIX_A]
        y = [abs(a - b) for a, b in zip(mat_a, self.MATRIX_B)]
        return sum(y)

    def make_population(self, individuals=INDIVIDUALS):
        return [self.make_individual() for i in range(individuals)]

    @staticmethod
    def make_individual(gens=GENS, min_val=GEN_MIN_VAL, max_val=GEN_MAX_VAL):
        return [random.randrange(min_val, max_val) for i in range(gens)]


if __name__ == "__main__":
    pop = GeneticSystem()
    print(pop.calculate_goal_func([1, 1, 1, 1, 1, 1, 1, 1, 1]))
