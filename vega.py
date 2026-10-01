from base import BaseAlgorithm
import random

class VEGA(BaseAlgorithm):
    def __init__(self, filepath, generations, mutation_rate, crossover_rate, pop_size):
        super().__init__(
            filepath=filepath,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            pop_size=pop_size,
        )

    def initialize_population(self):
        '''
        individual[0] = solution
        individual[1] = f1 = opening cost
        individual[2] = f2 = customer cost
        '''
        return self.create_population()

    def vega_selection(self, population):
        shuffled_population = population.copy()
        random.shuffle(shuffled_population)

        selection_size = self.pop_size // 2

        sub_opening_cost = shuffled_population[:selection_size]
        sub_customer_cost = shuffled_population[selection_size:]
        
        # Objective 1: opening cost
        worst = max(individual[1] for individual in sub_opening_cost)
        opening_weights = [
            worst - 1 + individual[1]
            for individual in sub_opening_cost
        ]

        selected_opening = random.choices(
            sub_opening_cost,
            weights=opening_weights,
            k=selection_size
        )

        # Objective 2: customer cost
        worst = max(individual[2] for individual in sub_customer_cost)
        customer_weights = [
            worst - 1 + individual[2]
            for individual in sub_customer_cost
        ]

        selected_customer = random.choices(
            sub_customer_cost,
            weights=customer_weights,
            k=selection_size
        )

        mating_pool = selected_opening + selected_customer
        random.shuffle(mating_pool)

        return mating_pool

    def reproduce(self, mating_pool):
        offspring = []

        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i][0]
            parent2 = mating_pool[i + 1][0]

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            # evaluates children bcause next vega_selection() expects same structure
            opening_cost1, customer_cost1 = self.evaluate_solution(child1)
            opening_cost2, customer_cost2 = self.evaluate_solution(child2)

            offspring.append([child1, opening_cost1, customer_cost1])
            offspring.append([child2, opening_cost2, customer_cost2])

        return offspring

    def dominates(self, a, b):
        """Objective vector a is not worse in every objective and strictly better in at least one objective."""
        objectives_a = a[1:]
        objectives_b = b[1:]
        
        return all(x <= y for x, y in zip(objectives_a, objectives_b)) and any(
            x < y for x, y in zip(objectives_a, objectives_b)
        )

    def nondominated_solutions(self, population):
        nondominated = []

        for i, individual in enumerate(population):
            is_dominated = False

            for j, other in enumerate(population):
                if i != j and self.dominates(other, individual):
                    is_dominated = True
                    break

            if not is_dominated:
                nondominated.append(individual)

        return nondominated

    def unique_solutions(self, population):
        unique = {}

        for individual in population:
            solution = individual[0]
            key = tuple(solution)
            unique[key] = individual

        return list(unique.values())

    def run_algorithm(self):
        population = self.initialize_population()
        archive = []

        for _ in range(self.generations):
            archive.extend(population)
            archive = self.nondominated_solutions(archive)
            archive = self.unique_solutions(archive)

            mating_pool = self.vega_selection(population)
            population = self.reproduce(mating_pool)

        archive.extend(population)
        archive = self.nondominated_solutions(archive)
        archive = self.unique_solutions(archive)

        return archive

def main():
    algorithm = VEGA(
        filepath="./data/cap61.txt",
        mutation_rate=0.05,
        crossover_rate=0.2,
        pop_size=200,
        generations=100
    )

    pareto_front = algorithm.run_algorithm()

    for solution in pareto_front:
        print(solution[1], solution[2])

if __name__ == "__main__":
    main()