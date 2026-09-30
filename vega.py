from base import BaseAlgorithm
import random


class VEGA(BaseAlgorithm):
    def initialize_population(self):
        """
        individual[0] = solution
        individual[1] = f1 = opening cost
        individual[2] = f2 = customer cost
        """
        return self.create_population()

    def vega_selection(self, population):
        shuffled_population = population.copy()
        random.shuffle(shuffled_population)

        selection_size = self.pop_size // 2

        sub_opening_cost = shuffled_population[:selection_size]
        sub_customer_cost = shuffled_population[selection_size:]

        mating_pool = []

        # Objective 1: opening cost
        opening_weights = [
            1.0 / (1.0 + individual[1]) for individual in sub_opening_cost
        ]

        selected_opening = random.choices(
            sub_opening_cost, weights=opening_weights, k=selection_size
        )

        # Objective 2: customer cost
        customer_weights = [
            1.0 / (1.0 + individual[2]) for individual in sub_customer_cost
        ]

        selected_customer = random.choices(
            sub_customer_cost, weights=customer_weights, k=selection_size
        )

        mating_pool.extend(selected_opening)
        mating_pool.extend(selected_customer)

        random.shuffle(mating_pool)

        return mating_pool

    def reproduce(self, mating_pool):
        offspring = []

        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i][0]
            parent2 = mating_pool[i + 1][0]

            if random.random() <= self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            child1 = self.feasibility_repair(child1)
            child2 = self.feasibility_repair(child2)


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

    def run(self):
        population = self.initialize_population()

        for _ in range(self.generations):
            mating_pool = self.vega_selection(population)
            population = self.reproduce(mating_pool)

        pareto_front = self.nondominated_solutions(population)

        return pareto_front


def main():
    algorithm = VEGA(
        filepath="./data/cap61.txt",
        mutation_rate=0.05,
        crossover_rate=0.8,
        pop_size=200,
        generations=100,
    )

    pareto_front = algorithm.run()

    for solution in pareto_front:
        print(solution[1], solution[2])


if __name__ == "__main__":
    main()
