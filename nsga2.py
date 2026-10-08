import os
import random
from dataclasses import dataclass
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from base import *


# dataclass representing an individual solution with costs, rank and crowding distance
@dataclass(slots=True)
class Individual:
    solution: list[int] | None = None
    opening_cost: float = 0.0
    customer_cost: float = 0.0
    domination_count: int = 0
    rank: int = 0
    crowding_distance: float = 0.0


class NSGA2(BaseAlgorithm):
    # creates initial population, evaluates costs and assigns ranks and crowding distances
    def create_population(self):
        solutions = [self.create_solution() for _ in range(self.pop_size)]

        population = []
        for solution in solutions:
            opening_cost, customer_cost = self.evaluate_solution(solution)
            population.append(Individual(solution, opening_cost, customer_cost))

        return self.assign_rank_and_distance(population)

    # calculates crowding distance for each individual in a front to maintain diversity
    def calculate_crowding_distance(self, front, population):
        distances = {index: 0.0 for index in front}

        # assign infinite distance if front has two or fewer individuals
        if len(front) <= 2:
            for index in front:
                distances[index] = float("inf")
            return distances

        # calculate crowding distance for each objective
        for obj_attr in ["opening_cost", "customer_cost"]:
            sorted_front = sorted(
                front, key=lambda idx: getattr(population[idx], obj_attr)
            )

            # boundary points get infinite distance
            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            min_val = getattr(population[sorted_front[0]], obj_attr)
            max_val = getattr(population[sorted_front[-1]], obj_attr)
            val_range = max_val - min_val

            if val_range == 0:
                continue
            # add normalized distance to neighboring solutions
            for k in range(1, len(sorted_front) - 1):
                prev_val = getattr(population[sorted_front[k - 1]], obj_attr)
                next_val = getattr(population[sorted_front[k + 1]], obj_attr)
                distances[sorted_front[k]] += (next_val - prev_val) / val_range

        return distances

    # performs fast non-dominated sorting and returns list of fronts
    def non_dominated_sort(self, population):
        P = population
        # initialize a list to hold the solutions dominated by a given solution
        S = [[] for _ in range(len(P))]
        # initialize a list to hold the domination count of each individual in pop
        n = [0] * len(P)
        fronts = [[]]

        # find domination relationships between all pairs of individuals
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if (
                    P[i].opening_cost <= P[j].opening_cost
                    and P[i].customer_cost <= P[j].customer_cost
                ) and (
                    P[i].opening_cost < P[j].opening_cost
                    or P[i].customer_cost < P[j].customer_cost
                ):
                    S[i].append(j)
                    n[j] += 1
                elif (
                    P[j].opening_cost <= P[i].opening_cost
                    and P[j].customer_cost <= P[i].customer_cost
                ) and (
                    P[j].opening_cost < P[i].opening_cost
                    or P[j].customer_cost < P[i].customer_cost
                ):
                    S[j].append(i)
                    n[i] += 1

        # first front contains all non-dominated individuals
        fronts[0] = [index for index, value in enumerate(n) if value == 0]

        # iteratively build subsequent fronts by decrementing domination counts
        current_front_index = 0
        while fronts[current_front_index]:
            next_front = []
            for front in fronts[current_front_index]:
                for q in S[front]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            current_front_index += 1
            if not next_front:
                break
            fronts.append(next_front)

        return fronts

    # sorts population into non-dominated fronts and assigns rank and crowding distance
    def assign_rank_and_distance(self, population):
        fronts = self.non_dominated_sort(population)
        for front_index, front in enumerate(fronts):
            distances = self.calculate_crowding_distance(front, population)
            for index in front:
                population[index].rank = front_index
                population[index].crowding_distance = distances[index]
        return population

    # generates offspring, combines with parents and selects best individuals using elitism
    def create_new_population(self, parent_population):
        new_population = []
        child_population = self.create_children(parent_population)
        # combine parents and offspring into 2N population
        population = parent_population + child_population
        fronts = self.non_dominated_sort(population)

        for front_index, front in enumerate(fronts):
            distances = self.calculate_crowding_distance(front, population)
            for index in front:
                population[index].rank = front_index
                population[index].crowding_distance = distances[index]

            # if current front cannot fit completely, sort by crowding distance and pick most diverse
            if len(front) + len(new_population) > self.pop_size:
                sorted_front = sorted(
                    front, key=lambda idx: distances[idx], reverse=True
                )
                remaining = self.pop_size - len(new_population)
                for index in sorted_front[:remaining]:
                    new_population.append(population[index])
                break
            # otherwise add all individuals from the front
            else:
                for index in front:
                    new_population.append(population[index])
                if len(new_population) == self.pop_size:
                    break

        return new_population

    # selects individual using tournament selection based on rank and crowding distance
    def tournament_selection(self, population):
        candidates = random.sample(population, self.tournament_size)
        return min(candidates, key=lambda ind: (ind.rank, -ind.crowding_distance))

    # creates offspring population through selection, crossover, mutation and repair
    def create_children(self, parent_population):
        children_cost_list = []
        children = []
        while len(children) < self.pop_size:
            # select two parents using tournament selection
            parent1 = self.tournament_selection(parent_population)
            parent2 = self.tournament_selection(parent_population)

            parent_solution1 = parent1.solution
            parent_solution2 = parent2.solution

            # apply crossover based on crossover rate
            if random.random() <= self.crossover_rate:
                child1, child2 = self.crossover(parent_solution1, parent_solution2)
            else:
                child1 = parent_solution1.copy()
                child2 = parent_solution2.copy()

            # mutate and repair both children
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            child1 = self.feasibility_repair(child1)
            child2 = self.feasibility_repair(child2)

            children.append(child1)
            if len(children) < self.pop_size:
                children.append(child2)

        # evaluate costs and wrap into individual objects
        for child in children:
            opening_cost, customer_cost = self.evaluate_solution(child)
            children_cost_list.append(Individual(child, opening_cost, customer_cost))

        return children_cost_list

    # runs the NSGA-II algorithm for the given number of generations
    def run(self):
        population = self.create_population()
        for _ in range(0,self.generations):
            population = self.create_new_population(population)
        return population


# loads problem instance, runs NSGA-II and plots the pareto front
def main():
    path = "./data/cap61.txt"
    instance = Problem.from_file(path)
    algorithm = NSGA2(
        instance,
        generations=1000,
        mutation_rate=0.05,
        crossover_rate=0.8,
        pop_size=200
    )

    final_population = algorithm.run()

    # extract non-dominated solutions (rank 0)
    front = sorted({
        (ind.opening_cost, ind.customer_cost)
        for ind in final_population
        if ind.rank == 0
    })
    opening, customer = zip(*front)

    # plot Pareto front
    plt.plot(opening, customer, "o-")
    plt.xlabel("Facility opening cost")
    plt.ylabel("Customer allocation cost")
    plt.title("Pareto front – cap61")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
