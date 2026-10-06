import random
from dataclasses import dataclass


@dataclass(slots=True)
class Problem:
    warehouse_count: int
    customer_count: int
    warehouse_data: list
    customer_data: list

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "r") as file:
            words = iter(file.read().split())

        warehouse_count = int(next(words))
        customer_count = int(next(words))

        warehouse_data = [
            (int(next(words)), float(next(words)))
            for _ in range(warehouse_count)
        ]

        customer_data = []

        for _ in range(customer_count):
            demand = int(next(words))
            costs = [
                float(next(words))
                for _ in range(warehouse_count)
            ]
            customer_data.append((demand, costs))

        return cls(
            warehouse_count,
            customer_count,
            warehouse_data,
            customer_data,
        )

class BaseAlgorithm:
    def __init__(self, instance, generations, mutation_rate, crossover_rate, pop_size):
        self.instance = instance
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.pop_size = pop_size
        self.tournament_size = 2

    def create_solution(self):
        # index=customer; entry=warehouse
        solution = []

        for _ in range(self.instance.customer_count):
            solution.append(random.randint(0, self.instance.warehouse_count - 1))
        solution = self.feasibility_repair(solution)

        return solution

    def feasibility_repair(self, solution):
        solution = solution.copy()
        unassigned_customers = []
        facility_usage = [0] * self.instance.warehouse_count

        for customer in range(len(solution)):
            warehouse = solution[customer]
            demand = self.instance.customer_data[customer][0]
            if facility_usage[warehouse] + demand <= self.instance.warehouse_data[warehouse][0]:
                facility_usage[warehouse] += demand
            else:
                unassigned_customers.append(customer)

        for customer in unassigned_customers:
            demand = self.instance.customer_data[customer][0]
            costs = self.instance.customer_data[customer][1]

            open_facilities_with_space = [
                f
                for f in range(self.instance.warehouse_count)
                if facility_usage[f] > 0
                and facility_usage[f] + demand <= self.instance.warehouse_data[f][0]
            ]

            if open_facilities_with_space:
                best_facility = min(open_facilities_with_space, key=lambda f: costs[f])
            else:
                closed_facilities_with_space = [
                    f
                    for f in range(self.instance.warehouse_count)
                    if facility_usage[f] == 0 and demand <= self.instance.warehouse_data[f][0]
                ]
                if closed_facilities_with_space:
                    best_facility = min(
                        closed_facilities_with_space,
                        key=lambda f: self.instance.warehouse_data[f][1] + costs[f],
                    )
                else:
                    best_facility = max(
                        range(self.instance.warehouse_count),
                        key=lambda f: self.instance.warehouse_data[f][0] - facility_usage[f],
                    )

            solution[customer] = best_facility
            facility_usage[best_facility] += demand

        return solution

    def crossover(self, solution1, solution2):
        child_solution1 = [None] * len(solution1)
        child_solution2 = [None] * len(solution1)

        for index in range(len(solution1)):
            if random.random() < 0.5:
                child_solution1[index] = solution1[index]
                child_solution2[index] = solution2[index]
            else:
                child_solution1[index] = solution2[index]
                child_solution2[index] = solution1[index]

        return child_solution1, child_solution2

    def mutation(self, solution):
        solution = solution.copy()
        for customer_index in range(len(solution)):
            mutation_random = random.random()
            if mutation_random <= self.mutation_rate:
                new_warehouse = random.randint(0, self.instance.warehouse_count - 1)
                if new_warehouse != solution[customer_index]:
                    solution[customer_index] = new_warehouse
                else:
                    if new_warehouse < self.instance.warehouse_count - 1:
                        new_warehouse += 1
                        solution[customer_index] = new_warehouse
                    else:
                        new_warehouse -= 1
                        solution[customer_index] = new_warehouse

        return solution

    def calculate_opening_cost(self, solution):
        open_facilities = set(solution)
        return sum(self.instance.warehouse_data[f][1] for f in open_facilities)

    def calculate_customer_cost(self, solution):
        total_customer_cost = 0.0
        for customer in range(len(solution)):
            customer_cost_list = self.instance.customer_data[customer][1]
            total_customer_cost += customer_cost_list[solution[customer]]

        return total_customer_cost

    def evaluate_solution(self, solution):
        opening_cost = round(self.calculate_opening_cost(solution), 5)
        customer_cost = round(self.calculate_customer_cost(solution), 5)

        return opening_cost, customer_cost

    # returns population list with [pop, warehouse opening cost, customer cost]
    def create_population(self):
        population = [self.create_solution() for _ in range(self.pop_size)]

        population_cost_list = []
        for solution in population:
            opening_cost, customer_cost = self.evaluate_solution(solution)
            population_cost_list.append([solution, opening_cost, customer_cost])

        return population_cost_list
