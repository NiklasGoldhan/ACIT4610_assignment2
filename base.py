import random


class BaseAlgorithm:
    def __init__(self, filepath, mutation_rate, crossover_rate, pop_size):
        with open(filepath, "r") as file:
            words = iter(file.read().split())

            warehouse_count = int(next(words))
            customer_count = int(next(words))
            warehouse_data = []
            customer_data = []

            for _ in range(warehouse_count):
                warehouse_data.append((int(next(words)), float(next(words))))

            for _ in range(customer_count):
                customer_demand = int(next(words))
                customer_cost = []
                for index in range(warehouse_count):
                    customer_cost.append(float(next(words)))
                customer_data.append((customer_demand, customer_cost))
        self.warehouse_count = warehouse_count
        self.customer_count = customer_count
        self.warehouse_data = warehouse_data
        self.customer_data = customer_data
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = 2
        self.pop_size = pop_size

    def create_solution(self):
        # index=customer; entry=warehouse
        solution = []

        for _ in range(self.customer_count):
            solution.append(random.randint(0, self.warehouse_count - 1))

        return solution

    def feasibility_repair(self, solution):
        solution = solution.copy()
        unassigned_customers = []
        facility_usage = [0] * self.warehouse_count

        for customer in range(len(solution)):
            warehouse = solution[customer]
            demand = self.customer_data[customer][0]
            if facility_usage[warehouse] + demand <= self.warehouse_data[warehouse][0]:
                facility_usage[warehouse] += demand
            else:
                unassigned_customers.append(customer)

        for customer in unassigned_customers:
            demand = self.customer_data[customer][0]
            costs = self.customer_data[customer][1]

            open_facilities_with_space = [
                f
                for f in range(self.warehouse_count)
                if facility_usage[f] > 0
                and facility_usage[f] + demand <= self.warehouse_data[f][0]
            ]

            if open_facilities_with_space:
                best_facility = min(open_facilities_with_space, key=lambda f: costs[f])
            else:
                closed_facilities_with_space = [
                    f
                    for f in range(self.warehouse_count)
                    if facility_usage[f] == 0 and demand <= self.warehouse_data[f][0]
                ]
                if closed_facilities_with_space:
                    best_facility = min(
                        closed_facilities_with_space,
                        key=lambda f: self.warehouse_data[f][1] + costs[f],
                    )
                else:
                    best_facility = max(
                        range(self.warehouse_count),
                        key=lambda f: self.warehouse_data[f][0] - facility_usage[f],
                    )

            solution[customer] = best_facility
            facility_usage[best_facility] += demand

        return solution

    def crossover(self, solution1, solution2):
        child_solution1 = [None] * len(solution1)
        child_solution2 = [None] * len(solution1)

        for index in range(len(solution1)):
            value_choice = random.choice([0, 1])
            if value_choice == 1:
                child_solution1[index] = solution1[index]
                child_solution2[index] = solution2[index]
            else:
                child_solution1[index] = solution2[index]
                child_solution2[index] = solution1[index]

        # i think we can remove this if we always run mutation wich we should with how i implemented mutation
        # child_solution1 = feasibility_repair(child_solution1, num_warehouse, warehouse_data, customer_data)
        # child_solution2 = feasibility_repair(child_solution2, num_warehouse, warehouse_data, customer_data)

        return child_solution1, child_solution2

    def mutation(self, solution):
        solution = solution.copy()
        for customer_index in range(len(solution)):
            mutation_random = random.random()
            if mutation_random <= self.mutation_rate:
                new_warehouse = random.randint(0, self.warehouse_count - 1)
                if new_warehouse != solution[customer_index]:
                    solution[customer_index] = new_warehouse
                else:
                    if new_warehouse < self.warehouse_count - 1:
                        new_warehouse += 1
                        solution[customer_index] = new_warehouse
                    else:
                        new_warehouse -= 1
                        solution[customer_index] = new_warehouse

        solution = self.feasibility_repair(solution)
        return solution

    def calculate_opening_cost(self, solution):
        open_facilities = set(solution)
        return sum(self.warehouse_data[f][1] for f in open_facilities)

    def calculate_customer_cost(self, solution):
        total_customer_cost = 0.0
        for customer in range(len(solution)):
            customer_cost_list = self.customer_data[customer][1]
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
            solution = self.feasibility_repair(solution)
            opening_cost, customer_cost = self.evaluate_solution(solution)
            population_cost_list.append([solution, opening_cost, customer_cost])

        return population_cost_list

    def turnament_selection(self, population):
        candidate_list = random.sample(
            population,
            self.tournament_size
        )

        return min(candidate_list, key=lambda item: item[1])


    def create_children(self, parent_population):
        children_cost_list = []
        children = []
        while len(children) < self.pop_size:
            parent1 = self.turnament_selection(parent_population)
            parent2 = self.turnament_selection(parent_population)

            parent_solution1 = parent1[0]
            parent_solution2 = parent2[0]

            if random.random() <= self.crossover_rate:
                child1, child2 = self.crossover(parent_solution1, parent_solution2)
            else:
                child1 = parent_solution1.copy()
                child2 = parent_solution2.copy()

            if random.random() < self.mutation_rate:
                child1 = self.mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = self.mutation(child2)

            children.append(child1)
            if len(children) < self.pop_size:
                    children.append(child2)

        for child in children:
            opening_cost, customer_cost = self.evaluate_solution(child)
            children_cost_list.append([child, opening_cost, customer_cost])
            
        return children_cost_list
    

def main():
    path = "./data/cap61.txt"
    instance = BaseAlgorithm(path, 0.05, 0.7 , 200)
    population = instance.create_population()
    children = instance.create_children(population,)
    print()


if __name__ == "__main__":
    main()
