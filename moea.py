import random



def load_data(filepath):
    '''
    Loads and parses a JSSP benchmark file into (num_jobs, num_machines, jobs).
    '''
    with open(filepath, "r") as file:
        words = file.read().split()
        words_iter = iter(words)
        num_warehouse = 0
        num_customers = 0
        warehouse_data = []
        customer_data = []
        num_warehouse = int(next(words_iter))
        num_customers = int(next(words_iter))

        for _ in range(0,num_warehouse):
            warehouse_data.append((int(next(words_iter)),float(next(words_iter))))

        for _ in range(0,num_customers):
            customer_demand = int(next(words_iter))
            customer_cost = []
            for index in range(0,num_warehouse):
                customer_cost.append(float(next(words_iter)))
            customer_data.append((customer_demand,customer_cost))

    return [num_warehouse, num_customers, warehouse_data, customer_data]

def create_solution(num_warehouse,num_customers):
    # index=customer; entry=warehouse
    solution = []

    for _ in range(num_customers):
        solution.append(random.randint(0, num_warehouse-1))

    return solution 

def feasibility_repair(solution, num_warehouse, warehouse_data, customer_data):
    solution = solution.copy()
    unassigned_customers = []
    facility_usage = [0] * num_warehouse

    for customer in range(len(solution)):
        warehouse = solution[customer]
        demand = customer_data[customer][0]
        if facility_usage[warehouse] + demand <= warehouse_data[warehouse][0]:
            facility_usage[warehouse] += demand
        else:
            unassigned_customers.append(customer)

    for customer in unassigned_customers:
        demand = customer_data[customer][0]
        costs = customer_data[customer][1]

        open_facilities_with_space = [
            f for f in range(num_warehouse)
            if facility_usage[f] > 0 and facility_usage[f] + demand <= warehouse_data[f][0]
        ]

        if open_facilities_with_space:
            best_facility = min(open_facilities_with_space, key=lambda f: costs[f])
        else:
            closed_facilities_with_space = [
                f for f in range(num_warehouse)
                if facility_usage[f] == 0 and demand <= warehouse_data[f][0]
            ]
            if closed_facilities_with_space:
                best_facility = min(closed_facilities_with_space, key=lambda f: warehouse_data[f][1] + costs[f])
            else:
                best_facility = max(range(num_warehouse), key=lambda f: warehouse_data[f][0] - facility_usage[f])

        solution[customer] = best_facility
        facility_usage[best_facility] += demand

    return solution

def crossover(solution1, solution2, num_warehouse, warehouse_data, customer_data):
    child_solution1 = [None] * len(solution1)
    child_solution2 = [None] * len(solution1)

    for index in range(0,len(solution1)):
        value_choice = random.choice([0, 1])
        if value_choice == 1:
            child_solution1[index] = solution1[index]
            child_solution2[index] = solution2[index]
        else:
            child_solution1[index] = solution2[index]
            child_solution2[index] = solution1[index]

    # i think we can remove this if we alsway run mutation wich we should with how i implemented mutation
    # child_solution1 = feasibility_repair(child_solution1, num_warehouse, warehouse_data, customer_data)
    # child_solution2 = feasibility_repair(child_solution2, num_warehouse, warehouse_data, customer_data)


    return child_solution1,child_solution2


def mutation(solution, num_warehouse, warehouse_data, customer_data, mutation_rate):
    solution = solution.copy()
    for customer_index in range(0,len(solution)):
        mutation_random = random.random()
        if mutation_random <= mutation_rate:
            new_warehouse = random.randint(0,num_warehouse-1)
            if new_warehouse != solution[customer_index]:
                solution[customer_index] = new_warehouse
            else:
                if new_warehouse < num_warehouse-1:
                    new_warehouse += 1
                    solution[customer_index] = new_warehouse
                else:
                    new_warehouse -= 1
                    solution[customer_index] = new_warehouse

    solution = feasibility_repair(solution, num_warehouse, warehouse_data, customer_data)
    return solution

def calculate_opening_cost(solution, warehouse_data):
    open_facilities = set(solution)
    return sum(warehouse_data[f][1] for f in open_facilities)

def calculate_customer_cost(solution, customer_data):
    total_customer_cost = 0.0
    for customer in range(0,len(solution)):
        customer_cost_list = customer_data[customer][1]
        total_customer_cost += customer_cost_list[solution[customer]]

    return total_customer_cost


def evaluate_solution(solution, warehouse_data, customer_data):
    opening_cost = round(calculate_opening_cost(solution, warehouse_data), 5)
    customer_cost = round(calculate_customer_cost(solution, customer_data), 5)

    return opening_cost, customer_cost


# returns population list with [pop, warehouse opening cost, customer cost]
def create_population(pop_size, num_warehouse, num_customers, warehouse_data, customer_data):
    population = [create_solution(num_warehouse, num_customers) for _ in range(0, pop_size)]

    population_cost_list = []
    for solution in population:
        solution = feasibility_repair(solution,num_warehouse,warehouse_data,customer_data)
        opening_cost, customer_cost = evaluate_solution(solution, warehouse_data, customer_data)
        population_cost_list.append([solution,opening_cost,customer_cost])

    return population_cost_list

    

def main():
    path = './data/cap61.txt'
    mutation_rate = 0.05
    pop_size = 200
    num_warehouse, num_customers, warehouse_data, customer_data = load_data(path)
    population = create_population(pop_size, num_warehouse, num_customers, warehouse_data, customer_data)
    print()


if __name__ == "__main__":
    main()