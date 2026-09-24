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

def main():
    path = './data/cap61.txt'
    num_warehouse, num_customers, warehouse_data, customer_data = load_data(path)
    solution = create_solution(num_warehouse,num_customers)
    solution_correct = feasibility_repair(solution,num_warehouse, warehouse_data, customer_data)
    x = feasibility_repair(solution_correct,num_warehouse, warehouse_data, customer_data)
    print()


if __name__ == "__main__":
    main()