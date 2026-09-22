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


load_data('./data/cap41.txt')