import random

from base import BaseAlgorithm


class NSGA2(BaseAlgorithm):
    def calculate_crowding_distance(self, front, population):
        distances = {index: 0.0 for index in front}
        if len(front) <= 2:
            for index in front:
                distances[index] = float("inf")
            return distances

        for obj_index in [1, 2]:
            sorted_front = sorted(front, key=lambda idx: population[idx][obj_index])

            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            min_val = population[sorted_front[0]][obj_index]
            max_val = population[sorted_front[-1]][obj_index]
            val_range = max_val - min_val

            if val_range == 0:
                continue
            for k in range(1, len(sorted_front) - 1):
                prev_val = population[sorted_front[k - 1]][obj_index]
                next_val = population[sorted_front[k + 1]][obj_index]
                distances[sorted_front[k]] += (next_val - prev_val) / val_range
        return distances

    def non_dominated_sort(self, population):
        p = population
        s = [[] for _ in range(len(p))]
        n = [0] * len(p)
        fronts = [[]]

        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if (p[i][1] <= p[j][1] and p[i][2] <= p[j][2]) and (
                    p[i][1] < p[j][1] or p[i][2] < p[j][2]
                ):
                    s[i].append(j)
                    n[j] += 1
                elif (p[j][1] <= p[i][1] and p[j][2] <= p[i][2]) and (
                    p[j][1] < p[i][1] or p[j][2] < p[i][2]
                ):
                    s[j].append(i)
                    n[i] += 1

        fronts[0] = [index for index, value in enumerate(n) if value == 0]

        current_front_index = 0
        while fronts[current_front_index]:
            next_front = []
            for front in fronts[current_front_index]:
                for q in s[front]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            current_front_index += 1
            if not next_front:
                break
            fronts.append(next_front)

        return fronts

    def assign_rank_and_distance(self, population):
        ranked_population = []
        fronts = self.non_dominated_sort(population)
        for front_index, front in enumerate(fronts):
            distances = self.calculate_crowding_distance(front, population)
            for index in front:
                ranked_population.append(
                    population[index][:3] + [front_index, distances.get(index)]
                )
        return ranked_population

    def create_population(self):
        population = super().create_population()
        return self.assign_rank_and_distance(population)

    def create_new_population(self, parent_population):
        new_population = []
        child_population = self.create_children(parent_population)
        population = parent_population + child_population
        fronts = self.non_dominated_sort(population)
        front_index = 0

        for front in fronts:
            if len(front) + len(new_population) > self.pop_size:
                distances = self.calculate_crowding_distance(front, population)
                sorted_front = sorted(
                    front, key=lambda idx: distances[idx], reverse=True
                )
                remaining = self.pop_size - len(new_population)
                for index in sorted_front[:remaining]:
                    new_population.append(population[index][:3] + [front_index, distances.get(index)])
                break
            else:
                distances = self.calculate_crowding_distance(front, population)
                for index in front:
                    new_population.append(population[index][:3] + [front_index, distances.get(index)])
                    if len(new_population) == self.pop_size:
                        break
            front_index += 1

        return new_population

    def tournament_selection(self, population):
        candidates = random.sample(population, self.tournament_size)
        return min(candidates, key=lambda ind: (ind[3], -ind[4]))

    def create_children(self, parent_population):
        children_cost_list = []
        children = []
        while len(children) < self.pop_size:
            parent1 = self.tournament_selection(parent_population)
            parent2 = self.tournament_selection(parent_population)

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

    def run(self):
        population = self.create_population()
        for _ in range(0,self.generations):
            population = self.create_new_population(population)
        return population


def main():
    path = "./data/cap61.txt"
    instance = NSGA2(path, 100, 0.05, 0.7, 200)
    population = instance.run()
    index = 0
    for solution in population:
        if solution[3] == 0:
            print(index)
            print(f"Warehpouse opening cost: {solution[1]}")
            print(f"Customer cost: {solution[2]}")
            print("______________________________________")    
        index += 1    
    pass


if __name__ == "__main__":
    main()
