from base import BaseAlgorithm


class NSGA2(BaseAlgorithm):
    def __init__(self, filepath, mutation_rate, crossover_rate, pop_size):
        super().__init__(filepath, mutation_rate, crossover_rate, pop_size)


    def calculate_crowding_distance(self, front, population):
        distances = {index: 0.0 for index in front}
        if len(front) <= 2:
            for index in front:
                distances[index] = float('inf')
            return distances

        for obj_index in [1, 2]:
            sorted_front = sorted(front, key=lambda idx: population[idx][obj_index])

            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

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
        s = [[] for _ in range(len(population))]
        n = [0] * len(population)
        fronts = [[]]

        for i in range(0, len(population)):
            for j in range(i+1, len(population)):
                if (population[i][1] <= population[j][1] and population[i][2] <= population[j][2]):
                    if population[i][1] < population[j][1] or population[i][2] < population[j][2]:
                        s[i].append(j)
                        n[j] += 1
                elif population[j][1] <= population[i][1] and population[j][2] <= population[i][2]:
                    if population[j][1] < population[i][1] or population[j][2] < population[i][2]:
                        s[j].append(i)
                        n[i] += 1
        
        fronts[0] = [index for index, value in enumerate(n) if value == 0]

        current_front_index = 0
        while fronts[current_front_index]:
            next_front = []
            for p in fronts[current_front_index]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            current_front_index += 1
            if not next_front:
                break
            fronts.append(next_front)

        return fronts


    def determine_new_population(self, parent_pop, children_pop):
        new_population = []

        population = parent_pop + children_pop

        fronts = self.non_dominated_sort(population)

        for front in fronts:
            if len(front) + len(new_population) > self.pop_size:
                distances = self.calculate_crowding_distance(front, population)
                sorted_front = sorted(front, key=lambda idx: distances[idx], reverse=True)
                remaining = self.pop_size - len(new_population)
                for index in sorted_front[:remaining]:
                    new_population.append(population[index])
                break
            else:
                for solution_index in front:
                    new_population.append(population[solution_index])
                    if len(new_population) == self.pop_size:
                        break

        return new_population


def main():
    path = "./data/cap61.txt"
    instance = NSGA2(path,0.05,0.7,200)
    parents = instance.create_population()
    children = instance.create_children(parents)
    new_population = instance.determine_new_population(parents,children)


if __name__ == "__main__":
    main()