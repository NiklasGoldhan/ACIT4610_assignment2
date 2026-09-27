from base import BaseAlgorithm

class NSGA2(BaseAlgorithm):
    def __init__(self, filepath, mutation_rate,crossover_rate, pop_size):

        super().__init__(filepath, mutation_rate,crossover_rate, pop_size)
        self.population = self.create_population()

    def calculate_domination(self, parent_pop, children_pop):
        new_population = []

        combined_pop = parent_pop + children_pop
        s = [[] for _ in range(len(combined_pop))]
        n = [0] * len(combined_pop)
        fronts = [[]]
        for i in range(0,len(combined_pop)):
            for j in range(i+1,len(combined_pop)):
                if combined_pop[i][1] <= combined_pop[j][1] and combined_pop[i][2] <= combined_pop[j][2]:
                    if combined_pop[i][1] < combined_pop[j][1] or combined_pop[i][2] < combined_pop[j][2]:
                        s[i].append(j)
                        n[j] += 1
                elif combined_pop[j][1] <= combined_pop[i][1] and combined_pop[j][2] <= combined_pop[i][2]:
                    if combined_pop[j][1] < combined_pop[i][1] or combined_pop[j][2] < combined_pop[i][2]:
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
            if next_front == []:
                break
            fronts.append(next_front)

        for front in fronts:
            if len(front) + len(new_population) > self.pop_size:
                distances = self.calculate_crowding_distance(front, combined_pop)
                sorted_front = sorted(front, key=lambda idx: distances[idx], reverse=True)
                remaining = self.pop_size - len(new_population)
                for index in sorted_front[:remaining]:
                    new_population.append(combined_pop[index])
                break
            else:
                for solution_index in front:
                    new_population.append(combined_pop[solution_index])
                    if len(new_population) == self.pop_size:
                        break

        return new_population

    def calculate_crowding_distance(self, front, combined_pop):
        distances = {index: 0.0 for index in front}
        if len(front) <= 2:
            for index in front:
                distances[index] = float('inf')
            return distances

        for obj in [1, 2]:
            sorted_front = sorted(front, key=lambda idx: combined_pop[idx][obj])

            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            min_val = combined_pop[sorted_front[0]][obj]
            max_val = combined_pop[sorted_front[-1]][obj]
            val_range = max_val - min_val

            if val_range == 0:
                continue
            for k in range(1, len(sorted_front) - 1):
                prev_val = combined_pop[sorted_front[k - 1]][obj]
                next_val = combined_pop[sorted_front[k + 1]][obj]
                distances[sorted_front[k]] += (next_val - prev_val) / val_range
        return distances
        

        

    

def main():
    path = "./data/cap61.txt"
    instance = NSGA2(path,0.05,0.7,200)
    parents = instance.create_population()
    children = instance.create_children(parents)
    new_population = instance.calculate_domination(parents,children)

    print()

if __name__ == "__main__":
    main()