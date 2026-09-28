from base import BaseAlgorithm
import random


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

    def determine_new_population(self, parent_population, child_population):
        new_population = []
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
                    population[index].append(front_index)
                    population[index].append(distances.get(index))
                    new_population.append(population[index])
                break
            else:
                distances = self.calculate_crowding_distance(front, population)
                for solution_index in front:
                    population[solution_index].append(front_index)
                    population[solution_index].append(distances.get(solution_index))
                    new_population.append(population[solution_index])
                    if len(new_population) == self.pop_size:
                        break
            front_index += 1

        return new_population

    def tournament_selection(self, population):
        candidate_list = random.sample(population, 2)
        if candidate_list[0][3] < candidate_list[1][3]:
            return candidate_list[0]
        elif candidate_list[0][3] > candidate_list[1][3]:
            return candidate_list[1]
        elif candidate_list[0][3] == candidate_list[1][3]:
            if candidate_list[0][4] < candidate_list[1][4]:
                return candidate_list[0]
            elif candidate_list[0][4] > candidate_list[1][4]:
                return candidate_list[1]
            else:
                return candidate_list[random.randint(0, 1)]


def main():
    path = "./data/cap61.txt"
    instance = NSGA2(path, 5000, 0.05, 0.7, 200)
    parents = instance.create_population()
    pass


if __name__ == "__main__":
    main()
