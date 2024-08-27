import random
from deap import base, creator, tools, algorithms


creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))


creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()


toolbox.register("attr_float", random.uniform, 0, 100)  # Range for both objectives


toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # 2 objectives


toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    distance = sum(individual)  # Sum of variables as distance
    value = sum((x - 50)**2 for x in individual)  # Sum of squared deviations from 50 as value
    return distance, value


toolbox.register("evaluate", evaluate)


toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)


toolbox.register("select", tools.selNSGA2)


population = toolbox.population(n=50)

# Run the multi-objective optimization using NSGA-II algorithm
algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100,
                          cxpb=0.7, mutpb=0.2, ngen=50, verbose=False)

# Print the Pareto front solutions
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
for ind in pareto_front:
    print("Distance:", ind.fitness.values[0], "| Value:", ind.fitness.values[1])
