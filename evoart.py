import random
import numpy as np
from PIL import Image, ImageDraw

# Constants
MAX_POLYGONS = 100
MIN_POLYGONS = 10
SIDES = 3
IMAGE_SIZE = 200


def initialise():
    # Start with a random number of polygons between MIN_POLYGONS and 20
    num_polygons = random.randint(MIN_POLYGONS, 20)
    return [make_polygon(SIDES) for _ in range(num_polygons)]


def make_polygon(n):
    # Generate vertices within image bounds
    vertices = [random.randint(10, IMAGE_SIZE - 10) for _ in range(n * 2)]
    # Semi-transparent color
    colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))
    return [vertices, colour]


def draw(solution):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[0], fill=polygon[1])
    return image


def mutate(solution, rate):
    solution = list(solution)
    mutation_type = random.random()

    if mutation_type < 0.4:  # Mutate vertex positions
        if solution:
            i = random.randrange(len(solution))
            polygon = list(solution[i])
            coords = [x + random.normalvariate(0, 10) if random.random() < rate else x for x in polygon[0]]
            polygon[0] = [max(0, min(int(x), IMAGE_SIZE)) for x in coords]
            solution[i] = polygon

    elif mutation_type < 0.7:  # Mutate color
        if solution:
            i = random.randrange(len(solution))
            polygon = list(solution[i])
            r, g, b, a = polygon[1]
            r = max(0, min(255, int(r + random.normalvariate(0, 20))))
            g = max(0, min(255, int(g + random.normalvariate(0, 20))))
            b = max(0, min(255, int(b + random.normalvariate(0, 20))))
            a = max(30, min(60, int(a + random.normalvariate(0, 5))))
            polygon[1] = (r, g, b, a)
            solution[i] = polygon

    elif mutation_type < 0.9:  # Add or remove a polygon
        if len(solution) < MAX_POLYGONS and random.random() < 0.5:
            solution.append(make_polygon(SIDES))
        elif len(solution) > MIN_POLYGONS:
            solution.pop(random.randrange(len(solution)))

    else:  # Reorder polygons
        random.shuffle(solution)

    return solution


def select(population, tournament_size=3):
    # Tournament selection
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)


def combine(*parents):
    # Create a child by blending polygons from parents
    child = []
    max_length = min(MAX_POLYGONS, max(len(p) for p in parents))

    for i in range(max_length):
        # Collect available polygons at index i from parents
        available = [p[i] for p in parents if i < len(p)]
        if available:
            # Randomly select a polygon from available ones
            child.append(random.choice(available))
        else:
            # Fallback: select a random polygon from a random parent
            parent = random.choice(parents)
            if parent:  # Ensure parent is not empty
                child.append(random.choice(parent))

    return child[:MAX_POLYGONS]


def evolve(population, args):
    # Adaptive mutation rate based on generation progress
    gen = int(args.get('--generations', 500))
    current_gen = population.evals // int(args.get('--pop-size', 100))
    mutation_rate = 0.2 if current_gen < gen // 2 else 0.05  # Higher early, lower later

    population.survive(fraction=0.5)  # Keep top 50%
    population.breed(parent_picker=lambda pop: [select(pop) for _ in range(2)], combiner=combine)
    population.mutate(mutate_function=lambda sol: mutate(sol, mutation_rate))
    return population