import random
import numpy as np
from PIL import Image, ImageDraw

# Constants
MAX_SHAPES = 100
MIN_SHAPES = 10
IMAGE_SIZE = 200
TRIANGLE_SIDES = 3
CIRCLE_PROB = 0.2  # Probability of generating a circle instead of a triangle


def initialise():
    # Start with a random number of shapes between MIN_SHAPES and 20
    num_shapes = random.randint(MIN_SHAPES, 20)
    return [make_shape() for _ in range(num_shapes)]


def make_shape():
    # Decide whether to create a triangle or circle
    if random.random() < CIRCLE_PROB:
        # Circle: center (x, y), radius, color
        x, y = random.randint(10, IMAGE_SIZE - 10), random.randint(10, IMAGE_SIZE - 10)
        radius = random.randint(5, 20)
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(10, 100))
        return ['circle', [x, y, radius], colour]
    else:
        # Triangle: vertices, color
        vertices = [random.randint(10, IMAGE_SIZE - 10) for _ in range(TRIANGLE_SIDES * 2)]
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(10, 100))
        return ['triangle', vertices, colour]


def draw(solution):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    canvas = ImageDraw.Draw(image, "RGBA")
    for shape in solution:
        shape_type, data, colour = shape
        if shape_type == 'triangle':
            canvas.polygon(data, fill=colour)
        else:  # circle
            x, y, radius = data
            canvas.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)
    return image


def mutate(solution, rate, generation, max_generations):
    solution = list(solution)
    # Two-stage mutation: coarse early, fine later
    is_early = generation < max_generations // 2
    vertex_sigma = 15 if is_early else 5  # Larger changes early, smaller later
    color_sigma = 30 if is_early else 10  # Larger color changes early, smaller later
    mutation_type = random.random()

    if mutation_type < 0.5:  # Mutate shape data (vertices or circle params)
        if solution:
            i = random.randrange(len(solution))
            shape = list(solution[i])
            if shape[0] == 'triangle':
                coords = [x + random.normalvariate(0, vertex_sigma) if random.random() < rate else x for x in shape[1]]
                shape[1] = [max(0, min(int(x), IMAGE_SIZE)) for x in coords]
            else:  # circle
                x, y, radius = shape[1]
                x += random.normalvariate(0, vertex_sigma)
                y += random.normalvariate(0, vertex_sigma)
                radius += random.normalvariate(0, 3 if is_early else 1)
                shape[1] = [max(0, min(int(x), IMAGE_SIZE)), max(0, min(int(y), IMAGE_SIZE)),
                            max(5, min(20, int(radius)))]
            solution[i] = shape

    elif mutation_type < 0.9:  # Mutate color (increased probability)
        if solution:
            i = random.randrange(len(solution))
            shape = list(solution[i])
            r, g, b, a = shape[2]
            r = max(0, min(255, int(r + random.normalvariate(0, color_sigma))))
            g = max(0, min(255, int(g + random.normalvariate(0, color_sigma))))
            b = max(0, min(255, int(b + random.normalvariate(0, color_sigma))))
            a = max(10, min(100, int(a + random.normalvariate(0, 10 if is_early else 5))))
            shape[2] = (r, g, b, a)
            solution[i] = shape

    elif mutation_type < 0.95:  # Add or remove a shape
        if len(solution) < MAX_SHAPES and random.random() < 0.5:
            solution.append(make_shape())
        elif len(solution) > MIN_SHAPES:
            solution.pop(random.randrange(len(solution)))

    else:  # Reorder shapes
        random.shuffle(solution)

    return solution


def select(population, tournament_size=3):
    # Tournament selection
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)


def combine(*parents):
    # Create a child by blending shapes from parents
    child = []
    max_length = min(MAX_SHAPES, max(len(p) for p in parents))

    for i in range(max_length):
        available = [p[i] for p in parents if i < len(p)]
        if available:
            child.append(random.choice(available))
        else:
            parent = random.choice(parents)
            if parent:
                child.append(random.choice(parent))

    return child[:MAX_SHAPES]


def evolve(population, args):
    # Adaptive mutation rate and stage-based mutation
    gen = int(args.get('--generations', 500))
    current_gen = population.evals // int(args.get('--pop-size', 100))
    mutation_rate = 0.3 if current_gen < gen // 2 else 0.1  # Higher early, lower later

    population.survive(fraction=0.5)  # Keep top 50%
    population.breed(parent_picker=lambda pop: [select(pop) for _ in range(2)], combiner=combine)
    population.mutate(mutate_function=lambda sol: mutate(sol, mutation_rate, current_gen, gen))
    return population