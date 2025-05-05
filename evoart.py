

import random

from PIL import Image, ImageDraw

POLYGON_COUNT = 100
SIDES = 3

def initialise():
    return [make_polygon(SIDES) for i in range(POLYGON_COUNT)]

def make_polygon(n):
    vertices = [random.randint(10, 190) if i % 2 == 0 else random.randint(10, 190) for i in range(n * 2)]
    colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))
    return [vertices, colour]

def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[0], fill=polygon[1])
    return image

def mutate(solution, rate):
    solution = list(solution)

    if random.random() < 0.5:
        # mutate points
        i = random.randrange(len(solution))
        polygon = list(solution[i])
        coords = [x if random.random() > rate else
            x + random.normalvariate(0, 10) for x in polygon[0]]
        polygon[0] = [max(0, min(int(x), 200)) for x in coords]
        solution[i] = polygon
    else:
        # reorder polygons
        random.shuffle(solution)

    return solution

def select(population):
    return [random.choice(population) for i in range(2)]

def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]

def evolve(population, args):
    population.survive(fraction=0.5)
    population.breed(parent_picker=select, combiner=combine)
    population.mutate(mutate_function=mutate, rate=0.1)
    return population
