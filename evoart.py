import random
import numpy as np
from PIL import Image, ImageDraw

# Constants
MAX_SHAPES = 100
MIN_SHAPES = 2  # Start with 2 shapes
IMAGE_SIZE = 200
TRIANGLE_SIDES = 3

# Access global TARGET from run.py
try:
    from run import TARGET
except ImportError:
    TARGET = None

def initialise():
    # Start with exactly 2 triangles
    num_shapes = 2
    shapes = []
    if TARGET is not None:
        # Convert target to grayscale and compute intensity
        gray_target = TARGET.convert("L")
        pixels = np.array(gray_target)
        # Find high-contrast areas (high variance in 5x5 window)
        high_contrast_points = []
        for y in range(10, IMAGE_SIZE - 10):
            for x in range(10, IMAGE_SIZE - 10):
                window = pixels[max(0, y-2):min(IMAGE_SIZE, y+3), max(0, x-2):min(IMAGE_SIZE, x+3)]
                if window.size > 0 and np.std(window) > 30:  # High variance indicates contrast
                    high_contrast_points.append((x, y))
        # Sample 2 points for triangle placement
        if high_contrast_points:
            centers = random.sample(high_contrast_points, min(2, len(high_contrast_points)))
            for x, y in centers:
                shapes.append(make_shape(x, y))
        else:
            shapes = [make_shape() for _ in range(num_shapes)]
    else:
        shapes = [make_shape() for _ in range(num_shapes)]
    return shapes

def make_shape(center_x=None, center_y=None):
    # Triangle: vertices, color
    if center_x is not None and center_y is not None and TARGET is not None:
        # Place vertices around the center to cover high-contrast area
        vertices = []
        # Estimate size based on local variance
        gray_target = TARGET.convert("L")
        pixels = np.array(gray_target)
        window = pixels[max(0, center_y-5):min(IMAGE_SIZE, center_y+6),
                        max(0, center_x-5):min(IMAGE_SIZE, center_x+6)]
        variance = np.std(window) if window.size > 0 else 50
        size = 20 if variance > 50 else 40  # Smaller triangles in high-variance areas
        for _ in range(TRIANGLE_SIDES):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(size * 0.5, size)
            vx = center_x + int(dist * np.cos(angle))
            vy = center_y + int(dist * np.sin(angle))
            vertices.extend([max(10, min(vx, IMAGE_SIZE - 10)), max(10, min(vy, IMAGE_SIZE - 10))])
        # Compute average color in bounding box
        x_coords, y_coords = vertices[0::2], vertices[1::2]
        x_min, x_max = max(0, min(x_coords)), min(IMAGE_SIZE - 1, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(IMAGE_SIZE - 1, max(y_coords))
        region = TARGET.crop((x_min, y_min, x_max + 1, y_max + 1))
        pixels = np.array(region)
        if pixels.size > 0:
            avg_color = np.mean(pixels, axis=(0, 1)).astype(int)
            r, g, b = avg_color[:3] if len(avg_color) >= 3 else (random.randint(0, 255),) * 3
        else:
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        a = random.randint(20, 80)
        colour = (r, g, b, a)
    else:
        vertices = [random.randint(10, IMAGE_SIZE - 10) for _ in range(TRIANGLE_SIDES * 2)]
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(20, 80))
    return ['triangle', vertices, colour]

def draw(solution):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    canvas = ImageDraw.Draw(image, "RGBA")
    for shape in solution:
        shape_type, data, colour = shape
        canvas.polygon(data, fill=colour)
    return image

def mutate(solution, rate, generation, max_generations):
    solution = list(solution)
    # Two-stage mutation: coarse early, fine later
    is_early = generation < max_generations // 2
    vertex_sigma = 12 if is_early else 3  # Finer adjustments for precise placement
    color_sigma = 25 if is_early else 6   # Finer color adjustments
    mutation_type = random.random()

    if mutation_type < 0.5:  # Mutate shape data (vertices)
        if solution:
            i = random.randrange(len(solution))
            shape = list(solution[i])
            coords = [x + random.normalvariate(0, vertex_sigma) if random.random() < rate else x for x in shape[1]]
            shape[1] = [max(0, min(int(x), IMAGE_SIZE - 1)) for x in coords]
            solution[i] = shape

    elif mutation_type < 0.85:  # Mutate color
        if solution:
            i = random.randrange(len(solution))
            shape = list(solution[i])
            r, g, b, a = shape[2]
            r = max(0, min(255, int(r + random.normalvariate(0, color_sigma))))
            g = max(0, min(255, int(g + random.normalvariate(0, color_sigma))))
            b = max(0, min(255, int(b + random.normalvariate(0, color_sigma))))
            a = max(20, min(80, int(a + random.normalvariate(0, 10 if is_early else 4))))
            shape[2] = (r, g, b, a)
            solution[i] = shape

    elif mutation_type < 0.95:  # Add or remove a shape
        if len(solution) < MAX_SHAPES and random.random() < 0.7:  # Higher chance to add
            solution.append(make_shape())
        elif len(solution) > MIN_SHAPES:
            solution.pop(random.randrange(len(solution)))

    else:  # Reorder shapes
        if solution:
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
    mutation_rate = 0.3 if current_gen < gen // 2 else 0.1
    population.survive(fraction=0.5)  # Keep top 50%
    population.breed(parent_picker=lambda pop: [select(pop) for _ in range(2)], combiner=combine)
    population.mutate(mutate_function=lambda sol: mutate(sol, mutation_rate, current_gen, gen))
    return population