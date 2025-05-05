#!/usr/bin/env python3
# coding=utf-8

# Copyright (C) 2024-25  Paweł Widera
# ICOS, School of Computing Science, Newcastle University
#
# This program can only be used as part of CSC2034 coursework.
# You can't redistribute it and/or share it with others.

"""
Runs the evolution for a given target image while logging the
population statistics.

Usage: run.py [options] <target-file>
       run.py -h | --help

Options:
  -s NUM, --seed=NUM    seed for a random number generator [default: 31]
  -j NUM, --jobs=NUM    number of parallel jobs [default: 1]

Logging options:
  -t NUM, --step=NUM    log every NUM generation [default: 10]
  -l NAME, --log=NAME   name of the log file

Evolution parameters:
  --pop-size=NUM        number of solutions in population [default: 100]
  --generations=NUM     number of evolution iterations [default: 500]

"""
import sys
import random
import atexit
import multiprocessing

import evol
import PIL.Image
import PIL.ImageChops

from pathlib import Path
from docopt import docopt

from evoart import evolve, initialise, draw


SHAPES = 100
MAX = 255 * 200 * 200
TARGET = None


# For parallel evaluation on win/macos, target image has to be read
# before declaration of the evaluate function. This is due to no option
# to fork the process. Instead, a new process is created and code is
# imported. This results in later assignments to TARGET being ignored.
# We could pass the image around, instead of using a global but that
# significantly slows down the evaluation process. Therefore, unless
# you all switch to GNU/Linux, we have to suffer this ugliness here.
args = docopt(__doc__)
path = Path(args["<target-file>"])
# check if file exists
if not path.exists():
    print("Cannot find", path, file=sys.stderr)
    exit(1)
# read the target image
TARGET = PIL.Image.open(path)
TARGET.load()  # closes the file, needed for parallel eval


def evaluate(solution):
    image = draw(solution)
    diff = PIL.ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX + 0.03 * (SHAPES - len(solution)) / SHAPES


class Population(evol.population.BasePopulation):
    def __init__(self, initialise, size, parallel_map=None):
        chromosomes = [initialise() for i in range(size)]
        super().__init__(chromosomes, None, maximize=True)
        self.apply = parallel_map or (lambda f, x: list(map(f, x)))
        self.evals = 0

    def evaluate(self, lazy=True):
        offspring = [x for x in self.individuals if x.fitness is None]

        if offspring:
            scores = self.apply(evaluate, (x.chromosome for x in offspring))
            for ind, score in zip(offspring, scores):
                ind.fitness = score
            self.evals += len(scores)

        return self


class Logger(evol.logger.BaseLogger):
    def __init__(self, target=None, stdout=False, step=1):
        if target:
            target = Path(target).absolute()
            target.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(target, stdout)

        self.step = step
        self.count = 0

    def log(self, population, *, generation):
        self.count += 1
        if self.count >= self.step:
            self.count = 0
            values = [i.fitness for i in population]
            stats = [generation, population.evals, round(min(values), 3), round(max(values), 3)]
            self.logger.info(",".join(map(str, stats)))


if __name__ == "__main__":
    # setup logging
    log_step = int(args["--step"])
    if (log_file := args["--log"]):
        logger = Logger(step=log_step, target=log_file)
    else:
        logger = Logger(step=log_step, stdout=True)

    # fix the RNG seed for reproducibility
    random.seed(int(args["--seed"]))

    # setup parallel evaluation
    jobs = int(args["--jobs"])
    parallel_map = None
    if jobs > 1:
        pool = multiprocessing.Pool(jobs)
        parallel_map = pool.map
        atexit.register(pool.close)

    # create the first population
    population = Population(initialise, int(args["--pop-size"]), parallel_map)

    # run the evolution
    for i in range(1, int(args["--generations"])):
        evolve(population, args).callback(logger.log, generation=i)

    # log the last iteration (ignoring the step)
    logger.step = 1
    evolve(population, args).callback(logger.log, generation=i + 1)

    # save best solution as image
    name = "best_{}_p{--pop-size}_g{--generations}".format(path.stem, **args)
    draw(population.current_best.chromosome).save(path.with_stem(name))