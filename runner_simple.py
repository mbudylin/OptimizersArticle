# -*- coding: utf-8 -*-
import os
import sys
from itertools import product
import pandas as pd
import argparse

sys.path.append('./OptimizersArticle')

from data_generator.data_generator import generate_simple_data

from optimizers.optimizers_simple import ScipyModel, PyomoModel
from optimizers.optimization import pricing_optimization


IS_DOCKER = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
if IS_DOCKER:
    STAT_PATH = './data/docker/stat'
else:
    STAT_PATH = './data/stat'


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-m", "--mode", required=False, default=False, type=str,
                             help='')
    args_parser.add_argument("-N", "--N", required=False, default=10, type=int,
                             help='Размер задачи')
    args_parser.add_argument("-s", "--seed", required=False, default=42, type=int,
                             help='seed')

    args = vars(args_parser.parse_args())

    data = generate_simple_data(args['N'], seed=args['seed'])

    if args['mode'] == 'pyomo':
        print('запуск модели Pyomo')
        # model = pricing_optimization(data, ScipyModel, {}, 'ipopt')

    if args['mode'] == 'scipy':
        print('запуск модели Scipy')
        # model = pricing_optimization(data, ScipyModel, {}, 'ipopt')
