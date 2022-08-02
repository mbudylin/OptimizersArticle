import os
import sys
from itertools import product
import pandas as pd

sys.path.append('./OptimizersArticle')

from data_generator.data_generator import generate_data

from optimizers.optimizers import (
    ScipyNlpOptimizationModel,
    PyomoNlpOptimizationModel,
    PyomoLpOptimizationModel,
    CvxpyLpOptimizationModel
)
from optimizers.optimization import pricing_optimization

STAT_PATH = './data/stat'

SEED_GRID = list(range(25))
N_GRID = [10, 20, 50, 100, 200, 500, 1000]
# SEED_GRID = list(range(5))
# N_GRID = [10, 20]
GRID = product(N_GRID, SEED_GRID)
BOUNDS_PARAMS = {
    'main_bounds': {
        'lower': 0.9, 'upper': 1.1
    },
    'market_bounds': {
        'lower': 0.85, 'upper': 1.15
    }
}
LP_MAX_GRID_SIZE = 21


def get_files(file_path):
    files = []
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith('.csv')]
    return files


def optimizers_calc_stat(grid, file_path_stat, overwrite=False):

    existed_files = set(get_files(file_path_stat))

    for N, seed in grid:
        file_name = 's_' + str(N) + '_' + str(seed) + '.csv'

        # print(dump_key)
        # print(existed_stat)
        # assert ()
        if not overwrite:
            if file_name in existed_files:
                continue
        print('--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--')
        print(N, seed)
        # генерация данных для NLP и LP оптимизации
        data = generate_data(N, BOUNDS_PARAMS, LP_MAX_GRID_SIZE, seed)

        M_cur = sum(data['data_nlp']['Q'] * (data['data_nlp']['P'] - data['data_nlp']['C']))
        R_cur = sum(data['data_nlp']['Q'] * data['data_nlp']['P'])

        opt_params = {
            'alpha': 0.0,
            'con_mrg': M_cur,
        }

        statuses = []
        times = []
        solvers = []
        opt_types = []

        if N < 1000:
            res, t = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'slsqp')
            statuses.append('ok' if res['status'] == '0' else res['status'])
            times.append(t)
            solvers.append('scipy.slsqp')
            opt_types.append('NLP')
            print(f'slsqp finish \t{t}')

        if N < 500:
            res, t = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'cobyla')
            statuses.append('ok' if res['status'] == '1' else res['status'])
            times.append(t)
            solvers.append('scipy.cobyla')
            opt_types.append('NLP')
            print(f'cobyla finish \t{t}')

        if N < 500:
            res, t = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'trust-constr')
            statuses.append('ok' if res['status'] == '1' else res['status'])
            times.append(t)
            solvers.append('scipy.trust-constr')
            opt_types.append('NLP')
            print(f'trust-constr finish \t{t}')

        res, t = pricing_optimization(data, PyomoNlpOptimizationModel, opt_params, 'ipopt')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo.ipopt')
        opt_types.append('NLP')
        print(f'ipopt finish \t{t}')

        res, t = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'cbc')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo.cbc')
        opt_types.append('MILP')
        print(f'pyomo cbc finish \t{t}')

        res, t = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'glpk')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo.glpk')
        opt_types.append('MILP')
        print(f'pyomo glpk finish \t{t}')

        res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'CBC')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy.cbc')
        opt_types.append('MILP')
        print(f'cvxpy cbc finish \t{t}')

        res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'GLPK_MI')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy.glpk')
        opt_types.append('MILP')
        print(f'cvxpy glpk finish \t{t}')

        if N < 2000:
            res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'ECOS_BB')
            statuses.append('ok' if res['status'] == 'optimal' else res['status'])
            times.append(t)
            solvers.append('cvxpy.ecos')
            opt_types.append('MILP')
            print(f'ecos finish \t{t}')

        df = pd.DataFrame({
            'N': N,
            'seed': seed,
            'solver': solvers,
            'opt_type': opt_types,
            't': times,
            'status': statuses
        })
        print(os.path.join(file_path_stat, file_name))
        df.to_csv(os.path.join(file_path_stat, file_name), index=False)


def optimizers_collect_stat(file_path_stat):
    files_stat = get_files(file_path_stat)
    df = pd.concat([pd.read_csv(os.path.join(file_path_stat, file_name)) for file_name in files_stat])
    return df


if __name__ == '__main__':

    optimizers_calc_stat(GRID, STAT_PATH, overwrite=False)
    stat_df = optimizers_collect_stat(STAT_PATH)
