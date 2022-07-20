import os
import sys
from itertools import product
import pandas as pd

sys.path.append('./OptimizersArticle')

from data_generator.data_generator import (
    generate_data,
    construct_bounds,
    construct_lp_grid
)

from optimizers.optimizers import (
    ScipyNlpOptimizationModel,
    PyomoNlpOptimizationModel,
    PyomoLpOptimizationModel,
    CvxpyLpOptimizationModel
)
from optimizers.optimization import pricing_optimization

DATA_DUMP = './data/dump.hdf'

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


def get_keys(filepath_stat_dump):
    existed_stat = []
    if os.path.exists(filepath_stat_dump):
        with pd.HDFStore(filepath_stat_dump) as hdf:
            existed_stat = list(hdf.keys())
    return [el.replace('/', '') for el in existed_stat]


def optimizers_calc_stat(grid, filepath_stat_dump, overwrite=False):

    existed_stat = get_keys(filepath_stat_dump)

    for N, seed in grid:
        dump_key = 's_' + str(N) + '_' + str(seed)

        # print(dump_key)
        # print(existed_stat)
        # assert ()
        if not overwrite:
            if dump_key in existed_stat:
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
            'con_equal': True
        }

        statuses = []
        times = []
        solvers = []

        if N < 1000:
            res, t = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'slsqp')
            statuses.append('ok' if res['status'] == '0' else res['status'])
            times.append(t)
            solvers.append('slsqp')
            print(f'slsqp finish \t{t}')

        if N < 500:
            res, t = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'trust-constr')
            statuses.append('ok' if res['status'] == '1' else res['status'])
            times.append(t)
            solvers.append('trust-constr')
            print(f'trust finish \t{t}')

        res, t = pricing_optimization(data, PyomoNlpOptimizationModel, opt_params, 'ipopt')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('ipopt')
        print(f'ipopt finish \t{t}')

        res, t = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'cbc')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo_cbc')
        print(f'pyomo cbc finish \t{t}')

        res, t = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'glpk')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo_glpk')
        print(f'pyomo glpk finish \t{t}')

        res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'CBC')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy_cbc')
        print(f'cvxpy cbc finish \t{t}')

        res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'GLPK_MI')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy_glpk')
        print(f'cvxpy glpk finish \t{t}')

        if N < 2000:
            res, t = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'ECOS_BB')
            statuses.append('ok' if res['status'] == 'optimal' else res['status'])
            times.append(t)
            solvers.append('ecos')
            print(f'ecos finish \t{t}')

        df = pd.DataFrame({
            'N': N,
            'seed': seed,
            'solver': solvers,
            't': times,
            'status': statuses
        })
        df.to_hdf(filepath_stat_dump, dump_key)


def optimizers_collect_stat(data_dump):
    existed_stat = get_keys(data_dump)
    df = pd.concat([pd.read_hdf(data_dump, df_name) for df_name in existed_stat])
    return df


if __name__ == '__main__':

    optimizers_calc_stat(GRID, DATA_DUMP, overwrite=True)
    stat_df = optimizers_collect_stat(DATA_DUMP)
