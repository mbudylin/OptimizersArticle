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

bounds_params = {
    'main_bounds': {
        'lower': 0.9, 'upper': 1.1
    },
    'market_bounds': {
        'lower': 0.85, 'upper': 1.15
    }
}

def get_keys(filename_stat_dump):
    existed_stat = []
    if os.path.exists(filename_stat_dump):
        with pd.HDFStore(filename_stat_dump) as hdf:
            existed_stat = list(hdf.keys())
    return [el.replace('/', '') for el in existed_stat]


def optimizers_calculate_stat(grid, filename_stat_dump, overwrite=False):

    existed_stat = get_keys(filename_stat_dump)

    for N, seed in grid:
        dump_key = str((N, seed))
        # print(dump_key)
        # print(existed_stat)
        # assert ()
        if not overwrite:
            if dump_key in existed_stat:
                continue
        print('--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--')
        print(N, seed)
        # генерация данных для непрерывной NLP оптимизации
        data_c = generate_data(N, seed)
        data_c = construct_bounds(data_c, bounds_params)
        # генерация данных для MILP оптимизации
        data_d = construct_lp_grid(data_c, bounds_params, 21)

        M_cur = sum(data_c['Q'] * (data_c['P'] - data_c['C']))
        P_cur = sum(data_c['Q'] * data_c['P'])

        opt_params = {
            'alpha': 0.0,
            'con_mrg': M_cur,
            'con_equal': True
        }

        statuses = []
        times = []
        solvers = []

        if N < 1000:
            res, t = pricing_optimization(data_c, ScipyNlpOptimizationModel, opt_params, 'slsqp')
            statuses.append('ok' if res['status'] == '0' else res['status'])
            times.append(t)
            solvers.append('slsqp')
            print(f'slsqp finish \t{t}')

        if N < 500:
            res, t = pricing_optimization(data_c, ScipyNlpOptimizationModel, opt_params, 'trust-constr')
            statuses.append('ok' if res['status'] == '1' else res['status'])
            times.append(t)
            solvers.append('trust-constr')
            print(f'trust finish \t{t}')

        res, t = pricing_optimization(data_c, PyomoNlpOptimizationModel, opt_params, 'ipopt')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('ipopt')
        print(f'ipopt finish \t{t}')

        res, t = pricing_optimization(data_d, PyomoLpOptimizationModel, opt_params, 'cbc')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo_cbc')
        print(f'pyomo cbc finish \t{t}')

        res, t = pricing_optimization(data_d, PyomoLpOptimizationModel, opt_params, 'glpk')
        statuses.append('ok' if res['status'] == 'ok' else res['status'])
        times.append(t)
        solvers.append('pyomo_glpk')
        print(f'pyomo glpk finish \t{t}')

        res, t = pricing_optimization(data_d, CvxpyLpOptimizationModel, opt_params, 'CBC')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy_cbc')
        print(f'cvxpy cbc finish \t{t}')

        res, t = pricing_optimization(data_d, CvxpyLpOptimizationModel, opt_params, 'GLPK_MI')
        statuses.append('ok' if res['status'] == 'optimal' else res['status'])
        times.append(t)
        solvers.append('cvxpy_glpk')
        print(f'cvxpy glpk finish \t{t}')

        if N < 2000:
            res, t = pricing_optimization(data_d, CvxpyLpOptimizationModel, opt_params, 'ECOS_BB')
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
        df.to_hdf(filename_stat_dump, dump_key)


def optimizers_collect_stat(data_dump):

    existed_stat = get_keys(data_dump)
    stats_df = pd.concat([pd.read_hdf(data_dump, df_name) for df_name in existed_stat])
    return stats_df


if __name__ == '__main__':

    optimizers_calculate_stat(GRID, DATA_DUMP)
    stats_df = optimizers_collect_stat(DATA_DUMP)