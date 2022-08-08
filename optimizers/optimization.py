from typing import Dict
from time import time
from functools import wraps
import pandas as pd

from optimizers.optimizers import OptimizationModel


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        duration = time() - t0
        return result, duration
    return wrapper


@timeit
def pricing_optimization(data: pd.DataFrame,
                         opt_model: OptimizationModel,
                         opt_params: Dict,
                         solver: str,
                         solver_option={}):
    """
    Запуск расчета оптимальных цен с помощью указанного класса оптимизатора и параметров
    :param data: входные данные для оптимизации
    :param opt_model: класс модели оптимизатора
    :param opt_params: параметры оптимизации
    :param solver: солвер для оптимизации
    :param solver_option: параметры солвера
    :return: словарь, возвращаемый моделью оптимизации
    """

    model = opt_model(data, opt_params['alpha'])

    model.init_variables()
    model.init_objective()

    if opt_params.get('con_mrg') is not None:
        model.add_con_mrg(opt_params['con_mrg'])

    if opt_params.get('con_chg_cnt') is not None:
        model.add_con_chg_cnt(opt_params['con_chg_cnt'])

    result = model.solve(solver=solver, options=solver_option)
    result['model_class'] = model
    return result
