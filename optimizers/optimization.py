from time import time
from functools import wraps


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        duration = time() - t0
        return result, duration

    return wrapper


@timeit
def pricing_optimization(data, opt_model, opt_params, solver, solver_option={}):
    """
    Запуск расчета оптимальных цен с помощью указанного класса оптимизатора и параметров
    :param data: входные данные для оптимизации
    :param opt_model: класс модели оптимизатора
    :param opt_params: параметры оптимизации
    :param solver: солвер для оптиимзации
    :param solver_option: параметры солвера
    :return: словарь, возвращаемый моделью оптимизации
    """
    alpha = opt_params['alpha']
    model = opt_model(data, alpha)

    model.init_variables()
    model.init_objective()

    if opt_params.get('con_mrg') is not None:
        model.add_con_mrg(opt_params['con_mrg'])

    if opt_params.get('con_rev') is not None:
        model.add_con_rev(opt_params['con_rev'])

    if opt_params.get('con_equal') is not None and opt_params['con_equal']:
        model.add_con_equal()

    if opt_params.get('con_chg_cnt') is not None:
        model.add_con_chg_cnt(opt_params['con_chg_cnt'])

    result = model.solve(solver=solver, options=solver_option)

    return result
