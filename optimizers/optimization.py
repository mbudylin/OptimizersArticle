from time import time


def timeit(function):
    def wrapper(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        duration = time()-t0
        return result, duration
    return wrapper


@timeit
def pricing_optimization(data, opt_model, opt_params, solver, solver_option={}):
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
