import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import triang, uniform
from numpy.random import triangular, uniform
from scipy.optimize import minimize


def calc_Q(Q0, P0, P, E):
    # расчёт спроса для новой цены по формуле спроса от цены и эластичности.
    Q = Q0 * np.exp(E * (P / P0 - 1.))
    return Q


def sku_sim(sku_id):
    # сгенерируем текущий объём продаж и маржу
    Q0 = uniform(10, 100)  # текущий объём
    P0 = uniform(50, 1000)  # текущая цена
    M0 = triangular(0, 0.25, 0.3)  # текущая маржа
    E = triangular(-10, -1, -0.01)  # текущая маржа

    C = P0 * (1. - M0)  # себестоимость

    return {'sku_id': sku_id,
            'Q0': Q0,
            'P0': P0,
            'M0': M0,
            'C': C,
            'E': E}


def sim_group(N):
    # сгенерируем данные для N товаров
    return [sku_sim(i) for i in range(1, N + 1)]


# def find_price_constr(sku, target_percent):
#     # найти цену, чтобы продажи товара выросли на заданный процент

#     Q0 = sku['Q0']
#     P0 = sku['P0']
#     E = sku['E']

#     def func(x):
#         q_new = calc_Q(Q0=Q0, P0=P0, P=x, E=E)
#         sales_delta = (1. - (Q0*P0) / (q_new*x))
#         pi_new = x / P0
#         return abs(sales_delta - target_percent) + abs(1.02 - pi_new)

#     res = minimize(func, x0=P0, tol=1e-6)
#     Q_NEW = calc_Q(Q0=Q0, P0=P0, P=res.x[0], E=E)
#     P_NEW = res.x[0]

#     sku['P_NEW_CONSTR'] = P_NEW
#     sku['Q_NEW_CONSTR'] = Q_NEW
#     sku['M_NEW_CONSTR'] = (1. - sku['C'] / sku['P_NEW_CONSTR'])
#     sku['PI_CONSTR'] = sku['P0'] / sku['P_NEW_CONSTR']
#     return sku

def find_price(sku, target_percent):
    # найти цену, чтобы продажи товара выросли на заданный процент

    Q0 = sku['Q0']
    P0 = sku['P0']
    E = sku['E']

    def func(x):
        q_new = calc_Q(Q0=Q0, P0=P0, P=x, E=E)
        return abs((1. - (Q0 * P0) / (q_new * x)) - target_percent)

    res = minimize(func, x0=P0, tol=1e-6)
    Q_NEW = calc_Q(Q0=Q0, P0=P0, P=res.x[0], E=E)
    P_NEW = res.x[0]

    sku['P_NEW'] = P_NEW
    sku['Q_NEW'] = Q_NEW
    sku['M_NEW'] = (1. - sku['C'] / sku['P_NEW'])
    sku['PI'] = sku['P_NEW'] / sku['P0']
    return sku


def sales_total_current(data):
    return sum([el['Q0'] * el['P0'] for el in data])


def sales_total_new(data):
    return sum([el['Q_NEW'] * el['P_NEW'] for el in data])


def sales_total_new_constr(data):
    return sum([el['Q_NEW_CONSTR'] * el['P_NEW_CONSTR'] for el in data])


def margin_total_current(data):
    return sum([el['Q0'] * el['P0'] * el['M0'] for el in data])


def margin_total_new(data):
    return sum([el['Q_NEW'] * el['P_NEW'] * el['M_NEW'] for el in data])


def margin_total_new_constr(data):
    return sum([el['Q_NEW_CONSTR'] * el['P_NEW_CONSTR'] * el['M_NEW_CONSTR'] for el in data])


def change_sales(SP):
    # предположим, что стоит задача поднять суммарные продажи на 2%

    pass

def sim_data():
    np.random.seed(5)
    data = sim_group(40)
    data = list(map(lambda x: find_price(x, 0.02), data))
    # data_constr = list(map(lambda x: find_price_constr(x, 0.02), data))
    return data


def plot_what_if(data):

    sales_diff = 100 * round(sales_total_new(data) / sales_total_current(data) - 1, 4)
    margin_diff = 100 * round(margin_total_new(data) / margin_total_current(data) - 1, 4)
    e1_cnt = sum([1 if el['E'] <= -1 else 0 for el in data])
    e1_up_cnt = sum([1 if el['E'] <= -1 and (el['P_NEW'] > el['P0']) else 0 for el in data])

    t0 = f'количество товаров: {len(data)}; кол-во с E <= -1: {e1_cnt} шт среди которых на повышение: {e1_up_cnt} шт'
    t1 = f'Прогнозируемое изменение РТО на {sales_diff}%, ВД на {margin_diff}%'

    a = [(el['PI'] - 1.) * 100. for el in data]
    b = [el['E'] for el in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'{t0}\n {t1}')
    sns.scatterplot(b, a, ax=ax1)
    ax1.set_xlabel('E')
    ax1.set_ylabel('индекс')
    ax2.set_xlabel('индекс')
    ax2.set_ylabel('кол-во')
    sns.histplot(a, bins=25, ax=ax2)
