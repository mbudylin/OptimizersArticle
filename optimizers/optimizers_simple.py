"""
Классы оптимизаторов.
"""
from typing import Dict
import abc
import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize
import cvxpy as cp
import pyomo.environ as pyo


class OptimizationModel(abc.ABC):
    """
    Базовый класс для оптимизаторов с ЦО
    """

    def __init__(self, data_sources, table_link):

        self.data_sources = data_sources
        self.data = data_sources[table_link].copy()
        self.plu_idx_in_line = data_sources['plu_idx_in_line'].copy()

        if 'plu_idx' in self.data.columns:
            self.plu_idx = self.data['plu_idx'].values

        self.N = self.data['plu_line_idx'].nunique()
        self.N_SIZE = len(self.data['plu_line_idx'])
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.P = self.data['P'].values

        if 'Q' in self.data.columns:
            self.Q = self.data['Q'].values

        if 'E' in self.data.columns:
            self.E = self.data['E'].values

        self.PC = self.data['PC'].values
        self.C = self.data['C'].values

        # границы для индексов
        if 'x_lower' in self.data.columns:
            self.x_lower = self.data['x_lower'].values
        if 'x_upper' in self.data.columns:
            self.x_upper = self.data['x_upper'].values
        if 'x_init' in self.data.columns:
            self.x_init = self.data['x_init'].values

    @abc.abstractmethod
    def init_variables(self):
        """
        Инициализация переменных в модели
        """
        pass

    @abc.abstractmethod
    def init_objective(self):
        """
        Инициализация целевой функции - выручка
        """
        pass

    @abc.abstractmethod
    def init_constraints(self):
        """
        Добавление в модель ограничения на маржу
        """
        pass

    @abc.abstractmethod
    def solve(self, solver, options) -> Dict:
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
        pass


class ScipyModel(OptimizationModel):
    """
    Класс, который создаёт NLP оптимизационную модель на базе библиотеки scipy
    """

    def __init__(self, data):
        super().__init__(data, 'data_nlp')

        # Задаём объект для модели scipy
        self.obj = None
        self.bounds = None
        self.x0 = None
        self.constraints = []

        # нормировка для целевой функции
        self.k = 0.1 * sum(self.P * self.Q)

    def _el(self, E, x):
        return np.exp(E * (x - 1.0))

    def init_variables(self):
        self.bounds = np.array([[None] * self.N] * 2, dtype=float)

        for plu_line_idx_, plu_ in self.plu_idx_in_line.items():
            self.bounds[0][plu_line_idx_] = self.x_lower[plu_[0]]
            self.bounds[1][plu_line_idx_] = self.x_upper[plu_[0]]

        self.x0 = 0.5 * (self.bounds[0] + self.bounds[1])
        # self.x0 = np.random.uniform(self.bounds[0], self.bounds[1])
        A = np.eye(self.N, self.N)
        constr_bounds = LinearConstraint(A, self.bounds[0], self.bounds[1])
        self.constraints.append(constr_bounds)

    def init_objective(self):
        def objective(x):
            x_ = x[self.plu_line_idx[self.plu_idx]]
            f = -sum(self.P * x_ * self.Q * self._el(self.E, x_))
            return f / self.k

        self.obj = objective

    def add_con_mrg(self, m_min):
        def con_mrg(x):
            x_ = x[self.plu_line_idx[self.plu_idx]]
            m = sum((self.P * x_ - self.C) * self.Q * self._el(self.E, x_))
            return m
        constr = NonlinearConstraint(con_mrg, m_min, np.inf)
        self.constraints.append(constr)

    def solve(self, solver='slsqp', options={}):

        result = minimize(self.obj,
                          self.x0,
                          method=solver,
                          constraints=self.constraints,
                          options=options)

        self.data['x_opt'] = result['x'][self.plu_line_idx[self.plu_idx]]
        self.data['P_opt'] = self.data['x_opt'] * self.data['P']
        self.data['Q_opt'] = self.Q * self._el(self.E, self.data['x_opt'])

        return {
            'message': str(result['message']),
            'status': str(result['status']),
            'model': result,
            'data': self.data
        }


class PyomoModel(OptimizationModel):

    def __init__(self, data):
        super().__init__(data, 'data_nlp')

        self.N = len(self.data['plu_idx'])

        # Задаём объект модели pyomo
        self.model = pyo.ConcreteModel()

    def _el(self, i):
        # вспомогательная функция для пересчета спроса при изменении цены
        return pyo.exp(self.E[i] * (self.model.x[i] - 1.0))

    def init_variables(self):
        def bounds_fun(model, i):
            return self.x_lower[i], self.x_upper[i]

        def init_fun(model, i):
            return self.x_init[i]

        self.model.x = pyo.Var(range(self.N), domain=pyo.Reals, bounds=bounds_fun, initialize=init_fun)

        # добавление условия на равенство цен в линейке
        if len(self.plu_idx_in_line) == 0:
            return

        self.model.con_equal = pyo.Constraint(pyo.Any)

        # название ограничения = plu_line_idx
        for con_idx, idxes in self.plu_idx_in_line.items():
            for i in range(1, len(idxes)):
                con_name = str(con_idx) + '_' + str(i)
                self.model.con_equal[con_name] = (self.model.x[idxes[i]] - self.model.x[idxes[i - 1]]) == 0

    def init_objective(self):
        objective = sum(self.P[i] * self.model.x[i] * self.Q[i] * self._el(i) for i in range(self.N))
        self.model.obj = pyo.Objective(expr=objective, sense=pyo.maximize)

    def add_con_mrg(self, m_min):
        con_mrg_expr = sum((self.P[i] * self.model.x[i] - self.C[i]) * self.Q[i] * self._el(i)
                           for i in range(self.N)) >= m_min
        self.model.con_mrg = pyo.Constraint(rule=con_mrg_expr)

    def add_con_chg_cnt(self, nmax=10000, thr_l=0.98, thr_u=1.02, ):
        self.model.ind_l = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_r = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_m = pyo.Var(range(self.N), domain=pyo.Binary, initialize=1)

        self.model.x_interval = pyo.Constraint(pyo.Any)
        self.model.con_ind = pyo.Constraint(pyo.Any)
        K = 0.15
        for i in range(self.N):
            self.model.x_interval['l' + str(i)] = self.model.x[i] - K * (1 - self.model.ind_l[i]) <= thr_l
            self.model.x_interval['r' + str(i)] = self.model.x[i] + K * (1 - self.model.ind_r[i]) >= thr_u
            self.model.x_interval['ml' + str(i)] = self.model.x[i] - K * (1 - self.model.ind_m[i]) <= 1.
            self.model.x_interval['mr' + str(i)] = self.model.x[i] + K * (1 - self.model.ind_m[i]) >= 1.
            self.model.con_ind[i] = (self.model.ind_l[i] + self.model.ind_m[i] + self.model.ind_r[i]) == 1

        self.model.con_max_chg = pyo.Constraint(expr=sum(
            self.model.ind_m[i]
            for i in range(self.N)
        ) >= self.N - nmax)

    def solve(self, solver='ipopt', options={}):
        solver = pyo.SolverFactory(solver, tee=False)
        for option_name, option_value in options.items():
            solver.options[option_name] = option_value

        result = solver.solve(self.model)

        self.data['x_opt'] = [self.model.x[i].value for i in self.model.x]
        self.data['P_opt'] = self.data['x_opt'] * self.data['P']
        self.data['Q_opt'] = [self.Q[i] * pyo.value(self._el(i)) for i in self.model.x]

        return {
            'message': str(result.solver.termination_condition),
            'status': str(result.solver.status),
            'model': self.model,
            'data': self.data
        }



