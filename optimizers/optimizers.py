import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize
import cvxpy as cp
import pyomo.environ as pyo


class ScipyNlpOptimizationModel:
    """
    Класс, который создаёт оптимизационную модель на базе библиотеки scipy
    """

    def __init__(self, data: pd.DataFrame, alpha: int):
        if (alpha < 0.0) | (alpha > 1.0):
            raise ValueError('alpha должен быть между 0 и 1')
        self.alpha = float(alpha)
        self.data = data['data_nlp'].copy()
        self.N = len(self.data['plu_idx'])
        self.plu_idx_in_line = data['plu_idx_in_line'].copy()
        self.plu_idx = self.data['plu_idx'].values
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.plu_mask = None
        self.P = self.data['P'].values
        self.Q = self.data['Q'].values
        self.E = self.data['E'].values
        self.PC = self.data['PC'].values
        self.C = self.data['C'].values
        # границы для индексов
        self.x_lower = self.data['x_lower'].values
        self.x_upper = self.data['x_upper'].values
        self.x_init = self.data['x_init'].values
        self.idx_map = None
        # Задаём объект для модели scipy
        self.obj = None
        self.jac = None
        self.bounds = None
        self.x0 = None
        self.constraints = []

        self.k = 0.1 * sum(self.P * self.Q)

    def _el(self, E, x):
        return np.exp(E * (x - 1.0))

    def init_variables(self):
        """
        Инициализация переменных в модели
        """

        self.idx_map = self.plu_line_idx
        self.N = len(np.unique(self.idx_map))

        self.bounds = np.array([[None] * self.N] * 2, dtype=float)
        for plu_line_idx_, plu_ in self.plu_idx_in_line.items():
            self.bounds[0][plu_line_idx_] = self.x_lower[plu_[0]]
            self.bounds[1][plu_line_idx_] = self.x_upper[plu_[0]]

        # self.bounds = np.array([self.x_lower, self.x_upper])
        self.x0 = 0.5 * (self.bounds[0] + self.bounds[1])
        A = np.eye(self.N, self.N)
        constr_bounds = LinearConstraint(A, self.bounds[0], self.bounds[1])
        self.constraints.append(constr_bounds)

    def init_objective(self):
        """
        Инициализация целевой функции(выручка или фронт-маржа)
        """
        def objective(x):
            x_ = x[self.idx_map[self.plu_idx]]
            f = -sum(
                (self.P * x_ - self.alpha * self.C) * self.Q * self._el(self.E, x_)
            )
            return f / self.k
        self.obj = objective

    def add_con_mrg(self, m_min, m_max=np.inf):
        """
        Добавление в модель ограничения на маржу
        """
        def con_mrg(x):
            x_ = x[self.idx_map[self.plu_idx]]
            m = sum((self.P * x_ - self.C) * self.Q * self._el(self.E, x_))
            return m
        constr = NonlinearConstraint(con_mrg, m_min, m_max)
        self.constraints.append(constr)

    def add_con_equal(self):
        """
        Добавление ограничения на равенство цен внутри группы
        """
        n_con = sum(len(plu_idx) for plu_idx in self.plu_idx_in_line.values()) - len(self.plu_idx_in_line)

        if len(self.plu_idx_in_line) == 0:
            return

        A = np.zeros((n_con, self.N))
        i_con = 0
        for plu_line, plu_idxes in self.plu_idx_in_line.items():
            for plu_idx1, plu_idx2 in zip(plu_idxes[:-1], plu_idxes[1:]):
                A[i_con, plu_idx1] = 1.
                A[i_con, plu_idx2] = -1.
                i_con += 1
        constr = LinearConstraint(A, 0.0, 0.0)
        self.constraints.append(constr)

    def solve(self, solver='slsqp', options={}):
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
        result = minimize(self.obj, self.x0, method=solver,
                          constraints=self.constraints, options=options)
        self.data['x_opt'] = result['x'][self.idx_map[self.plu_idx]]
        self.data['P_opt'] = self.data['x_opt'] * self.data['P']
        self.data['Q_opt'] = self.Q * self._el(self.E, self.data['x_opt'])

        return {
            'message': str(result['message']),
            'status': str(result['status']),
            'model': result,
            'data': self.data
        }


class PyomoNlpOptimizationModel:
    """
    Класс, который создаёт оптимизационную модель на базе библиотеки pyomo
    """

    def __init__(self, data: pd.DataFrame, alpha: int):
        if (alpha < 0.0) | (alpha > 1.0):
            raise ValueError('alpha должен быть между 0 и 1')
        self.alpha = float(alpha)
        self.data = data['data_nlp'].copy()
        self.plu_idx_in_line = data['plu_idx_in_line'].copy()
        self.N = len(self.data['plu_idx'])
        self.plu = self.data['plu_idx'].to_list()
        self.P = self.data['P'].to_list()
        self.Q = self.data['Q'].to_list()
        self.E = self.data['E'].to_list()
        self.PC = self.data['PC'].to_list()
        self.C = self.data['C'].to_list()
        # границы для индексов
        self.x_lower = self.data['x_lower'].to_list()
        self.x_upper = self.data['x_upper'].to_list()
        self.x_init = self.data['x_init'].to_list()
        self.fixed = self.data['fixed'].to_list()
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
        for i, fix in enumerate(self.fixed):
            if fix:
                self.model.x[i].fixed = True

    def init_objective(self):
        """
        Инициализация целевой функции(выручка или фронт-маржа)
        """
        objective = sum(
            (self.P[i] * self.model.x[i] - self.alpha * self.C[i]) *
            self.Q[i] * self._el(i) for i in range(self.N)
        )
        self.model.obj = pyo.Objective(expr=objective, sense=pyo.maximize)

    def add_con_rev(self, s_min, s_max=None):
        """
        Добавление в модель ограничения на выручку
        """

        def con_rev(model):
            r = sum(
                self.P[i] * model.x[i] * self.Q[i] * self._el(model, i)
                for i in range(self.N)
            )
            return s_min, r, s_max

        self.model.con_rev = pyo.Constraint(rule=con_rev)

    def add_con_mrg(self, m_min, m_max=None):
        """
        Добавление в модель ограничения на маржу
        """

        def con_mrg(model):
            r = sum(
                (self.P[i] * model.x[i] - self.C[i]) * self.Q[i] * self._el(i)
                for i in range(self.N)
            )
            return m_min, r, m_max

        self.model.con_mrg = pyo.Constraint(rule=con_mrg)

    def add_con_equal(self):
        """
        Добавление ограничения на равенство цен внутри линейки
        """
        if len(self.plu_idx_in_line) == 0:
            return

        self.model.con_equal = pyo.Constraint(pyo.Any)
        # название ограничения = plu_line
        for con_idx, idxes in self.plu_idx_in_line.items():
            for i in range(1, len(idxes)):
                con_name = str(con_idx) + '_' + str(i)
                self.model.con_equal[con_name] = (self.model.x[idxes[i]] - self.model.x[idxes[i - 1]]) == 0

    def add_con_chg(self, chg_max=None):
        """
        Добавление в модель ограничения на 'разброс' цен относительно текущей
        """
        chg_max2 = chg_max ** 2

        def con_chg(model):
            r = sum(
                ((self.model.x[i]) - 1) ** 2
                for i in range(self.N)
            ) / self.N
            return None, r, chg_max2

        self.model.con_chg = pyo.Constraint(rule=con_chg)

    def add_con_chg_cnt(self, thr_l=0.98, thr_u=1.02, nmax=10000):
        """
        Добавление в модель ограничения на изменение цен
        """
        self.model.ind_l = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_r = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_m = pyo.Var(range(self.N), domain=pyo.Binary, initialize=1)

        self.model.x_interval = pyo.Constraint(pyo.Any)
        self.model.con_ind = pyo.Constraint(pyo.Any)

        for i in range(self.N):
            self.model.x_interval['l' + str(i)] = self.model.x[i] <= thr_l + 10. * (1 - self.model.ind_l[i])
            self.model.x_interval['r' + str(i)] = self.model.x[i] >= thr_u - 10. * (1 - self.model.ind_r[i])
            self.model.x_interval['ml' + str(i)] = self.model.x[i] <= 1. + 10. * (1 - self.model.ind_m[i])
            self.model.x_interval['mr' + str(i)] = self.model.x[i] >= 1. - 10. * (1 - self.model.ind_m[i])
            self.model.con_ind[i] = (self.model.ind_l[i] + self.model.ind_m[i] + self.model.ind_r[i]) == 1

        self.model.con_max_chg = pyo.Constraint(expr=sum(
            self.model.ind_m[i]
            for i in range(self.N)
        ) >= self.N - nmax)

    def solve(self, solver='ipopt', options={}):
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
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


class PyomoLpOptimizationModel:
    """
    Класс, который создаёт оптимизационную модель на базе библиотеки pyomo
    """

    def __init__(self, data: pd.DataFrame, alpha: int):
        if (alpha < 0.0) | (alpha > 1.0):
            raise ValueError('alpha должен быть между 0 и 1')
        self.alpha = float(alpha)
        self.data = data['data_milp'].copy()
        self.N = len(self.data['plu_line_idx'])
        self.grid_size = self.data['grid_size'].values
        self.g_max = max(self.grid_size)
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.n_plu = self.data['n_plu'].values
        self.P = self.data['P'].values
        self.P_idx = self.data['P_idx'].values
        self.PC = self.data['PC'].values
        self.Ps = np.vstack(self.data['Ps'].values)
        self.Qs = np.vstack(self.data['Qs'].values)
        self.C = self.data['C'].values
        # границы для индексов
        self.xs = self.data['xs'].values
        self.fixed = self.data['fixed'].values
        # Задаём объект модели pyomo
        self.model = pyo.ConcreteModel()

    def init_variables(self):
        """
        Инициализация переменных в модели
        """

        # задаем бинарную метку для цены
        def init_fun(model, i, j):
            return 1 if self.P_idx[i] == j else 0

        self.model.x = pyo.Var(range(self.N), range(self.g_max), initialize=init_fun, domain=pyo.Binary)
        # одна единичка для каждого товара
        self.model.con_any_price = pyo.Constraint(pyo.Any)
        for i in range(self.N):
            self.model.con_any_price[i] = sum(self.model.x[i, j] for j in range(self.grid_size[i])) == 1

        for i, fix in enumerate(self.fixed):
            if fix:
                self.model.x[i, self.P_idx] = 1
                self.model.x[i, :].fixed = True

    def init_objective(self):
        """
        Инициализация целевой функции(выручка или фронт-маржа)
        """
        expr = sum(sum((self.Ps - self.alpha * self.C.reshape(-1, 1)) * self.Qs * self.model.x))
        self.model.obj = pyo.Objective(expr=expr, sense=pyo.maximize)

    def add_con_rev(self, s_min, s_max=None):
        """
        Добавление в модель ограничения на РТО
        """
        def con_rev(model):
            r = sum(
                self.Ps[i][j] * model.x[i, j] * self.Qs[i][j]
                for i in range(self.N)
                for j in range(self.grid_size[i])
            )
            return s_min, r, s_max

        self.model.con_rev = pyo.Constraint(rule=con_rev)

    def add_con_mrg(self, m_min, m_max=None):
        """
        Добавление в модель ограничения на маржу
        """
        def con_mrg(model):
            r = sum(
                model.x[i, j] * (self.Ps[i][j] - self.C[i]) * self.Qs[i][j]
                for i in range(self.N)
                for j in range(self.grid_size[i])
            )
            return m_min, r, m_max

        expr = sum(sum((self.Ps - self.C.reshape(-1, 1)) * self.Qs * self.model.x)) >= m_min
        self.model.con_mrg = pyo.Constraint(expr=expr)

    def add_con_equal(self):
        pass

    def add_con_chg_cnt(self, nmax=10000):
        """
        Добавление ограничения на количество изменяемых цен
        """
        con_expr = sum(self.model.x[i, self.P_idx[i]] * self.n_plu[i]
                       for i in range(self.N) if self.P_idx[i] > 0) >= sum(self.n_plu) - nmax
        self.model.con_chg_cnt = pyo.Constraint(expr=con_expr)

    def solve(self, solver='cbc', options={}):
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
        solver = pyo.SolverFactory(solver, io_format='python', symbolic_solver_labels=False)
        for option_name, option_value in options.items():
            solver.options[option_name] = option_value
        result = solver.solve(self.model)
        x_sol = [[self.model.x[i, j].value for j in range(self.grid_size[i])] for i in range(self.N)]
        x_opt_idx = [np.argmax(x_sol[i]) for i in range(self.N)]
        x_opt = [self.xs[i][x_opt_idx[i]] for i in range(self.N)]
        P_opt = [self.Ps[i][x_opt_idx[i]] for i in range(self.N)]
        Q_opt = [self.Qs[i][x_opt_idx[i]] for i in range(self.N)]

        self.data['P_opt'] = P_opt
        self.data['Q_opt'] = Q_opt
        self.data['x_opt'] = x_opt
        return {
            'message': str(result.solver.termination_condition),
            'status': str(result.solver.status),
            'model': self.model,
            'data': self.data,
            'x_sol': x_sol,
            'opt_idx': x_opt_idx
        }


class CvxpyLpOptimizationModel:
    """
    Класс, который создаёт оптимизационную модель на базе библиотеки pyomo
    """
    def __init__(self, data: pd.DataFrame, alpha: int):
        if (alpha < 0.0) | (alpha > 1.0):
            raise ValueError('alpha должен быть между 0 и 1')
        self.alpha = float(alpha)
        self.data = data['data_milp'].copy()
        self.N = len(self.data['plu_line_idx'])
        self.grid_size = self.data['grid_size'].values
        self.g_max = max(self.grid_size)
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.n_plu = self.data['n_plu'].values
        self.P = self.data['P'].values
        self.P_idx = self.data['P_idx'].values
        self.PC = self.data['PC'].values
        self.Ps = np.array(self.data['Ps'].to_list())
        self.Qs = np.array(self.data['Qs'].to_list())
        self.C = self.data['C'].values.reshape(-1, 1)
        # границы для индексов
        self.xs = np.array(self.data['xs'].to_list())
        self.fixed = self.data['fixed'].values
        # Задаём объекты для формирования
        self.x = None
        self.obj = None
        self.constraints = []
        self.x_mask = None

    def init_variables(self):
        """
        Инициализация переменных в модели
        """
        self.x = cp.Variable(shape=(self.N, self.g_max), boolean=True)
        # должна быть хотя бы одна цена из диапазона
        # вспомогательная маска для упрощения матричных операций при формирований задачи
        mask_idx = np.repeat(np.arange(self.g_max), self.N).reshape(self.g_max, self.N).T
        mask = np.ones((self.N, self.g_max))
        mask[mask_idx > np.array(self.grid_size).reshape(-1, 1) - 1] = 0
        self.x_mask = mask
        con_any_price = cp.sum(cp.multiply(self.x, self.x_mask), axis=1) == 1
        self.constraints.append(con_any_price)

        if sum(self.fixed) > 0:
            con_fix = self.x[self.fixed == 1, self.P_idx[self.fixed == 1]] == 1
            self.constraints.append(con_fix)

    def init_objective(self):
        """
        Инициализация целевой функции(выручка или фронт-маржа)
        """
        self.obj = cp.Maximize(cp.sum(cp.multiply(self.x, (self.Ps - self.alpha * self.C) * self.Qs)))

    def add_con_rev(self, s_min, s_max=None):
        """
        Добавление в модель ограничения на РТО
        """
        con_mrg = cp.sum(cp.multiply(self.x, self.Ps * self.Qs)) >= s_min
        self.constraints.append(con_mrg)

    def add_con_mrg(self, m_min, m_max=None):
        """
        Добавление в модель ограничения на маржу
        """
        con_mrg = cp.sum(cp.multiply(self.x, (self.Ps - self.C) * self.Qs)) >= m_min
        self.constraints.append(con_mrg)

    def add_con_equal(self):
        pass

    def add_con_chg_cnt(self, nmax=10000):
        """
        Добавление ограничения на количество изменяемых цен
        """
        con_chg_cnt = cp.sum(
            cp.multiply(self.x[np.arange(self.N), self.P_idx], self.n_plu)[self.P_idx > 0]
        ) >= sum(self.n_plu) - nmax
        self.constraints.append(con_chg_cnt)

    def solve(self, solver='ECOS_BB', options={}):
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
        problem = cp.Problem(self.obj, self.constraints)
        problem.solve(solver, **options)

        if self.x.value is None:
            return {
                'message': str(problem.solution.status),
                'status': str(problem.status),
                'model': problem,
                'data': self.data,
            }

        x_opt_idx = [np.argmax(self.x.value[i, :self.grid_size[i]]) for i in range(self.N)]
        x_opt = self.xs[np.arange(self.N), x_opt_idx]
        P_opt = self.Ps[np.arange(self.N), x_opt_idx]
        Q_opt = self.Qs[np.arange(self.N), x_opt_idx]
        self.data['P_opt'] = P_opt
        self.data['Q_opt'] = Q_opt
        self.data['x_opt'] = x_opt

        return {
            'message': str(problem.solution.status),
            'status': str(problem.status),
            'model': problem,
            'data': self.data,
            'opt_idx': x_opt_idx
        }
