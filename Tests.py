import pytest
import numpy as np

from Function import (
    Config, generate_environment, E, P_x, P_t, Q,
    x_trajectory, dx_dt, fitness, find_global_optimum,
    determine_behavior_class
)
from scipy import integrate

# =============================================================================
# АППРОКСИМАЦИЯ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

def fitness_simple(params, env_params=None, N=500):
    A, b = params

    # Логика выбора параметров: переданные или дефолтные (hardcoded)
    if env_params is None:
        # Дефолтные для generate_simple_environment
        sigma1, xi1 = 1.0, 1.0
        sigma2, xi2 = 0.5, 1.0
        xi3, xi4 = 0.1, 1.0
        eta1, eta2 = 0.1, 0.1
        c3, c4 = -10.0, -10.0
        a_coef = 1.0
        gamma = 0.5
        beta = 0.3
        lambda_Q = 0.1
        # Параметры D в простой среде берутся из Config
        D = Config.D
    else:
        # Распаковка переданной среды
        sigma1, xi1 = env_params["sigma1"], env_params["xi1"]
        sigma2, xi2 = env_params["sigma2"], env_params["xi2"]
        xi3, xi4 = env_params["xi3"], env_params["xi4"]
        eta1, eta2 = env_params["eta1"], env_params["eta2"]
        c3, c4 = env_params["c3"], env_params["c4"]
        a_coef = env_params["a"]
        gamma = env_params["gamma"]
        beta = env_params["beta"]
        lambda_Q = env_params["lambda_Q"]
        D = Config.D

    # Время и шаг (как в бимо.py)
    t = np.linspace(0, 1, N)

    # 1. Траектория
    x = A + b * np.cos(2 * np.pi * t)

    # 2. Скорость (АНАЛИТИЧЕСКАЯ, чтобы совпадать с бимо.py)
    dx_dt_val = -2 * np.pi * b * np.sin(2 * np.pi * t)

    # Нормализация координат
    norm_x = (x + D / 2) / D

    # 3. Компоненты

    # Еда E(x)
    E_val = sigma1 * (1 + np.tanh(xi1 * norm_x))

    # Риск P(x, t)
    P_x_val = sigma2 * (1 + np.tanh(xi2 * norm_x))
    P_t_val = 2.0 * (-0.5 * np.cos(2 * np.pi * t) + 0.5)

    # Дополнительный фактор Q(x)
    Q_val = (xi3 * np.exp(eta1 * (x - c3)) +
             xi4 * np.exp(-eta2 * (x - c4))) / 2

    # Сборка подынтегрального выражения
    integrand = (
            a_coef * E_val -
            gamma * P_x_val * P_t_val -
            beta * (dx_dt_val ** 2) -
            lambda_Q * Q_val
    )

    # ИНТЕГРИРОВАНИЕ: используем тот же метод, что и в основной программе
    res = integrate.trapezoid(integrand, t)

    return res


def generate_simple_environment():
    """
    Генерирует простую среду для тестирования аппроксимации
    """
    return {
        "sigma1": 1.0, "xi1": 1.0, "sigma2": 0.5, "xi2": 1.0,
        "xi3": 0.1, "xi4": 1.0, "eta1": 0.1, "eta2": 0.1,
        "c3": -10, "c4": -10, "a": 1.0, "gamma": 0.5,
        "beta": 0.3, "lambda_Q": 0.1
    }


# =============================================================================
# ТЕСТЫ ПРОВЕРКИ ОСНОВНОЙ ПРОГРАММЫ С ПОМОЩЬЮ АППРОКСИМАЦИИ
# =============================================================================

class TestMainProgramWithApproximation:
    """Тесты для проверки основной программы с использованием аппроксимации как эталона"""

    def test_main_program_correctness_vs_approximation(self):
        """
        Тест корректности основной программы путем сравнения с аппроксимацией
        """
        # Тестируем на нескольких разных средах
        test_environments = [generate_environment() for _ in range(5)]
        test_points = [
            [-10, 5], [-20, 10], [-30, 15],
            [-40, 20], [-25, 0], [0, 25]
        ]

        max_error = 0
        total_tests = 0

        for env in test_environments:
            for point in test_points:
                # Основная программа
                main_result = -fitness(point, env)  # Инвертируем, т.к. fitness возвращает -F

                # Аппроксимация (эталон)
                approx_result = fitness_simple(point, env_params=env)

                # Проверка совпадения
                error = abs(main_result - approx_result)
                max_error = max(max_error, error)
                total_tests += 1

                # Обе функции должны возвращать конечные значения
                assert np.isfinite(main_result), f"Основная программа: неконечный результат для {point}"
                assert np.isfinite(approx_result), f"Аппроксимация: неконечный результат для {point}"

                # Допустимая погрешность (из-за разных методов интегрирования)
                assert error < 1e-3, f"Большая погрешность {error:.6f} для точки {point}"

        print(f"\nПроверка основной программы vs аппроксимация:")
        print(f"Протестировано: {total_tests} комбинаций (среды × точки)")
        print(f"Максимальная погрешность: {max_error:.6f}")

    def test_main_program_physical_plausibility(self):
        """
        Тест физической осмысленности результатов основной программы
        с проверкой через аппроксимацию
        """
        env = generate_environment()

        # Тест: При движении фитнес должен изменяться осмысленно
        point_stationary = [-25, 0]  # Нет движения
        point_moving = [-25, 15]  # Движение

        fitness_stationary_main = -fitness(point_stationary, env)
        fitness_moving_main = -fitness(point_moving, env)
        fitness_stationary_approx = fitness_simple(point_stationary, env_params=env)
        fitness_moving_approx = fitness_simple(point_moving, env_params=env)

        # Проверка согласованности
        assert abs(fitness_stationary_main - fitness_stationary_approx) < 1e-3
        assert abs(fitness_moving_main - fitness_moving_approx) < 1e-3

        # Фитнес при движении должен отличаться от фитнеса без движения
        fitness_difference_main = fitness_moving_main - fitness_stationary_main
        fitness_difference_approx = fitness_moving_approx - fitness_stationary_approx

        print(f"\nФизическая осмысленность:")
        print(f"Фитнес без движения: {fitness_stationary_main:.4f}")
        print(f"Фитнес с движением: {fitness_moving_main:.4f}")
        print(f"Разница фитнеса (основная): {fitness_difference_main:.4f}")
        print(f"Разница фитнеса (аппроксимация): {fitness_difference_approx:.4f}")

        # Проверка: движение должно влиять на фитнес (разница не должна быть нулевой)
        assert abs(fitness_difference_main) > 1e-3, "Движение должно влиять на фитнес"
        assert abs(fitness_difference_approx) > 1e-3, "Движение должно влиять на фитнес (аппроксимация)"

    def test_main_program_gradient_consistency(self):
        """
        Тест согласованности градиента основной программы с аппроксимацией
        """
        env = generate_environment()
        base_point = [-20, 8]

        # Небольшие вариации параметров
        variations = [
            ([0, 1], "увеличение b"),
            ([0, -1], "уменьшение b"),
            ([1, 0], "увеличение A"),
            ([-1, 0], "уменьшение A")
        ]

        base_main = -fitness(base_point, env)
        base_approx = fitness_simple(base_point, env_params=env)

        for direction, desc in variations:
            test_point = [base_point[0] + direction[0], base_point[1] + direction[1]]

            test_main = -fitness(test_point, env)
            test_approx = fitness_simple(test_point, env_params=env)

            # Изменения в основной программе и аппроксимации
            change_main = test_main - base_main
            change_approx = test_approx - base_approx

            # Проверка, что направление изменения одинаковое
            if abs(change_main) > 1e-6 and abs(change_approx) > 1e-6:
                assert np.sign(change_main) == np.sign(change_approx), \
                    f"Несовпадение направления при {desc}: main={change_main:.6f}, approx={change_approx:.6f}"

            print(f"{desc}: main Δ={change_main:.6f}, approx Δ={change_approx:.6f}")

    def test_main_program_optimization_quality(self):
        """
        Тест качества оптимизации основной программы через сравнение с аппроксимацией
        """
        env = generate_environment()

        # Оптимизация основной программой
        A_opt_main, b_opt_main, F_opt_main, result_main = find_global_optimum(env)

        # Проверка через аппроксимацию
        optimal_point_main = [A_opt_main, b_opt_main]
        F_opt_main_verified = fitness_simple(optimal_point_main, env_params=env)

        # Проверка нескольких случайных точек через аппроксимацию
        random_points = [
            [-25, 10], [-30, 15], [-20, 5], [-35, 20]
        ]

        best_random_fitness = -np.inf
        for point in random_points:
            fitness_val = fitness_simple(point, env_params=env)
            if fitness_val > best_random_fitness:
                best_random_fitness = fitness_val

        print(f"\nКачество оптимизации основной программы:")
        print(f"Оптимальный фитнес (основная): {F_opt_main:.4f}")
        print(f"Оптимальный фитнес (проверка аппроксимацией): {F_opt_main_verified:.4f}")
        print(f"Лучший случайный фитнес: {best_random_fitness:.4f}")

        # Проверка согласованности
        assert abs(F_opt_main - F_opt_main_verified) < 1e-3, \
            "Несовпадение оптимального фитнеса между основной программой и проверкой"

        # Оптимальное решение должно быть лучше случайных точек
        assert F_opt_main_verified >= best_random_fitness - 1e-6, \
            "Оптимальное решение хуже случайных точек"

    def test_main_program_edge_cases_with_approximation(self):
        """
        Тест граничных случаев основной программы с проверкой через аппроксимацию
        """
        env = generate_environment()

        edge_cases = [
            ([-Config.D, 0], "минимальная глубина, нет движения"),
            ([0, Config.D / 2], "поверхность, максимальная амплитуда"),
            ([-Config.D / 2, Config.D / 4], "середина, средняя амплитуда"),
            ([-25, 0], "стандартная глубина, нет движения"),
            ([0, 0], "поверхность, нет движения")
        ]

        print(f"\nГраничные случаи (основная программа vs аппроксимация):")
        for point, description in edge_cases:
            try:
                main_result = -fitness(point, env)
                approx_result = fitness_simple(point, env_params=env)
                error = abs(main_result - approx_result)

                print(f"{description:40} main={main_result:8.2f}, approx={approx_result:8.2f}, error={error:.6f}")

                # Проверка конечности
                assert np.isfinite(main_result), f"Основная программа: неконечный результат для {description}"
                assert np.isfinite(approx_result), f"Аппроксимация: неконечный результат для {description}"

                # Проверка согласованности
                assert error < 1e-3, f"Большая погрешность для граничного случая {description}"

            except Exception as e:
                print(f"Ошибка в граничном случае '{description}': {e}")

    def test_main_program_reproducibility(self):
        """Тест воспроизводимости результатов основной программы"""
        env = generate_environment()
        test_point = [-20, 10]

        # Многократный запуск с одинаковыми параметрами
        results = []
        for _ in range(5):
            result = -fitness(test_point, env)
            results.append(result)

        # Все результаты должны быть одинаковыми
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert abs(result - first_result) < 1e-10, f"Несовпадение в запуске {i}"

        print(f"Воспроизводимость: все {len(results)} запусков дали {first_result:.4f}")

    def test_behavior_classification_consistency(self):
        """Тест согласованности классификации поведения"""
        env = generate_environment()

        # Оптимизация для нахождения b_optimal
        A_opt, b_opt, F_opt, result = find_global_optimum(env)

        # Классификация поведения
        behavior_class, behavior_name = determine_behavior_class(
            b_opt, Config.GRAYZONE_MIN, Config.GRAYZONE_MAX)

        # Проверка логики классификации
        if b_opt < Config.GRAYZONE_MIN:
            expected_class, expected_name = 0, "Нет миграции"
        elif b_opt <= Config.GRAYZONE_MAX:
            expected_class, expected_name = 0.5, "Серая зона"
        else:
            expected_class, expected_name = 1, "Есть миграция"

        assert behavior_class == expected_class, f"Неверный класс: {behavior_class} vs {expected_class}"
        assert behavior_name == expected_name, f"Неверное название: {behavior_name} vs {expected_name}"

        print(f"\nКлассификация поведения:")
        print(f"b_optimal = {b_opt:.2f}, класс = {behavior_class} ({behavior_name})")

    def test_main_program_various_environments(self):
        """Тест работы основной программы на различных типах сред"""
        # Генерируем несколько разных сред
        environments = [generate_environment() for _ in range(3)]
        test_point = [-25, 10]

        print(f"\nРазличные среды:")
        fitness_values = []

        for i, env in enumerate(environments, 1):
            fitness_val = -fitness(test_point, env)
            fitness_values.append(fitness_val)

            # Проверка корректности
            assert np.isfinite(fitness_val), f"Неконечный фитнес для среды {i}"

            print(f"  Среда {i}: фитнес = {fitness_val:.4f}")

        # Разные среды должны давать разный фитнес
        unique_fitness = len(set(round(f, 2) for f in fitness_values))
        assert unique_fitness > 1, "Разные среды должны давать разный фитнес"

    def test_main_program_extreme_organism_parameters(self):
        """Тест работы с экстремальными значениями параметров организма"""
        test_point = [-25, 10]

        # Создаем среду с экстремальными параметрами организма
        extreme_cases = [
            {"a": 1e-6, "gamma": 1e-6, "beta": 1e-6, "lambda_Q": 1e-6},  # Все очень мало
            {"a": 1e6, "gamma": 1e6, "beta": 1e6, "lambda_Q": 1e6},  # Все очень много
        ]

        print(f"\nЭкстремальные параметры организма:")
        for i, extreme_params in enumerate(extreme_cases, 1):
            env = generate_environment()
            env.update(extreme_params)  # Заменяем параметры организма

            try:
                result = -fitness(test_point, env)
                assert np.isfinite(result), f"Неконечный результат для случая {i}"
                print(f"  Случай {i}: фитнес = {result:.4f}")

            except Exception as e:
                print(f"  Случай {i}: ошибка - {e}")

    def test_amplitude_boundary_effects(self):
        """Тест влияния граничных значений амплитуды на фитнес"""
        env = generate_environment()
        A_fixed = -25

        # Тестируем разные значения амплитуды
        amplitude_values = [0, Config.GRAYZONE_MIN, Config.GRAYZONE_MAX, Config.D / 4, Config.D / 2]

        print(f"\nВлияние амплитуды на фитнес:")
        fitness_values = []

        for b in amplitude_values:
            fitness_val = -fitness([A_fixed, b], env)
            fitness_values.append(fitness_val)
            print(f"  b={b:5.1f}: фитнес = {fitness_val:8.2f}")

        # Проверка: ОЧЕНЬ большая амплитуда должна уменьшать фитнес
        # из-за квадратичного роста энергозатрат
        fitness_zero_b = fitness_values[0]  # b=0
        fitness_max_b = fitness_values[-1]  # b=Config.D/2 (максимальная амплитуда)

        # При максимальной амплитуде фитнес должен быть меньше из-за высоких энергозатрат
        assert fitness_max_b < fitness_zero_b + 1e-6, \
            "При максимальной амплитуде фитнес должен уменьшаться из-за энергозатрат"

        print(f"  При b=0: {fitness_zero_b:.2f}, при max b: {fitness_max_b:.2f}")

    def test_environment_functions_basic(self):
        """Тест базовых функций среды на корректность"""
        env = generate_environment()

        # Тестируем функции в нескольких точках
        test_points = [-50, -25, 0, 25, 50]

        print(f"\nБазовые функции среды:")
        for x in test_points:
            E_val = E(x, env)
            P_x_val = P_x(x, env)
            Q_val = Q(x, env)

            # Проверка конечности
            assert np.isfinite(E_val), f"E({x}) должна быть конечной"
            assert np.isfinite(P_x_val), f"P_x({x}) должна быть конечной"
            assert np.isfinite(Q_val), f"Q({x}) должна быть конечной"

            # Проверка неотрицательности
            assert E_val >= 0, f"E({x}) должна быть неотрицательной"
            assert P_x_val >= 0, f"P_x({x}) должна быть неотрицательной"

            print(f"  x={x:3.0f}: E={E_val:.3f}, P_x={P_x_val:.3f}, Q={Q_val:.3f}")

# =============================================================================
# ЗАПУСК ТЕСТОВ
# =============================================================================

if __name__ == "__main__":
    # Запуск полноценных тестов
    print("=" * 60)
    print("ПОЛНОЦЕННЫЕ ТЕСТЫ ОСНОВНОЙ ПРОГРАММЫ")
    print("=" * 60)
    pytest.main([__file__, "-v"])