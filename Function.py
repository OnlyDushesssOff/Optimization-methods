import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy import integrate
import pickle
import csv
import os

# =============================================================================
# БЛОК 1: КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# =============================================================================

class Config:
    """Класс для хранения конфигурационных параметров"""
    D = 100  # глубина водоёма
    GRAYZONE_MIN = 5
    GRAYZONE_MAX = 20
    T_POINTS = 500  # Количество точек для интегрирования

# =============================================================================
# БЛОК 2: ГЕНЕРАЦИЯ СРЕДЫ
# =============================================================================

def generate_environment():
    """
    Генерирует случайные параметры среды и организма
    """
    env = {
        # Параметры среды
        "sigma1": np.random.uniform(0.25, 8.61),
        "xi1": np.random.uniform(0.025, 9.18),
        "sigma2": np.random.uniform(0.003, 8.99),
        "xi2": np.random.uniform(0.025, 9.18),
        "xi3": np.random.uniform(0.025, 9.18),
        "xi4": np.random.uniform(0.5, 1.0),
        "eta1": np.random.uniform(0.05, 0.2),
        "eta2": np.random.uniform(0.05, 0.2),
        "c3": np.random.uniform(-20, -5),
        "c4": np.random.uniform(-90, -60),

        # Коэффициенты организма (теперь они генерируются, но не будут в признаках)
        "a": np.random.uniform(0.1, 1000),
        "gamma": np.random.uniform(0.001, 1000),
        "beta": np.random.uniform(1e-9, 0.001),
        "lambda_Q": np.random.uniform(1e-5, 1000),

        # Границы серой зоны
        "grayzone_min": Config.GRAYZONE_MIN,
        "grayzone_max": Config.GRAYZONE_MAX
    }
    return env

# =============================================================================
# БЛОК 3: ФУНКЦИИ СРЕДЫ И ТРАЕКТОРИИ
# =============================================================================

def E(x, env_params):
    """Функция распределения еды"""
    norm_x = (x + Config.D/2) / Config.D
    return env_params["sigma1"] * (1 + np.tanh(env_params["xi1"] * norm_x))

def P_x(x, env_params):
    """Пространственная компонента риска"""
    norm_x = (x + Config.D/2) / Config.D
    return env_params["sigma2"] * (1 + np.tanh(env_params["xi2"] * norm_x))

def P_t(t, sigma_2=2.0):
    """Временная компонента риска"""
    return sigma_2 * (-0.5 * np.cos(2 * np.pi * t) + 0.5)

def Q(x, env_params):
    """Дополнительная функция среды"""
    return (env_params["xi3"] * np.exp(env_params["eta1"] * (x - env_params["c3"])) +
            env_params["xi4"] * np.exp(-env_params["eta2"] * (x - env_params["c4"]))) / 2

def x_trajectory(t, A, b):
    """Траектория движения организма"""
    return A + b * np.cos(2 * np.pi * t)

def dx_dt(t, b):
    """Скорость движения организма"""
    return -2 * np.pi * b * np.sin(2 * np.pi * t)

# =============================================================================
# БЛОК 4: ФУНКЦИЯ ФИТНЕСА
# =============================================================================

def fitness(params, env_params):
    """
    Функция фитнеса F(A, b) - интеграл за сутки
    Возвращает -F(A, b) для минимизации
    """
    A, b = params

    t = np.linspace(0, 1, Config.T_POINTS)
    integrand_values = []

    for t_i in t:
        x_t = x_trajectory(t_i, A, b)
        velocity = dx_dt(t_i, b)

        # Компоненты фитнеса
        food_component = env_params["a"] * E(x_t, env_params)
        risk_component = env_params["gamma"] * P_x(x_t, env_params) * P_t(t_i)
        energy_component = env_params["beta"] * (velocity) ** 2
        other_component = env_params["lambda_Q"] * Q(x_t, env_params)

        integrand = food_component - risk_component - energy_component - other_component
        integrand_values.append(integrand)

    # Используем метод трапеций вместо простого суммирования
    total_fitness = integrate.trapezoid(integrand_values, t)

    return -total_fitness  # Возвращаем -F для минимизации

# =============================================================================
# БЛОК 5: ГЛОБАЛЬНАЯ ОПТИМИЗАЦИЯ
# =============================================================================

def find_global_optimum(env_params):
    """
    Находит ГЛОБАЛЬНЫЙ максимум F(A, b) методом дифференциальной эволюции
    """
    # Биологические и физические ограничения
    bounds = [(-Config.D, 0), (0, Config.D/2)]

    # Глобальная оптимизация методом дифференциальной эволюции
    result = differential_evolution(
        fitness,
        bounds=bounds,
        args=(env_params,),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        recombination=0.7,
        seed=42
    )

    if result.success:
        A_opt, b_opt = result.x
        F_opt = -result.fun  # F(A*, b*)
        return A_opt, b_opt, F_opt, result
    else:
        print(f"Предупреждение: Глобальная оптимизация не сошлась. Сообщение: {result.message}")
        # Резервная стратегия - используем лучшее найденное решение
        A_opt, b_opt = result.x
        F_opt = -result.fun
        return A_opt, b_opt, F_opt, result

# =============================================================================
# БЛОК 6: КЛАССИФИКАЦИЯ ПОВЕДЕНИЯ
# =============================================================================

def determine_behavior_class(b_optimal, grayzone_min, grayzone_max):
    """
    Присваивает класс поведения согласно найденному значению b*:
    • 0 — нет миграции, если b* < grayzone_min
    • 0.5 — серая зона, если grayzone_min ≤ b* ≤ grayzone_max
    • 1 — есть миграции, если b* > grayzone_max
    """
    if b_optimal < grayzone_min:
        return 0, "Нет миграции"
    elif grayzone_min <= b_optimal <= grayzone_max:
        return 0.5, "Серая зона"
    else:
        return 1, "Есть миграция"

# =============================================================================
# БЛОК 7: УЛУЧШЕННАЯ СИСТЕМА БАЛАНСИРОВКИ КЛАССОВ
# =============================================================================

class BalancedDatasetGenerator:
    """Класс для интеллектуальной генерации сбалансированного датасета"""
    
    def __init__(self):
        # Используем оригинальные диапазоны, но смещаем распределение для каждого класса
        self.class_biases = {
            0: {  # Нет миграции - смещаем в сторону высокого риска/стоимости
                "sigma2_bias": 0.7,    # 70% к верхней границе риска
                "beta_bias": 0.8,      # 80% к верхней границе стоимости
                "sigma1_bias": 0.3,    # 30% к верхней границе еды (меньше стимула)
                "gamma_bias": 0.8      # 80% к верхней границе чувствительности
            },
            1: {  # Есть миграция - смещаем в сторону низкого риска/стоимости
                "sigma2_bias": 0.3,    # 30% к верхней границе риска  
                "beta_bias": 0.2,      # 20% к верхней границе стоимости
                "sigma1_bias": 0.7,    # 70% к верхней границе еды (больше стимула)
                "gamma_bias": 0.3      # 30% к верхней границе чувствительности
            },
            0.5: {  # Серая зона - баланс вокруг середины
                "sigma2_bias": 0.5,    # 50% - средние значения
                "beta_bias": 0.5,      # 50% - средние значения
                "sigma1_bias": 0.5,    # 50% - средние значения
                "gamma_bias": 0.5      # 50% - средние значения
            }
        }
    
    def generate_balanced_dataset(self, num_samples=1000, save_path="migration_dataset_balanced.pkl"):
        """Основной метод генерации сбалансированного датасета"""
        dataset = self._initialize_dataset()
        
        print(f"🎯 Генерация сбалансированной выборки: {num_samples} образцов")
        target_counts = self._calculate_target_distribution(num_samples)
        
        # Генерация для каждого класса
        for target_class in [0, 0.5, 1]:
            self._generate_class_samples(target_class, target_counts[target_class], dataset)
        
        return self._finalize_dataset(dataset, save_path)
    
    def _generate_class_samples(self, target_class, target_count, dataset):
        """Генерация образцов для конкретного класса"""
        print(f"\n📊 Генерация класса {target_class}...")
        generated = 0
        attempts = 0
        max_attempts = target_count * 20
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            
            if attempts % 20 == 0:
                success_rate = (generated / attempts) * 100 if attempts > 0 else 0
                print(f"  Попытка {attempts}, сгенерировано {generated}/{target_count} (успех: {success_rate:.1f}%)")
            
            env = self._apply_class_bias(target_class, generate_environment())
            A_opt, b_opt, F_opt, result = find_global_optimum(env)
            
            if result.success:
                behavior_class, behavior_name = determine_behavior_class(
                    b_opt, env["grayzone_min"], env["grayzone_max"])
                
                if behavior_class == target_class:
                    self._add_sample_to_dataset(dataset, env, A_opt, b_opt, F_opt, behavior_class, behavior_name, result)
                    generated += 1
                elif attempts % 25 == 0:
                    # Диагностика: почему не получился нужный класс
                    print(f"    Получен класс {behavior_class} вместо {target_class}, b*={b_opt:.2f}")
                    print(f"    Параметры: sigma1={env['sigma1']:.2f}, sigma2={env['sigma2']:.2f}, beta={env['beta']:.2e}, gamma={env['gamma']:.2f}")
        
        print(f"  ✅ Класс {target_class}: {generated}/{target_count} (попыток: {attempts})")
    
    def _apply_class_bias(self, target_class, env):
        """Применяем смещение параметров для целевого класса, сохраняя оригинальные диапазоны"""
        biases = self.class_biases[target_class]
        
        # Оригинальные диапазоны
        original_ranges = {
            "sigma1": (0.25, 8.61),
            "sigma2": (0.003, 8.99), 
            "beta": (1e-9, 0.001),
            "gamma": (0.001, 1000)
        }
        
        # Применяем смещение в рамках оригинальных диапазонов
        for param, bias in biases.items():
            param_name = param.replace("_bias", "")
            low, high = original_ranges[param_name]
            
            if bias < 0.5:
                # Смещаем к нижней границе
                new_low = low
                new_high = low + (high - low) * (bias * 2)
            else:
                # Смещаем к верхней границе  
                new_low = low + (high - low) * ((bias - 0.5) * 2)
                new_high = high
            
            # Генерируем значение в смещенном диапазоне
            if param_name == "beta":  # логарифмический масштаб для beta
                log_low = np.log10(new_low)
                log_high = np.log10(new_high)
                env[param_name] = 10 ** np.random.uniform(log_low, log_high)
            else:
                env[param_name] = np.random.uniform(new_low, new_high)
        
        return env

    # Остальные методы остаются без изменений...
    def _initialize_dataset(self):
        return {
            'features': [], 'targets': [], 'behavior_names': [],
            'env_params': [], 'optimization_info': []
        }
    
    def _calculate_target_distribution(self, num_samples):
        samples_per_class = num_samples // 3
        remaining = num_samples % 3
        return {
            0: samples_per_class + (1 if remaining >= 1 else 0),
            0.5: samples_per_class + (1 if remaining >= 2 else 0),
            1: samples_per_class
        }
    
    def _add_sample_to_dataset(self, dataset, env, A_opt, b_opt, F_opt, behavior_class, behavior_name, result):
        features = [env["sigma1"], env["xi1"], env["sigma2"], env["xi2"],
                   env["xi3"], env["xi4"], env["eta1"], env["eta2"], env["c3"], env["c4"]]
        
        dataset['features'].append(features)
        dataset['targets'].append([A_opt, b_opt, F_opt, behavior_class])
        dataset['behavior_names'].append(behavior_name)
        dataset['env_params'].append(env)
        dataset['optimization_info'].append({
            'success': result.success, 'message': result.message,
            'nfev': result.nfev, 'nit': result.nit
        })
    
    def _finalize_dataset(self, dataset, save_path):
        if len(dataset['features']) > 0:
            dataset['features'] = np.array(dataset['features'])
            dataset['targets'] = np.array(dataset['targets'])
        else:
            print("⚠️ Внимание: не удалось сгенерировать ни одного образца!")
            dataset['features'] = np.array([])
            dataset['targets'] = np.array([])
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        csv_path = save_path.replace('.pkl', '.csv')
        save_dataset_csv(dataset, csv_path)
        
        print(f"\n🎉 Датсет сохранен:")
        print(f"   📁 Pickle: {save_path}")
        print(f"   📁 CSV: {csv_path}")
        
        return dataset

# Создаем глобальный экземпляр генератора
balanced_generator = BalancedDatasetGenerator()

def generate_balanced_dataset(num_samples=1000, save_path="migration_dataset_balanced.pkl"):
    """Улучшенная генерация сбалансированного датасета"""
    return balanced_generator.generate_balanced_dataset(num_samples, save_path)

# =============================================================================
# БЛОК 8: БАЗОВАЯ ГЕНЕРАЦИЯ ВЫБОРКИ
# =============================================================================

def generate_training_dataset(num_samples=1000, save_path="migration_dataset.pkl"):
    """
    Генерирует выборку данных для обучения нейронной сети
    Только параметры среды в признаках, коэффициенты организма исключены
    """

    dataset = {
        'features': [],      # Только параметры среды
        'targets': [],       # Целевые переменные [A_opt, b_opt, fitness, class]
        'behavior_names': [], 
        'env_params': [],    
        'optimization_info': []
    }

    print(f"Генерация {num_samples} образцов данных методом глобальной оптимизации...")
    print("В признаки включены ТОЛЬКО параметры среды (исключены a, gamma, beta, lambda_Q)")

    successful_optimizations = 0

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"Обработано {i + 1}/{num_samples} образцов")

        # Генерация среды
        env = generate_environment()

        # Глобальная оптимизация
        A_opt, b_opt, F_opt, result = find_global_optimum(env)

        if result.success:
            successful_optimizations += 1

        # Определение класса поведения
        behavior_class, behavior_name = determine_behavior_class(
            b_opt, env["grayzone_min"], env["grayzone_max"])

        # Формирование признаков (ТОЛЬКО параметры среды)
        features = [
            env["sigma1"], env["xi1"], env["sigma2"], env["xi2"],
            env["xi3"], env["xi4"], env["eta1"], env["eta2"],
            env["c3"], env["c4"]
        ]

        # Формирование целевых переменных
        targets = [A_opt, b_opt, F_opt, behavior_class]

        # Информация об оптимизации
        opt_info = {
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'nit': result.nit
        }

        # Сохранение данных
        dataset['features'].append(features)
        dataset['targets'].append(targets)
        dataset['behavior_names'].append(behavior_name)
        dataset['env_params'].append(env)
        dataset['optimization_info'].append(opt_info)

    # Преобразование в numpy массивы
    dataset['features'] = np.array(dataset['features'])
    dataset['targets'] = np.array(dataset['targets'])

    # Статистика оптимизации
    success_rate = successful_optimizations / num_samples * 100
    print(f"\nСтатистика глобальной оптимизации:")
    print(f"Успешных оптимизаций: {successful_optimizations}/{num_samples} ({success_rate:.1f}%)")

    # Сохраняем в pickle
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    # Сохраняем в CSV
    csv_path = save_path.replace('.pkl', '.csv')
    save_dataset_csv(dataset, csv_path)

    return dataset

# =============================================================================
# БЛОК 9: АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ДАННЫХ
# =============================================================================

def analyze_dataset(dataset):
    """
    Анализ сгенерированного датасета
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ДАТАСЕТА")
    print("="*60)

    features = dataset['features']
    targets = dataset['targets']
    behavior_names = dataset['behavior_names']
    optimization_info = dataset['optimization_info']

    print(f"Размер датасета: {len(features)} образцов")
    print(f"Размерность признаков: {features.shape}")
    print(f"Размерность целевых переменных: {targets.shape}")

    # Анализ успешности оптимизации
    success_count = sum(1 for info in optimization_info if info['success'])
    print(f"Успешных оптимизаций: {success_count}/{len(optimization_info)} ({success_count/len(optimization_info)*100:.1f}%)")

    # Анализ классов поведения
    behavior_classes = targets[:, 3]
    unique_classes, class_counts = np.unique(behavior_classes, return_counts=True)

    print(f"\nРаспределение классов поведения:")
    for class_val, count in zip(unique_classes, class_counts):
        class_samples = [name for i, name in enumerate(behavior_names)
                        if behavior_classes[i] == class_val]
        class_name = class_samples[0] if class_samples else "Неизвестно"
        percentage = count / len(behavior_classes) * 100
        print(f"  Класс {class_val} ({class_name}): {count} samples ({percentage:.1f}%)")

    print(f"\nСтатистика по оптимальным параметрам:")
    print(f"  A_optimal: mean={targets[:, 0].mean():.2f}, std={targets[:, 0].std():.2f}")
    print(f"  b_optimal: mean={targets[:, 1].mean():.2f}, std={targets[:, 1].std():.2f}")
    print(f"  fitness_optimal: mean={targets[:, 2].mean():.2f}, std={targets[:, 2].std():.2f}")

    # Статистика по итерациям оптимизации
    nfev_values = [info['nfev'] for info in optimization_info]
    nit_values = [info['nit'] for info in optimization_info]
    print(f"\nСтатистика оптимизации:")
    print(f"  Среднее количество оценок функции: {np.mean(nfev_values):.1f}")
    print(f"  Среднее количество итераций: {np.mean(nit_values):.1f}")

def plot_dataset_distribution(dataset):
    """
    Визуализация распределения данных в датасете
    """
    targets = dataset['targets']
    behavior_classes = targets[:, 3]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Распределение классов поведения
    unique_classes, class_counts = np.unique(behavior_classes, return_counts=True)
    colors = {0: 'red', 0.5: 'orange', 1: 'green'}
    bar_colors = [colors[cls] for cls in unique_classes]

    axes[0, 0].bar(unique_classes, class_counts, color=bar_colors, alpha=0.7, width=0.3)
    axes[0, 0].set_xlabel('Класс поведения')
    axes[0, 0].set_ylabel('Количество образцов')
    axes[0, 0].set_title('Распределение классов поведения\n(Глобальная оптимизация)')
    axes[0, 0].set_xticks([0, 0.5, 1])

    # 2. Распределение оптимальной амплитуды b_optimal
    axes[0, 1].hist(targets[:, 1], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(x=Config.GRAYZONE_MIN, color='red', linestyle='--', label='Граница серой зоны (min)')
    axes[0, 1].axvline(x=Config.GRAYZONE_MAX, color='red', linestyle='--', label='Граница серой зоны (max)')
    axes[0, 1].set_xlabel('Оптимальная амплитуда b*')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение оптимальной амплитуды миграции')
    axes[0, 1].legend()

    # 3. Распределение оптимальной глубины A_optimal
    axes[1, 0].hist(targets[:, 0], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Оптимальная глубина A*')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение оптимальной глубины')

    # 4. Распределение фитнеса
    axes[1, 1].hist(targets[:, 2], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Оптимальный фитнес F(A*, b*)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title('Распределение значений фитнеса')

    plt.tight_layout()
    plt.show()

def save_dataset_csv(dataset, csv_path="migration_dataset.csv"):
    """Сохраняет датасет в CSV формате для Excel"""
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')

        headers = [
            'sigma1', 'xi1', 'sigma2', 'xi2', 'xi3', 'xi4', 'eta1', 'eta2',
            'c3', 'c4', 'A_optimal', 'b_optimal', 'fitness_optimal', 'behavior_class', 'behavior_name'
        ]
        writer.writerow(headers)

        for i in range(len(dataset['features'])):
            row = (
                list(dataset['features'][i]) +
                list(dataset['targets'][i]) +
                [dataset['behavior_names'][i]]
            )
            writer.writerow(row)

    print(f"Датасет сохранен в CSV (для Excel): {csv_path}")
    print("Примечание: В признаки включены только параметры среды (10 параметров)")

# =============================================================================
# БЛОК 10: ДЕМОНСТРАЦИЯ ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ
# =============================================================================

def demonstrate_global_optimization():
    """
    Демонстрирует работу глобальной оптимизации на одном примере
    """
    print("ДЕМОНСТРАЦИЯ ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ")
    print("=" * 50)

    # Генерация тестовой среды
    env = generate_environment()

    print("Параметры среды:")
    for key, value in env.items():
        if key not in ['grayzone_min', 'grayzone_max']:
            print(f"  {key}: {value:.4f}")

    # Глобальная оптимизация
    print("\nЗапуск глобальной оптимизации...")
    A_opt, b_opt, F_opt, result = find_global_optimum(env)

    print(f"\nРезультаты глобальной оптимизации:")
    print(f"  A* = {A_opt:.4f}")
    print(f"  b* = {b_opt:.4f}")
    print(f"  F(A*, b*) = {F_opt:.4f}")
    print(f"  Успех: {result.success}")
    print(f"  Сообщение: {result.message}")
    print(f"  Количество итераций: {result.nit}")
    print(f"  Количество оценок функции: {result.nfev}")

    # Определение класса поведения
    behavior_class, behavior_name = determine_behavior_class(
        b_opt, env["grayzone_min"], env["grayzone_max"])

    print(f"  Класс поведения: {behavior_class} ({behavior_name})")

    return env, A_opt, b_opt, F_opt, result

# =============================================================================
# БЛОК 11: ОСНОВНАЯ ПРОГРАММА
# =============================================================================

def main():
    """
    Основная функция для генерации датасета
    """
    print("ГЕНЕРАЦИЯ ДАТАСЕТА С ГЛОБАЛЬНОЙ ОПТИМИЗАЦИЕЙ")
    print("=" * 60)
    print("ВАЖНО: В признаки включены ТОЛЬКО параметры среды (10 параметров)")
    print("Коэффициенты организма (a, gamma, beta, lambda_Q) исключены из признаков")

    # Демонстрация на одном примере
    demonstrate_global_optimization()

    print("\n" + "=" * 60)

    # Выбор типа генерации
    print("Выберите тип генерации:")
    print("1 - Балансированная генерация (рекомендуется)")
    print("2 - Базовая генерация")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        # Генерация сбалансированного датасета
        dataset = generate_balanced_dataset(
            num_samples=30,  # 10 на класс
            save_path="migration_dataset_balanced.pkl"
        )
    else:
        # Генерация базового датасета
        dataset = generate_training_dataset(
            num_samples=30,
            save_path="migration_dataset_base.pkl"
        )

    # Анализ датасета
    analyze_dataset(dataset)

    # Визуализация распределения
    plot_dataset_distribution(dataset)

    # Вывод информации о структуре данных
    print("\nСтруктура датасета:")
    print(f"Признаки (features): массив размерности {dataset['features'].shape}")
    print(f"  Столбцы: sigma1, xi1, sigma2, xi2, xi3, xi4, eta1, eta2, c3, c4")
    print(f"  ИСКЛЮЧЕНЫ: a, gamma, beta, lambda_Q")
    print(f"Целевые переменные (targets): массив размерности {dataset['targets'].shape}")
    print(f"  Столбцы: A_optimal, b_optimal, fitness_optimal, behavior_class")

if __name__ == "__main__":
    main()