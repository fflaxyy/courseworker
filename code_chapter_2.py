# -*- coding: utf-8 -*-
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis, norm
DATA_FILE = 'NVDA_yfinance_clean.csv'
OUTPUT_DIR = Path('figures_chapter2')
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']
ALL_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
TRAIN_TEST_SPLIT_DATE = '2023-12-29'
DECOMPOSITION_PERIOD = 252

def setup_environment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='whitegrid')

def find_dataset_file() -> Path:
    path = Path(DATA_FILE)
    if path.exists():
        return path
    csv_files = list(Path('.').glob('*.csv'))
    if len(csv_files) == 1:
        print(f'Файл {DATA_FILE} не найден, но найден CSV: {csv_files[0]}')
        print('Использую его.')
        return csv_files[0]
    if len(csv_files) > 1:
        raise FileNotFoundError(f'Файл {DATA_FILE} не найден. В папке найдено несколько CSV: {[str(x) for x in csv_files]}. Укажи нужный файл в DATA_FILE.')
    raise FileNotFoundError(f'Файл {DATA_FILE} не найден. Положи CSV-файл в одну папку со скриптом.')

def load_nvda_data() -> pd.DataFrame:
    csv_path = find_dataset_file()
    print(f'Загружаю данные из файла: {csv_path}')
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'В датасете не хватает нужных столбцов: {missing}. Нужны столбцы: {required}')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').set_index('Date')
    for col in ALL_COLUMNS:
        df[col] = df[col].astype(str).str.replace(' ', '', regex=False).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[ALL_COLUMNS].dropna(how='all')
    print('\nРазмер датасета:', df.shape)
    print('Период:', df.index.min().date(), '—', df.index.max().date())
    print('\nПервые строки:')
    print(df.head())
    return df

def save_current_figure(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {path}')

def figure_2_1_multivariate_series(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    for ax, col in zip(axes, ALL_COLUMNS):
        if col == 'Volume':
            ax.plot(df.index, df[col] / 1000000, linewidth=1)
            ax.set_ylabel('млн шт.')
        else:
            ax.plot(df.index, df[col], linewidth=1)
            ax.set_ylabel('USD')
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Дата')
    fig.suptitle('Динамика дневных показателей акций NVIDIA', fontsize=14, y=0.995)
    save_current_figure('figure_2_1_nvda_multivariate_series.png')

def figure_2_2_close_train_test_split(df: pd.DataFrame) -> None:
    split_date = pd.to_datetime(TRAIN_TEST_SPLIT_DATE)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close', linewidth=1.5)
    plt.axvline(split_date, linestyle='--', label='Граница train/test')
    plt.title('Целевая переменная Close и граница обучающей/тестовой выборки')
    plt.xlabel('Дата')
    plt.ylabel('Цена закрытия, USD')
    plt.grid(True, alpha=0.3)
    plt.legend()
    train_count = int((df.index <= split_date).sum())
    test_count = int((df.index > split_date).sum())
    print(f'\nTrain наблюдений: {train_count}')
    print(f'Test наблюдений: {test_count}')
    print(f'Граница train/test: {split_date.date()}')
    save_current_figure('figure_2_2_close_train_test_split.png')

def figure_2_3_outlier_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    price_data = df[PRICE_COLUMNS].dropna()
    axes[0].boxplot([price_data[col] for col in PRICE_COLUMNS], tick_labels=PRICE_COLUMNS)
    axes[0].set_title('Цена')
    axes[0].set_ylabel('USD')
    axes[0].grid(True, alpha=0.3)
    volume_mln = df['Volume'].dropna() / 1000000
    axes[1].boxplot(volume_mln, tick_labels=['Volume'])
    axes[1].set_title('Объем торгов')
    axes[1].set_ylabel('млн шт.')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('Диаграммы размаха для выявления потенциальных выбросов', fontsize=14)
    save_current_figure('figure_2_3_outlier_boxplots.png')

def figure_2_4_range_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    data = [df[col].dropna() for col in ALL_COLUMNS]
    plt.boxplot(data, tick_labels=ALL_COLUMNS)
    plt.title('Сравнение диапазонов значений всех каналов на одной шкале')
    plt.ylabel('Исходные значения')
    plt.grid(True, alpha=0.3)
    save_current_figure('figure_2_4_raw_range_comparison.png')

def figure_2_5_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[ALL_COLUMNS].corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.3f', square=True, linewidths=0.5, cbar_kws={'label': 'Коэффициент корреляции'})
    plt.title('Матрица корреляции Пирсона')
    save_current_figure('figure_2_5_correlation_heatmap.png')

def figure_2_6_close_decomposition(df: pd.DataFrame) -> pd.Series:
    close = df['Close'].dropna()
    close_regular = close.asfreq('B').interpolate(method='time')
    if len(close_regular) < DECOMPOSITION_PERIOD * 2:
        raise ValueError(f'Для декомпозиции нужно хотя бы {DECOMPOSITION_PERIOD * 2} наблюдений. Сейчас: {len(close_regular)}')
    decomposition = seasonal_decompose(close_regular, model='additive', period=DECOMPOSITION_PERIOD, extrapolate_trend='freq')
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(close_regular.index, close_regular, linewidth=1)
    axes[0].set_title('Исходный ряд Close')
    axes[0].set_ylabel('USD')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(decomposition.trend.index, decomposition.trend, linewidth=1)
    axes[1].set_title('Тренд')
    axes[1].set_ylabel('USD')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, linewidth=1)
    axes[2].set_title('Сезонная компонента')
    axes[2].set_ylabel('USD')
    axes[2].grid(True, alpha=0.3)
    residuals = decomposition.resid.dropna()
    axes[3].plot(residuals.index, residuals, linewidth=1)
    axes[3].set_title('Остатки (шум)')
    axes[3].set_ylabel('USD')
    axes[3].set_xlabel('Дата')
    axes[3].grid(True, alpha=0.3)
    fig.suptitle(f'Аддитивная декомпозиция ряда Close, период {DECOMPOSITION_PERIOD} торговых дня', fontsize=14, y=0.995)
    save_current_figure('figure_2_6_close_decomposition.png')
    return residuals

def figure_2_7_residuals_histogram(residuals: pd.Series) -> None:
    residuals = residuals.dropna()
    mean_val = residuals.mean()
    std_val = residuals.std()
    skew_val = skew(residuals)
    kurt_val = kurtosis(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 300)
    y = norm.pdf(x, mean_val, std_val)
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, density=True, alpha=0.7, label='Остатки')
    plt.plot(x, y, linewidth=2, label='Нормальное распределение')
    plt.title('Распределение остатков после декомпозиции')
    plt.xlabel('Остаток, USD')
    plt.ylabel('Плотность')
    plt.grid(True, alpha=0.3)
    plt.legend()
    text = f'Среднее: {mean_val:.3f}\nСт. откл.: {std_val:.3f}\nАсимм.: {skew_val:.3f}\nЭксцесс: {kurt_val:.3f}'
    plt.text(0.98, 0.95, text, transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    print('\nСтатистика остатков после декомпозиции:')
    print(f'Среднее: {mean_val:.3f}')
    print(f'Стандартное отклонение: {std_val:.3f}')
    print(f'Асимметрия: {skew_val:.3f}')
    print(f'Эксцесс: {kurt_val:.3f}')
    save_current_figure('figure_2_7_residuals_histogram.png')

def main() -> None:
    setup_environment()
    df = load_nvda_data()
    figure_2_1_multivariate_series(df)
    figure_2_2_close_train_test_split(df)
    figure_2_3_outlier_boxplots(df)
    figure_2_4_range_comparison(df)
    figure_2_5_correlation_heatmap(df)
    residuals = figure_2_6_close_decomposition(df)
    figure_2_7_residuals_histogram(residuals)
    print('\nГотово! Все графики для 2 главы сохранены в папку:', OUTPUT_DIR.resolve())
if __name__ == '__main__':
    main()
