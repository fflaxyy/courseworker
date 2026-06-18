# -*- coding: utf-8 -*-
from pathlib import Path
import random
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
OUTPUT_DIR = Path('figures_chapter3_light')
SAMPLE_DIR = Path('sample_afhq_examples')
CLASS_NAMES = ['cat', 'dog', 'wildlife']
CLASS_COUNTS = {'cat': 5000, 'dog': 5000, 'wildlife': 5000}

def setup_environment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        (SAMPLE_DIR / class_name).mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

def save_current_figure(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {path}')

def wrap_text(text: str, width: int=45) -> str:
    return '\n'.join(textwrap.wrap(text, width=width))

def list_sample_images(class_name: str):
    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    folder = SAMPLE_DIR / class_name
    if not folder.exists():
        return []
    return [file for file in sorted(folder.iterdir()) if file.is_file() and file.suffix.lower() in valid_ext]

def figure_3_1_class_balance() -> None:
    labels = list(CLASS_COUNTS.keys())
    values = list(CLASS_COUNTS.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=11)
    plt.title('Баланс классов набора данных AFHQv2')
    plt.xlabel('Класс')
    plt.ylabel('Количество изображений')
    plt.ylim(0, max(values) * 1.15)
    plt.grid(True, axis='y', alpha=0.3)
    save_current_figure('figure_3_1_class_balance.png')

def figure_3_2_sample_images() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, class_name in zip(axes, CLASS_NAMES):
        sample_images = list_sample_images(class_name)
        if sample_images and PIL_AVAILABLE:
            img_path = random.choice(sample_images)
            with Image.open(img_path) as img:
                ax.imshow(img.convert('RGB'))
            ax.set_title(class_name)
            ax.axis('off')
        else:
            ax.set_title(class_name)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, wrap_text(f'Пример класса {class_name}\nПоложи изображение в папку:\n{SAMPLE_DIR / class_name}', width=25), ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Типичные изображения классов cat, dog и wildlife', fontsize=14)
    save_current_figure('figure_3_2_sample_images.png')

def figure_3_3_directory_structure() -> None:
    structure_text = 'afhq/\n├── train/\n│   ├── cat/\n│   ├── dog/\n│   └── wildlife/\n└── val/\n    ├── cat/\n    ├── dog/\n    └── wildlife/\n\nМетка класса определяется по имени родительской папки.\nДля задачи классификации отдельные bounding box или маски не требуются.'
    plt.figure(figsize=(8, 5))
    plt.axis('off')
    plt.text(0.02, 0.95, structure_text, ha='left', va='top', family='monospace', fontsize=12)
    plt.title('Структура каталогов датасета AFHQv2', pad=20)
    save_current_figure('figure_3_3_directory_structure.png')

def figure_3_5_quality_summary() -> None:
    criteria = ['разрешение\n512x512', 'формат\nPNG', 'разметка\nпо папкам', 'баланс\nклассов', 'отсутствие\nPII']
    values = [1, 1, 1, 1, 1]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(criteria, values)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, 1.02, 'OK', ha='center', va='bottom', fontsize=11)
    plt.title('Критерии качества датасета AFHQv2')
    plt.ylabel('Соответствие критерию')
    plt.ylim(0, 1.2)
    plt.yticks([0, 1], ['нет', 'да'])
    plt.grid(True, axis='y', alpha=0.3)
    save_current_figure('figure_3_5_quality_summary.png')

def main() -> None:
    setup_environment()
    print('\nЛегкая версия анализа 3 главы запущена.')
    print('Полный датасет AFHQv2 скачивать не требуется.')
    figure_3_1_class_balance()
    figure_3_2_sample_images()
    figure_3_3_directory_structure()
    figure_3_5_quality_summary()
    print('\nГотово! Файлы сохранены в папку:', OUTPUT_DIR.resolve())
    print('\nЕсли хочешь реальные примеры изображений на figure_3_2, положи картинки в:')
    for class_name in CLASS_NAMES:
        print(' -', SAMPLE_DIR / class_name)
if __name__ == '__main__':
    main()
