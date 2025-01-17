# Проект сегментации портретов

Данный проект направлен на создание модели для сегментации людей на изображениях с использованием набора данных [EasyPortrait: Face Parsing & Portrait Segmentation](https://www.kaggle.com/datasets/kapitanov/easyportrait). Он использует PyTorch, Albumentations и библиотеку моделей сегментации PyTorch (SMP) для разработки и обучения моделей. На основе обученной моделе был создан Telegram бот [NickRoot_Bot](https://t.me/NickRoot_Bot).

## Содержание

1. [Структура проекта](#структура-проекта)
2. [Структура датасета](#структура-датасета)
3. [Зависимости](#зависимости)
4. [Настройка и установка](#настройка-и-установка)
5. [Обзор модели](#обзор-модели)
6. [Обучающий Pipeline](#обучающий-Pipeline)
7. [Визуализация](#визуализация)
8. [Тестирование](#тестирование)
9. [Telegram бот для удаления фона](#telegram-бот-для-удаления-фона)
10. [Функции](#функции)
11. [Пример взаимодействия с ботом](#пример-взаимодействия-с-ботом)
12. [Установка](#установка)
13. [Использование](#использование)
14. [Обзор кода](#обзор-кода)
15. [Зависимости для Telegram бота](#зависимости-для-telegram-бота)
16. [Визуализация работы бота](#визуализация-работы-бота)
17. [License](#license)

---

## Структура проекта

```
/background_delete_bot
|-- examples/
|   |-- example.jpg #Скриншот работы бота
|   |-- test_result.png #Предсказание модели на тестовой выборке
|   |-- visualization.png #Скриншот работы визуализации
|
|-- model_training/
|   |-- background_delete_model.ipynb #Ноутбук для обучения модели
|   |-- deeplabplus_mobile0nes4_epoch10_binary.pth #Сохраненная модель
|   |-- requirements.txt
|
|-- telegram_bot/
    |-- bot.py #Код Telegram бота
    |-- requirements.txt
```

## Структура датасета

Набор данных состоит из трех основных частей:

```
/kaggle/input/easyportrait
|-- images/
|   |-- train/
|   |-- val/
|   |-- test/
|
|-- annotations/
    |-- train/
    |-- val/
    |-- test/
```

- **images/**: Содержит обучающие, валидационные и тестовые изображения.
- **annotations/**: Содержит маски для каждого изображения, где пиксели обозначают:
  - "0" — фон,
  - "1" — человек,
  - "2" — кожа,
  - "3" — левая бровь,
  - "4" — правая бровь,
  - "5" — левый глаз,
  - "6" — правый глаз,
  - "7" — губы,
  - "8" — зубы.

---

## Зависимости

Перед запуском проекта установите следующие зависимости:

- Python 3.8+
- PyTorch
- Albumentations
- Segmentation Models PyTorch (SMP)
- OpenCV
- NumPy
- Matplotlib
- tqdm

Установите необходимые библиотеки с помощью pip:

```bash
pip install -r requirements.txt
```

---

## Настройка и установка

1. Клонируйте или скачайте данный проект.
2. ```cd model_training```
3. ```pip install -r requirements.txt```
4. Установите переменную 'DATA_DIR' в корневую директорию набора данных [EasyPortrait: Face Parsing & Portrait Segmentation](https://www.kaggle.com/datasets/kapitanov/easyportrait).

---

## Обзор модели

### Архитектура

Модель сегментации строится с помощью SMP:
- **Decoder:** DeepLabV3+
- **Encoder:** MobileOne (предобученная на ImageNet)

### Функция ошибки

Используется комбинированная функция потерь, которая включает в себя:
- Dice Loss
- Cross Entropy Loss

### Оптимизатор

Используется оптимизатор AdamW с параметрами:
- learning rate: `1e-4`
- weight decay: `1e-5`

---

## Обучающий Pipeline

### Аугментация данных

Библиотека albumentations используется для аугментаций:
- **Тренировочные аугментации**: Сдвиг, масштабирование, вращение, добавление шума, регулировку яркости/контрастности и изменение размера до 512x512.
- **Проверочные аугментации**: Изменение размера до 512x512.

### DataLoader

Используются разные dataloader для обучения, валидации и проверки. Размера батча 4.

### Функция обучения

Функция обучения обрабатывает датасет партиями:
- Обучение с использованием смешанной точности реализовано с помощью torch.cuda.amp для повышения эффективности.
- Метрики: IoU score, Dice score.

---

## Визуализация

Функция '`visualize_seg_mask` используется для:
1. Отображение оригинального изображения.
2. Отображение предсказанной маски.
3. Наложения результатов сегментации на оригинальное изображение.
4. Результат удаления фона.

![Скриншот работы визуализации](examples/visualization.png)

---

## Тестирование

![Предсказание модели на тестовом датасете](examples/test_result.png)

- Test_loss:  0.0367
- Test_IoU:  0.986

---

## Telegram бот для удаления фона

Здесь реализован Telegram-бот, который удаляет фон с изображений с помощью предварительно обученной модели. Бот также позволяет пользователям заменять фон на пользовательское изображение или зеленый фон по умолчанию.

Вы можете протестировать бота, перейдя по следующей ссылке: [NickRoot_Bot](https://t.me/NickRoot_Bot).

## Функции

- **Удаление фона**: Удаляет фон с загруженного изображения.
- **Пользовательский фон**: Пользователи могут загрузить собственное изображение фона.
- **Стандартный зеленый фон**: Если пользователь не загрузил собственный фон, бот заменяет фон на зеленый экран.
- **Удобное взаимодействие**: Бот направляет пользователей через процесс с помощью простых инструкций.

---

## Пример взаимодействия с ботом

1. Пользователь отправляет изображение боту.
2. Бот обрабатывает изображение с помощью предварительно обученной модели для сегментации фона.
3. Пользователь может выбрать загрузку пользовательского фона или использовать стандартный зеленый фон.
4. Бот генерирует итоговое изображение и отправляет его обратно пользователю.

---

## Установка

1. Клонируй этот репозиторий
2. ```cd telegram_bot```
3. ```pip install -r requirements.txt```
4. Настройте токен Telegram-бота: Замените переменную `TELEGRAM_TOKEN` в `bot.py` на ваш токен бота, полученный от [BotFather](https://core.telegram.org/bots#botfather).
5. Разместите файл предварительно обученной модели: Убедитесь, что файл модели (`deeplabplus_mobile0nes4_epoch10_binary.pth`) находится в соответствующей директории, указанной в коде.

---

## Использование

1. Запусти бота:

   ```bash
   python bot.py
   ```

2. Взаимодействуйте с ботом в Telegram:

   - Отправь команду `/start`, чтобы начать.
   - Загрузите изображение для удаления фона.
   - Ответьте "да", чтобы загрузить пользовательский фон, или "нет", чтобы использовать стандартный зеленый фон.

---

## Обзор кода

- **Основные функции**:
  - `process_image`: Обрабатывает загруженное изображение для удаления фона и замены его на выбранный фон.
  - `create_green_background`: Генерирует стандартный зеленый фон.
  - `resize_or_crop_background`: Изменяет размер или обрезает пользовательский фон, чтобы он соответствовал размеру входного изображения.
  
- **Обработчики**:
  - `/start`: Инициализирует бота и предоставляет инструкции.
  - `handle_photo`: Обрабатывает загруженное изображение и выполняет его обработку.
  - `handle_text`: Интерпретирует ответы пользователя для определения использования пользовательского или стандартного фона.

## Зависимости для Telegram бота

- `torch`: Для загрузки и запуска предварительно обученной модели DeepLab.
- `cv2`: Для обработки изображений.
- `numpy`: Для числовых операций с массивами изображений.
- `Pillow`: Для работы с форматами изображений.
- `albumentations`: Для аугментации данных и преобразований.
- `python-telegram-bot`: Для интеграции с API Telegram-бота.

---

## Визуализация работы бота

![Скриншот работы бота](examples/example.jpg)

---

## License

This project is open-source.

---
