# AI Labs

Репозиторий содержит три учебные работы по компьютерному зрению. Основные эксперименты оформлены в ноутбуках `Lab1/lab1.ipynb`, `Lab2/lab2.ipynb` и `Lab3/lab3.ipynb`.

## Lab1. Simpsons Character Classification

- Датасет: [The Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset)
- Задача: классификация персонажей по изображениям, 42 класса, 20 933 изображения.
- Решение: transfer learning на `ResNet18` с предобученными весами ImageNet; замена последнего слоя под число классов; аугментации `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`; стратифицированное разбиение train/val/test; редкие классы оставлены в train; дисбаланс учтен через `CrossEntropyLoss` с весами классов.
- Метрики: `test_loss = 0.3154`, `test_acc = 0.9374`.

## Lab2. Massachusetts Roads Segmentation

- Датасет: [Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)
- Задача: бинарная сегментация дорог на спутниковых снимках.
- Решение: U-Net-подобная архитектура с блоками `DoubleConv`, skip connections и `ConvTranspose2d`; чтение GeoTIFF через `tifffile`; нарезка изображений на тайлы 512x512; train stride 512, inference stride 256 с усреднением перекрывающихся предсказаний; аугментации `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`; `BCEWithLogitsLoss`, `AdamW`, mixed precision.
- Метрики: `test_loss = 0.0556`, `test_iou = 0.6225`.

## Lab3. HRPlanes Detection

- Датасет: [HRPlanes v2](https://zenodo.org/records/14546832)
- Задача: детекция самолетов на спутниковых изображениях, 1 класс (`Plane`).
- Решение: `YOLOv8l` с предобученными весами COCO; `imgsz = 960`, `epochs = 30`, `batch = 16`, `AdamW`; подготовка `hrplanes.yaml`, split-файлов train/val/test и YOLO labels; использованы стандартные аугментации YOLO, включая HSV и mosaic.
- Метрики на test: `F1 = 0.9865`, `precision = 0.9890`, `recall = 0.9841`, `mAP50 = 0.9938`, `mAP50-95 = 0.8642`.

## Зависимости

Проект использует Python 3.13 и зависимости из `pyproject.toml`: PyTorch, torchvision, albumentations, scikit-learn, tifffile, ultralytics и вспомогательные библиотеки для анализа и визуализации.
