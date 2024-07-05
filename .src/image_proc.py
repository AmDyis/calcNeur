import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def get_bounding_box(image):
    image = np.array(image)  # Преобразование изображения в массив numpy
    rows = np.any(image, axis=1)  # Определение строк, содержащих хоть один ненулевой пиксель
    cols = np.any(image, axis=0)  # Определение столбцов, содержащих хоть один ненулевой пиксель
    ymin, ymax = np.where(rows)[0][[0, -1]]  # Находим минимальную и максимальную строки
    xmin, xmax = np.where(cols)[0][[0, -1]]  # Находим минимальный и максимальный столбцы
    return xmin, ymin, xmax, ymax  # Возвращаем координаты ограничивающего прямоугольника

def center_image(image):
    xmin, ymin, xmax, ymax = get_bounding_box(image)  # Получаем координаты ограничивающего прямоугольника
    image = image.crop((xmin, ymin, xmax, ymax))  # Обрезаем изображение по этим координатам
    width, height = image.size  # Получаем размеры обрезанного изображения
    max_dim = max(width, height)  # Определяем максимальный размер (ширина или высота)
    new_image = Image.new("L", (max_dim, max_dim), 0)  # Создаем новое квадратное изображение с черным фоном
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))  # Вставляем обрезанное изображение в центр нового
    return new_image  # Возвращаем центрированное изображение

augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(15),  # Поворот изображения на случайный угол в пределах [-15, 15] градусов
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Случайное масштабирование и обрезка изображения
    transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация
])

def preprocess_image(image):
    centered_image = center_image(image)  # Центрирование изображения
    
    # Преобразование изображения для PyTorch
    transform = transforms.Compose([
        transforms.Grayscale(),  # Преобразование изображения в оттенки серого
        transforms.Resize((28, 28)),  # Изменение размера изображения
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize((0.5,), (0.5,))  # Нормализация
    ])
    
    augmented_image = augmentation_transforms(centered_image)  # Применение аугментации к центрированному изображению
    return augmented_image   # Возвращаем окончательно предобработанное изображение
