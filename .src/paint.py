import tkinter as tk
from tkinter import colorchooser, simpledialog # модули для цвета и ввода данных
from PIL import Image, ImageDraw
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ПсевдоПейнт")

         # Получение размеров экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Установка размеров окна и холста
        self.canvas_width = screen_width - 100  # Немного меньше размера экрана
        self.canvas_height = screen_height - 100  # Немного меньше размера экрана

        self.brush_size = 10
        self.brush_color = "blue"

        # Создание холста
        self.canvas = tk.Canvas(self.root, bg = "white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True) # Ставим холст в окне

        # Привязываем рисование с зажатию лкм
                # Привязываем события к методам
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.set_previous_coords)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.previous_x, self.previous_y = None, None  # Инициализируем предыдущие координаты
        self.action = []

        self.setup_menu()

         # Бинды для горячих клавиш
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-s>", self.save_image)
        self.root.bind("<Control-c>", self.clear_canvas)

        # Загрузка предобученной модели ResNet18
        self.model = models.resnet18(pretrained=False)
        self.num_ftrs = self.model.fc.in_features

        # Изменение первого свёрточного слоя для работы с одноканальными изображениями
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Изменение последнего полностью связанного слоя для 17 классов
        self.model.fc = nn.Linear(self.num_ftrs, 17)

        model_path = 'F:\pet2\pythonProject4\mode2l.pth'
        # Загрузка сохранённых весов
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Перевод модели в режим оценки

        # Преобразования изображения
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Преобразование изображения в оттенки серого (если нужно)
            transforms.Resize((28, 28)),  # Изменение размера изображения (подстраивайте под вашу модель)
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,))  # Нормализация (подстраивайте под вашу модель)
        ])

    # Функция настройки меню
    def setup_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu) # Установка меню в главное окно

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear_canvas)  # Добавляем команду для очистки холста
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Recognize", command=self.recognize)  # Добавляем команду для распознавания изображения

        # Создаем подменю Brush
        brush_menu = tk.Menu(menu)
        menu.add_cascade(label="Brush", menu=brush_menu)
        brush_menu.add_command(label="Brush Size", command=self.choose_brush_size)  # Добавляем команду для выбора размера кисти
        brush_menu.add_command(label="Brush Color", command=self.choose_brush_color)  # Добавляем команду для выбора цвета кисти
    
    def choose_brush_size(self):
        # Запрашиваем у пользователя ввод размера кисти
        size = simpledialog.askinteger("Brush Size", "Enter brush size:", initialvalue=self.brush_size)
        if size:
            self.brush_size = size  # Устанавливаем новый размер кисти

    def choose_brush_color(self):
        # Запрашиваем у пользователя выбор цвета кисти
        color = colorchooser.askcolor(color=self.brush_color)[1]
        if color:
            self.brush_color = color  # Устанавливаем новый цвет кисти

    def set_previous_coords(self, event):
        self.previous_x, self.previous_y = event.x, event.y

    def paint(self, event):
        if self.previous_x and self.previous_y:
            self.canvas.create_line(self.previous_x, self.previous_y, event.x, event.y,
                                    fill=self.brush_color, width=self.brush_size)
            self.draw.line([self.previous_x, self.previous_y, event.x, event.y],
                           fill=self.brush_color, width=self.brush_size)
        self.previous_x, self.previous_y = event.x, event.y

    

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.previous_x, self.previous_y = None, None
        self.actions = []  # Очищаем стек действий

    def undo(self, event=None):
        if self.actions:
            last_action = self.actions.pop()
            self.canvas.delete(last_action)
            # Обновляем изображение
            self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
            self.draw = ImageDraw.Draw(self.image)
            # Перерисовываем все действия
            for action in self.actions:
                coords = self.canvas.coords(action)
                self.draw.line(coords, fill=self.brush_color, width=self.brush_size)
    
    def redraw_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        # Перерисовываем все действия из стека
        for line_coords in self.actions:
            self.canvas.create_line(*line_coords, fill=self.brush_color, width=self.brush_size)
            self.draw.line(line_coords, fill=self.brush_color, width=self.brush_size)

    
    def save_image(self, event=None):
        self.image.save("output.png")
        print("Изображение сохранено как output.png")

    def recognize(self):
        # Трансформации для изображений
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), #преобразование в оттенки серого
            transforms.Resize((28, 28)), #28х28 пикселей
            transforms.ToTensor(), #перевод в тензоры
            transforms.Normalize((0.5,), (0.5,)) #нормализовка изображения
        ])
        train_dataset = datasets.ImageFolder('final_symbols_split_ttv/train', transform=transform)
        class_labels = train_dataset.classes

        self.save_image()  # Сначала сохраняем текущее изображение
        image = Image.open("output.png")
        print("Исходное изображение:", image.size)  # Отладочный вывод размера изображения
        image = self.transform(image).unsqueeze(0)  # Преобразуем изображение для модели
        print("Преобразованное изображение:", image.shape)  # Отладочный вывод формы тензора
        output = self.model(image)
        print("Выход модели:", output)  # Отладочный вывод выхода модели
        _, predicted = torch.max(output, 1)
        print("Распознанное выражение:", predicted.item(), class_labels[predicted.item()])


def run_paint_app():
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_paint_app()
