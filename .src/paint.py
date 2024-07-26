import tkinter as tk
from tkinter import colorchooser, simpledialog # модули для цвета и ввода данных
from PIL import Image, ImageDraw
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from math_logic import update_math_expression

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ПсевдоПейнт")

         # Получение размеров экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Установка размеров окна и холста
        self.canvas_width = 300
        self.canvas_height = 300

        self.brush_size = 10
        self.brush_color = "blue"

        # Создание холста
        self.canvas = tk.Canvas(self.root, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Привязываем рисование с зажатию лкм
                # Привязываем события к методам
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.set_previous_coords)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.previous_x, self.previous_y = None, None
        self.expression = []
        self.line_points = []  # Для хранения точек линии

        self.setup_menu()

         # Бинды для горячих клавиш
        self.root.bind("<Control-s>", self.save_image)
        self.root.bind("<Control-c>", self.clear_canvas)
        self.root.bind("<Control-a>", self.recognize)
        self.root.bind("<Control-z>", self.undo_last_symbol)

        # Добавление виджета Text для вывода сообщений справа от холста
        self.text_output = tk.Text(self.root, height=20, width=30, state=tk.DISABLED)
        self.text_output.grid(row=0, column=1, padx=10, pady=10, sticky='ns')


        # Загрузка предобученной модели ResNet18
        self.model = models.resnet18(pretrained=False)
        self.num_ftrs = self.model.fc.in_features

        # Изменение первого свёрточного слоя для работы с одноканальными изображениями
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Изменение последнего полностью связанного слоя для 17 классов
        self.model.fc = nn.Linear(self.num_ftrs, 17)

        model_path = '.src\mode2l.pth'
        # Загрузка сохранённых весов
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Перевод модели в режим оценки

        # Преобразования изображения
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Преобразование изображения в оттенки серого (если нужно)
            transforms.Resize((28, 14)),  # Изменение размера изображения (подстраивайте под вашу модель)
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,))  # Нормализация (подстраивайте под вашу модель)
        ])

        # Словарь для отображения символов
        self.symbol_map = {
            '=': '=',
            'add': '+',
            'divide': '/',
            'eight': '8',
            'five': '5',
            'four': '4',
            'gt': '>',
            'lt': '<',
            'multiply': '*',
            'nine': '9',
            'one': '1',
            'seven': '7',
            'six': '6',
            'subtract': '-',
            'three': '3',
            'two': '2',
            'zero': '0'
        }

    # Функция настройки меню
    def setup_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu) # Установка меню в главное окно

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear_canvas)  # Добавляем команду для очистки холста
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Recognize", command=self.recognize)  # Добавляем команду для распознавания изображения
        file_menu.add_command(label="Undo", command=self.undo_last_symbol)


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
        self.line_points = [(self.previous_x, self.previous_y)]

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
    
    def save_image(self, event=None):
        self.image.save("output.png")
        print("Изображение сохранено как output.png")

    def recognize(self, event=None):
        self.save_image()
        image = Image.open("output.png")
        symbols = ['=', 'add', 'divide', 'eight', 'five', 'four', 'gt', 'lt', 'multiply', 'nine', 'one', 'seven', 'six', 'subtract', 'three', 'two', 'zero']
        print("Исходное изображение: " + str(image.size))
        image = self.transform(image).unsqueeze(0)
        print("Преобразованное изображение: " + str(image.shape))
        output = self.model(image)
        print("Выход модели: " + str(output))
        _, predicted = torch.max(output, 1)
        recognized_symbol = symbols[predicted.item()]
        print("Распознанное выражение: " + str(predicted.item()) + " " + symbols[predicted.item()])

        self.expression, result = update_math_expression(self.expression, self.symbol_map[recognized_symbol])
        self.clear_canvas()
        self.update_expression()


        if result is not None:
            self.log_to_text_output("Результат вычисления: " + str(result))

    def update_expression(self):
        expression_str = "".join(self.expression)
        self.log_to_text_output("Текущее выражение: " + expression_str)

    def log_to_text_output(self, message):
        if hasattr(self, 'text_output'):
            self.text_output.config(state=tk.NORMAL)
            self.text_output.insert(tk.END, message + "\n")
            self.text_output.config(state=tk.DISABLED)
            self.text_output.yview(tk.END)

    def clear_text_output(self):
        if hasattr(self, 'text_output'):
            self.text_output.config(state=tk.NORMAL)
            self.text_output.delete(1.0, tk.END)
            self.text_output.config(state=tk.DISABLED)

    def undo_last_symbol(self, event=None):
        if self.expression:
            self.expression.pop()
            self.log_to_text_output("Последний символ удален. Текущее выражение: " + "".join(self.expression))
        else:
            self.log_to_text_output("Нет символов для удаления.")


def run_paint_app():
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_paint_app()
