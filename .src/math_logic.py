# math_logic.py

def update_math_expression(expression, symbol):
    # Проверка на допустимость символа (цифра или оператор)
    if symbol.isdigit():
        if expression and expression[-1].isdigit():
            # Объединяем текущую цифру с предыдущей
            expression[-1] += symbol
        else:
            expression.append(symbol)
    else:
        if expression and not expression[-1].isdigit():
            print(f"Неверный ввод: два оператора подряд '{expression[-1]}' и '{symbol}'")
        else:
            expression.append(symbol)

    # Попытка вычислить выражение
    try:
        result = eval("".join(expression))
    except Exception as e:
        result = None
        print(f"Ошибка при вычислении выражения: {e}")

    return expression, result
