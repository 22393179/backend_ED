import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Método de Euler Normal
def euler_normal(f, y0, x0, xn, h):
    x_vals = np.arange(x0, xn + h, h)
    y_vals = [y0]
    for i in range(1, len(x_vals)):
        y_next = y_vals[-1] + h * f(x_vals[i - 1], y_vals[-1])
        y_vals.append(y_next)
    return x_vals, y_vals


# Método de Euler Mejorado (Heun)
def euler_mejorado(f, y0, x0, xn, h):
    x_vals = np.arange(x0, xn + h, h)
    y_vals = [y0]
    for i in range(1, len(x_vals)):
        x = x_vals[i - 1]
        y = y_vals[-1]
        # Paso predictor (Euler normal)
        y_pred = y + h * f(x, y)
        # Paso corrector (promedio de pendientes)
        y_next = y + (h / 2) * (f(x, y) + f(x + h, y_pred))
        y_vals.append(y_next)
    return x_vals, y_vals


# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, y0, x0, xn, h):
    x_vals = np.arange(x0, xn + h, h)
    y_vals = [y0]
    for i in range(1, len(x_vals)):
        x = x_vals[i - 1]
        y = y_vals[-1]
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_vals.append(y_next)
    return x_vals, y_vals


# Solución exacta para comparar (personalizable)
def exact_solution(x):
    return -x - 1 + 2 * np.exp(x)  # Solución exacta para dy/dx = x + y con y(0) = 1


# Función para calcular y mostrar resultados en la interfaz
def calculate_and_display():
    try:
        # Obtener valores de entrada
        y0 = float(entry_y0.get())
        x0 = float(entry_x0.get())
        xn = float(entry_xn.get())
        h = float(entry_h.get())
        
        # Definir la ecuación diferencial dy/dx = f(x, y)
        def f(x, y):
            return x + y  # Ejemplo: dy/dx = x + y
        
        # Resolver con Euler Normal
        x_euler, y_euler = euler_normal(f, y0, x0, xn, h)
        y_exact_euler = [exact_solution(x) for x in x_euler]
        
        # Resolver con Euler Mejorado
        x_euler_mejorado, y_euler_mejorado = euler_mejorado(f, y0, x0, xn, h)
        y_exact_euler_mejorado = [exact_solution(x) for x in x_euler_mejorado]
        
        # Resolver con Runge-Kutta
        x_rk, y_rk = runge_kutta_4(f, y0, x0, xn, h)
        y_exact_rk = [exact_solution(x) for x in x_rk]
        
        # Crear tabla comparativa
        table_data = []
        for x, ye, yem, yrk, y_ex in zip(x_euler, y_euler, y_euler_mejorado, y_rk, y_exact_euler):
            error_euler = abs(ye - y_ex)
            error_euler_mejorado = abs(yem - y_ex)
            error_rk = abs(yrk - y_ex)
            table_data.append([x, ye, yem, yrk, y_ex, error_euler, error_euler_mejorado, error_rk])
        
        # Limpiar tabla anterior
        for row in tree.get_children():
            tree.delete(row)
        
        # Insertar datos en la tabla
        for row in table_data:
            tree.insert("", "end", values=row)
        
        # Generar gráfica comparativa
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(x_euler, y_euler, label="Euler Normal", marker="o", linestyle="--")
        ax.plot(x_euler_mejorado, y_euler_mejorado, label="Euler Mejorado", marker="s", linestyle="-.")
        ax.plot(x_rk, y_rk, label="Runge-Kutta 4", marker="x", linestyle="-.")
        ax.plot(x_euler, y_exact_euler, label="Solución Exacta", linestyle="-", color="black")
        ax.set_title("Comparación de Métodos Numéricos")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        
        # Actualizar el canvas de la gráfica
        canvas.draw()
    
    except ValueError:
        tk.messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos.")


# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Resolución de Ecuaciones Diferenciales")

# Entradas de usuario
frame_inputs = ttk.Frame(root, padding=10)
frame_inputs.pack(fill="x")

ttk.Label(frame_inputs, text="y0 (Valor inicial de y):").grid(row=0, column=0, sticky="w")
entry_y0 = ttk.Entry(frame_inputs)
entry_y0.grid(row=0, column=1)

ttk.Label(frame_inputs, text="x0 (Valor inicial de x):").grid(row=1, column=0, sticky="w")
entry_x0 = ttk.Entry(frame_inputs)
entry_x0.grid(row=1, column=1)

ttk.Label(frame_inputs, text="xn (Valor final de x):").grid(row=2, column=0, sticky="w")
entry_xn = ttk.Entry(frame_inputs)
entry_xn.grid(row=2, column=1)

ttk.Label(frame_inputs, text="h (Tamaño del paso):").grid(row=3, column=0, sticky="w")
entry_h = ttk.Entry(frame_inputs)
entry_h.grid(row=3, column=1)

# Botón para calcular
btn_calculate = ttk.Button(frame_inputs, text="Calcular", command=calculate_and_display)
btn_calculate.grid(row=4, column=0, columnspan=2, pady=10)

# Tabla de resultados
frame_table = ttk.Frame(root, padding=10)
frame_table.pack(fill="both", expand=True)

columns = ("x", "Euler", "Euler Mejorado", "Runge-Kutta", "Exacto", "Error Euler", "Error Euler Mejorado", "Error RK4")
tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=10)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor="center")
tree.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
scrollbar.pack(side="right", fill="y")
tree.configure(yscrollcommand=scrollbar.set)

# Gráfica
frame_graph = ttk.Frame(root, padding=10)
frame_graph.pack(fill="both", expand=True)

fig = plt.Figure(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# Agregar barra de herramientas para zoom, pan, etc.
toolbar = NavigationToolbar2Tk(canvas, frame_graph)
toolbar.update()
canvas_widget.pack(side="top", fill="both", expand=True)

# Ejecutar la aplicación
root.mainloop()