import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class DifferentialEquationSolver:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Solver de Ecuaciones Diferenciales")
        self.root.geometry("1000x800")
        
        self.setup_ui()
        self.set_default_values()
        
    def setup_ui(self):
        """Configura la interfaz gráfica de usuario"""
        # Frame de controles
        control_frame = ttk.LabelFrame(self.root, text="Parámetros", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Campos de entrada
        ttk.Label(control_frame, text="Condición inicial x₀:").grid(row=0, column=0, sticky=tk.W)
        self.x0_entry = ttk.Entry(control_frame)
        self.x0_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Condición inicial y₀:").grid(row=1, column=0, sticky=tk.W)
        self.y0_entry = ttk.Entry(control_frame)
        self.y0_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Tamaño de paso (h):").grid(row=2, column=0, sticky=tk.W)
        self.h_entry = ttk.Entry(control_frame)
        self.h_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Número de pasos (n):").grid(row=3, column=0, sticky=tk.W)
        self.n_entry = ttk.Entry(control_frame)
        self.n_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Botón de cálculo
        self.calculate_btn = ttk.Button(control_frame, text="Calcular", command=self.solve)
        self.calculate_btn.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Frame de resultados
        result_frame = ttk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Notebook (pestañas) para resultados
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de tabla
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Tabla de Resultados")
        
        # Pestaña de gráfico
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Gráfico Comparativo")
        
    def set_default_values(self):
        """Establece valores por defecto"""
        self.x0_entry.insert(0, "0.0")
        self.y0_entry.insert(0, "1.0")
        self.h_entry.insert(0, "0.1")
        self.n_entry.insert(0, "10")
    
    def equation(self, x, y):
        """Ecuación diferencial: dy/dx = x - y"""
        return x - y
    
    def exact_solution(self, x, x0, y0):
        """Solución exacta para dy/dx = x - y con condición inicial y(x0) = y0"""
        return (x0 - y0 - 1) * np.exp(-(x - x0)) + x - 1
    
    def euler_method(self, f, x0, y0, h, n):
        """Método de Euler para resolver EDOs"""
        x, y = x0, y0
        xs, ys = [x], [y]
        for _ in range(n):
            y += h * f(x, y)
            x += h
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    def runge_kutta_method(self, f, x0, y0, h, n):
        """Método de Runge-Kutta (RK4) para resolver EDOs"""
        x, y = x0, y0
        xs, ys = [x], [y]
        for _ in range(n):
            k1 = h * f(x, y)
            k2 = h * f(x + h/2, y + k1/2)
            k3 = h * f(x + h/2, y + k2/2)
            k4 = h * f(x + h, y + k3)
            y += (k1 + 2*k2 + 2*k3 + k4) / 6
            x += h
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    def solve(self):
        """Resuelve la ecuación y muestra los resultados"""
        try:
            # Obtener parámetros de la interfaz
            x0 = float(self.x0_entry.get())
            y0 = float(self.y0_entry.get())
            h = float(self.h_entry.get())
            n = int(self.n_entry.get())
            
            # Validar entradas
            if h <= 0:
                raise ValueError("El tamaño de paso (h) debe ser positivo")
            if n <= 0:
                raise ValueError("El número de pasos (n) debe ser positivo")
            
            # Resolver con diferentes métodos
            xs_euler, ys_euler = self.euler_method(self.equation, x0, y0, h, n)
            xs_rk, ys_rk = self.runge_kutta_method(self.equation, x0, y0, h, n)
            
            # Solución exacta (solo si es posible)
            xs_exact = np.linspace(x0, x0 + n*h, 100)
            ys_exact = self.exact_solution(xs_exact, x0, y0)
            
            # Mostrar resultados
            self.show_results_table(xs_euler, ys_euler, ys_rk, ys_exact)
            self.show_plot(xs_euler, ys_euler, xs_rk, ys_rk, xs_exact, ys_exact)
            
        except ValueError as e:
            tk.messagebox.showerror("Error de entrada", f"Entrada inválida: {str(e)}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
    
    def show_results_table(self, xs, ys_euler, ys_rk, ys_exact):
        """Muestra los resultados en una tabla"""
        # Limpiar pestaña anterior
        for widget in self.table_tab.winfo_children():
            widget.destroy()
        
        # Crear Treeview (tabla)
        tree = ttk.Treeview(self.table_tab, columns=("x", "Euler", "RK4", "Exacta"), show="headings")
        
        # Configurar columnas
        tree.heading("x", text="x")
        tree.heading("Euler", text="Método de Euler")
        tree.heading("RK4", text="Método RK4")
        tree.heading("Exacta", text="Solución Exacta")
        
        tree.column("x", width=100, anchor=tk.CENTER)
        tree.column("Euler", width=150, anchor=tk.CENTER)
        tree.column("RK4", width=150, anchor=tk.CENTER)
        tree.column("Exacta", width=150, anchor=tk.CENTER)
        
        # Añadir scrollbar
        scrollbar = ttk.Scrollbar(self.table_tab, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Insertar datos
        for i, x in enumerate(xs):
            exact_val = self.exact_solution(x, float(self.x0_entry.get()), float(self.y0_entry.get()))
            tree.insert("", tk.END, values=(
                f"{x:.4f}",
                f"{ys_euler[i]:.6f}",
                f"{ys_rk[i]:.6f}" if i < len(ys_rk) else "",
                f"{exact_val:.6f}"
            ))
    
    def show_plot(self, xs_euler, ys_euler, xs_rk, ys_rk, xs_exact, ys_exact):
        """Muestra el gráfico comparativo"""
        # Limpiar pestaña anterior
        for widget in self.plot_tab.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Graficar resultados
        ax.plot(xs_euler, ys_euler, 'bo-', label="Método de Euler", markersize=4)
        ax.plot(xs_rk, ys_rk, 'gx-', label="Método RK4", markersize=6)
        ax.plot(xs_exact, ys_exact, 'r-', label="Solución Exacta", linewidth=2)
        
        # Configurar gráfico
        ax.set_title("Comparación de Métodos Numéricos")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        
        # Añadir figura a la interfaz
        canvas = FigureCanvasTkAgg(fig, master=self.plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Añadir barra de herramientas
        toolbar = NavigationToolbar2Tk(canvas, self.plot_tab)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()

if __name__ == "__main__":
    app = DifferentialEquationSolver()
    app.run()