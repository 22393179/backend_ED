import numpy as np
import matplotlib.pyplot as plt

# Función para ecuaciones diferenciales
def f(x, y):
    return x - y  # Ejemplo: dy/dx = x - y

# Método de Euler
def euler(f, x0, y0, h, n):
    x, y = x0, y0
    xs, ys = [x], [y]
    for _ in range(n):
        y += h * f(x, y)
        x += h
        xs.append(x)
        ys.append(y)
    return xs, ys

# Método de Runge-Kutta (RK4)
def runge_kutta(f, x0, y0, h, n):
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

# Funciones para Newton-Raphson
def F(xy):
    x, y = xy
    return np.array([
        x**2 + y**2 - 4,   # Ejemplo: x² + y² = 4
        x * y - 1          # Ejemplo: x * y = 1
    ])

def J(xy):
    x, y = xy
    return np.array([
        [2*x, 2*y],  # Derivadas parciales de F1
        [y, x]       # Derivadas parciales de F2
    ])

# Método de Newton-Raphson
def newton_raphson(F, J, xy0, tol=1e-6, max_iter=20):
    xy = np.array(xy0, dtype=float)
    for _ in range(max_iter):
        delta = np.linalg.solve(J(xy), -F(xy))
        xy += delta
        if np.linalg.norm(delta) < tol:
            return xy
    raise ValueError("No convergió")

# Menú para elegir qué método ejecutar
def main():
    print("Seleccione una opción:")
    print("1. Resolver ecuación diferencial (Euler y Runge-Kutta)")
    print("2. Resolver sistema de ecuaciones (Newton-Raphson)")
    
    opcion = input("Ingrese 1 o 2: ")
    
    if opcion == "1":
        x0, y0, h, n = 0, 1, 0.1, 20  # Parámetros iniciales
        xe, ye = euler(f, x0, y0, h, n)
        xr, yr = runge_kutta(f, x0, y0, h, n)

        # Graficar resultados
        plt.plot(xe, ye, 'r--', label="Euler")
        plt.plot(xr, yr, 'b-', label="Runge-Kutta")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Solución de dy/dx = x - y")
        plt.show()
    
    elif opcion == "2":
        sol = newton_raphson(F, J, [1.0, 1.0])
        print(f"Solución: x = {sol[0]:.6f}, y = {sol[1]:.6f}")
    
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    main()
