
# Matrices con Python - Ejercicios ROBOTICA U2
#Erick Hernandez Calderon

# Importamos librerías necesarias
import numpy as np        # Para operaciones numéricas y algebra lineal
from sympy import Matrix  # Para trabajar con matrices simbólicas (útil para espacio nulo)

# -------------------------
# Ejercicio 1: Suma de matrices
# -------------------------
A1 = np.array([[2, 4, 6], [1, 3, 5], [7, 9, 11]])  # Definimos la matriz A
B1 = np.array([[12, 10, 8], [6, 4, 2], [0, -2, -4]])  # Definimos la matriz B
C1 = A1 + B1   # Sumamos ambas matrices
print("Ejercicio 1 - Suma de matrices:\n", C1, "\n")  # Imprimimos el resultado

# -------------------------
# Ejercicio 2: Multiplicación de matrices
# -------------------------
A2 = np.array([[2, 1], [3, 4], [5, 6]])  # Matriz A
B2 = np.array([[7, 8], [9, 10]])         # Matriz B
C2 = np.dot(A2, B2)  # Multiplicación matricial A*B
print("Ejercicio 2 - Multiplicación de matrices:\n", C2, "\n")

# -------------------------
# Ejercicio 3: Inversión de matriz
# -------------------------
A3 = np.array([[4, 7, 2], [2, 6, 8], [3, 1, 9]])  # Matriz cuadrada
A3_inv = np.linalg.inv(A3)  # Calculamos la inversa de A
print("Ejercicio 3 - Inversa de matriz:\n", A3_inv, "\n")

# -------------------------
# Ejercicio 4: Resolución de sistema de ecuaciones lineales
# -------------------------
# Sistema:
# 2x + y + z = 8
# 3x + 5y + 2z = 21
# x + 2y + 4z = 11

A4 = np.array([[2, 1, 1], [3, 5, 2], [1, 2, 4]])  # Matriz de coeficientes A
B4 = np.array([8, 21, 11])  # Vector de términos independientes B
X4 = np.linalg.solve(A4, B4)  # Resolvemos AX = B
print("Ejercicio 4 - Solución del sistema [x, y, z]:\n", X4, "\n")

# -------------------------
# Ejercicio 5: Cálculo de determinante
# -------------------------
A5 = np.array([[3, -2, 1], [0, 5, 4], [2, 1, 7]])  # Definimos matriz A
det_A5 = np.linalg.det(A5)  # Calculamos determinante
print("Ejercicio 5 - Determinante:\n", det_A5, "\n")

# -------------------------
# Ejercicio 6: Producto cruz de vectores
# -------------------------
A6 = np.array([2, 3, -1])  # Vector A
B6 = np.array([1, -2, 4])  # Vector B
cross_A6_B6 = np.cross(A6, B6)  # Producto cruz A x B
print("Ejercicio 6 - Producto cruz:\n", cross_A6_B6, "\n")

# -------------------------
# Ejercicio 7: Proyección ortogonal
# -------------------------
V7 = np.array([5, -3, 2])  # Vector V
U7 = np.array([2, 1, 2])   # Vector U
# Fórmula: Proj(V sobre U) = (V·U / U·U) * U
proj_V7_U7 = (np.dot(V7, U7) / np.dot(U7, U7)) * U7
print("Ejercicio 7 - Proyección ortogonal de V sobre U:\n", proj_V7_U7, "\n")

# -------------------------
# Ejercicio 8: Producto escalar de proyecciones
# -------------------------
V8 = np.array([3, -1, 2])   # Vector V
U8 = np.array([2, 2, -1])   # Vector U
W8 = np.array([1, 4, -2])   # Vector W

# Proyección de V sobre U
proj_V8_U8 = (np.dot(V8, U8) / np.dot(U8, U8)) * U8
# Proyección de V sobre W
proj_V8_W8 = (np.dot(V8, W8) / np.dot(W8, W8)) * W8
# Producto escalar entre ambas proyecciones
dot_proj = np.dot(proj_V8_U8, proj_V8_W8)
print("Ejercicio 8 - Producto escalar de proyecciones:\n", dot_proj, "\n")

# -------------------------
# Ejercicio 9: Ortogonalización de Gram-Schmidt
# -------------------------
# Definimos tres vectores linealmente independientes
v1 = np.array([1, 1, 0], dtype=float)
v2 = np.array([1, 2, 1], dtype=float)
v3 = np.array([2, 1, 3], dtype=float)

# Función para aplicar el proceso de Gram-Schmidt
def gram_schmidt(vectors):
    ortho = []  # Lista de vectores ortogonales
    for v in vectors:
        w = v.copy()  # Copiamos el vector
        for u in ortho:  # Restamos la proyección sobre los vectores anteriores
            w -= np.dot(v, u) / np.dot(u, u) * u
        ortho.append(w)  # Agregamos el nuevo vector ortogonal
    return ortho

ortho_vectors = gram_schmidt([v1, v2, v3])  # Aplicamos el proceso
print("Ejercicio 9 - Vectores ortogonales (Gram-Schmidt):")
for vec in ortho_vectors:
    print(vec)
print()

# -------------------------
# Ejercicio 10: Espacio nulo
# -------------------------
A10 = Matrix([[1, 2, 0, 3], [0, 1, 0, 2], [0, 0, 1, 1]])  # Definimos matriz simbólica
null_space_A10 = A10.nullspace()  # Calculamos el espacio nulo
print("Ejercicio 10 - Base del espacio nulo:\n", null_space_A10, "\n")
