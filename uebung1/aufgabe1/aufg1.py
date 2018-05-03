# Diese Code beinhaltet Aufgaben 1 a) - m) vom Übungsblatt 1
# Stephan Wagner s853668

import numpy as np
import math

# a)
a = np.zeros(10)
a[4] = 1
print("a): ")
print(a)
print("\n")

# b)
b = np.arange(10, 50)
print("b): ")
print (b)
print("\n")

# c)
c = b[::-1]
print("c): ")
print (c)
print("\n")


# d)
d = np.arange(0, 16).reshape(4, 4)
print("d): ")
print(d)
print("\n")

# e)
e = np.random.randint(100, size=(8, 8))
print("e): ")
print(e)
print("min: ", e.min())
print("max: ", e.max())
print("normalisiert: ", e / np.full(e.shape, e.max()))
print("\n")

# f)
print("f): ")
f4 = np.random.randint(100, size=(4, 3))
f3 = np.random.randint(100, size=(3, 2))
print("4x3:", f4)
print("3x2:", f3)
print("4x3 x 3+2:", np.matmul(f4, f3))
print("\n")

# g)
print("g): ")
print("Voher: ")
g = np.arange(0, 21)
print(g)
print("Nachher: ")
g = np.concatenate((g[:9], np.negative(g[9:16]), g[16:]), axis=0)
print(g)
print("\n")

# h)
print("h):")
h = np.sum(np.random.randint(100, size=(1, 20)))
print(h)
print("\n")

# i)
print("i):")
i = np.random.randint(100, size=(5, 5))
print(i)
print("ungerade Zeile:")
print(i[::2])
print("gerade Zeile:")
print(i[1::2])
print("\n")

# j)
print("j):")
jM = np.random.randint(10, size=(4, 3))
jV = np.random.randint(10, size=(1, 3))
print("Matrix:")
print(jM)
print("Vector:")
print(jV)
print("Ergebnis:")
print(np.multiply(jM, jV))
print("\n")

# k)
def convertCart2Pol(a):
    rho = np.sqrt(a[0]**2 + a[1]**2)
    phi = np.arctan2(a[1], a[0])

    return rho, phi

print("k):")
print("Zufallsmatrix:")
k = np.random.randint(10, size=(10, 2))
print(k)
print("Konvertierte Matrix:")
k = np.apply_along_axis(convertCart2Pol, 1, k)
print(k)
print("\n")


# l)
def scalar_product(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0

    return sum(x * y for x, y in zip(vector1, vector2))

def vector_length(vector):

    return math.sqrt(sum(map(lambda x: pow(x, 2), vector)))

print("l):")
v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([-1, 9, 5, 3, 1])
print("scalar_product(vector1, vector2): ")
print(scalar_product(v1, v2))
print("np.inner(v1, v2): ")
print(np.inner(v1, v2))
print("Vektorlänger von v1: ")
print(vector_length(v1))
print("np.linalg.norm(v1): ")
print(np.linalg.norm(v1))
print("Vektorlänger von v2): ")
print(vector_length(v2))
print("np.linalg.norm(v2): ")
print(np.linalg.norm(v2))
print("\n")

# m)
print("m):")
v0 = np.matrix([[1], [1], [0]])
print("v0:")
print(v0)
v0t = v0.T
print("v0t:")
print(v0t)
v1 = np.matrix([[-1], [2], [5]])
print("v1:")
print(v1)
m = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]])
print("M:")
print(m)
l = np.multiply(v0t.dot(v1), m.dot(v1))
print("Ergebnis: ")
print(l)