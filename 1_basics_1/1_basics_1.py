import numpy as np


# Aufgabe 1a)
a = np.zeros((10))   # Create an array of all zeros
a[4] = 1
print("Aufgabe 1a:")
print(a)
print()

# Aufgabe 1b)
b = np.arange(10, 50)
print("Aufgabe 1b:")
print(b)
print()

# Aufgabe 1c)
c = b[::-1]
print("Aufgabe 1c:")
print(c)
print()

# Aufgabe 1d)
d = np.arange(16).reshape(4, 4)
print("Aufgabe 1d:")
print(d)
print()

# Aufgabe 1e)
e = np.random.randint(100, size=(8,8))
print("Aufgabe 1e:")
print(e)
print()
print("Minimum:", e.min())
print("Maximum:", e.max())
print()
en = e / np.full(e.shape, e.max())
print("normalisiert:")
print(en)
print()

# Aufgabe 1f)
f1 = np.random.randint(100, size=(4,3))
print("Aufgabe 1f:")
print("4x3:")
print(f1)
f2 = np.random.randint(100, size=(3,2))
print()
print("3x2")
print(f2)
f = np.matmul(f1, f2)
print()
print("multipliziert")
print(f)
print()

# Aufgabe 1g)
print("Aufgabe 1g:")
print("1D Array")
g = np.arange((21))
print(g)
# negativieren
gneg = np.negative(g[8:17])
# negierte Werte einsetzen
gnew = np.concatenate((g[:8], gneg, g[17:]), axis=0)
print("Lösung")
print(gnew)
print()

# Aufgabe 1h)
print("Aufgabe 1h:")
# Array erzeugen
h = np.random.randint(100, size=(1,20))
print("Array")
print(h)
#Array summieren
hsum = np.sum(h)
print("Array summiert")
print(hsum)
print()

# Aufgabe 1i)
print("Aufgabe 1i:")
# Array erzeugen
i = np.random.randint(100, size=(5,5))
print("5x5 Matrix")
print(i)
# ungerade Zeilen
print("ungerade Zeilen")
print(i[::2])
# gerade Zeilen
print("gerade Zeilen")
print(i[1::2])
print()

# Aufgabe 1j)
print("Aufgabe 1j:")
# Matrix erzeugen
jM = np.random.randint(100, size=(4, 3))
print("Matrix")
print(jM)
# Vektor erzeugen
jV = np.arange(3)
print()
print("Vector")
print(jV)
# Multiplizieren
jB = np.multiply(jM, jV)
print()
print("Lösung")
print(jB)
print()

# Aufgabe 1k)
