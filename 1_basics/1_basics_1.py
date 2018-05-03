# Blatt 1 "1_basics"
# Aufgabe 1: Linear Algebra und Numpy Basics
# Steffen Burlefinger (859077)


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
e = np.random.randint(100, size=(8, 8))
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
f1 = np.random.randint(100, size=(4, 3))
print("Aufgabe 1f:")
print("4x3:")
print(f1)
f2 = np.random.randint(100, size=(3, 2))
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
gnew = np.concatenate((g[:8], gneg, g[17:]), axis=0)  # 0 = horizontal, 1 = vertikal
print("Lösung")
print(gnew)
print()

# Aufgabe 1h)
print("Aufgabe 1h:")
# Array erzeugen
h = np.random.randint(100, size=(1, 20))
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
i = np.random.randint(100, size=(5, 5))
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
# https://www.w3resource.com/python-exercises/numpy/python-numpy-random-exercise-14.php
# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
print("Aufgabe 1k:")
print("Zufallsmatrix 10 x 2 = Kartesische Koordinaten")
k = np.random.randint(100, size=(10, 2))
print(k)
print()
# X-Koordinaten extrahieren
x = k[::, 0]
# Y-Koordinaten extrahieren
y= k[::, 1]


# Radius berechnen
r = np.sqrt(x**2 + y**2)
# Polarwinkel berechnen
phi = np.arctan2(y, x)
print("Ergebnis = Polarkoordinaten:")

# in Koordinatenform bringen
for i in range(r.size):
    resultk = np.array([r.item(i), phi.item(i)])
    print(resultk)
print()
print()

# Aufgabe 1l)
# https://www.mathebibel.de/skalarprodukt
print("Aufgabe 1l:")
# random Länge generieren
lrand = np.random.randint(30)
# Vektor 1 beliebiger Länge
lv1 = np.random.randint(0, 100, lrand)
# Vektor 2 beliebiger Länge
lv2 = np.random.randint(0, 100, lrand)
print("Vektor 1:")
print(lv1)
print()
print("Vektor 2:")
print(lv2)
print()

#leere Liste für Produkte aus Vektorkoordinaten
prodlarray = []
for i in range(lv1.size):
    #Produkte berechnen
    prodl = lv1.item(i) * lv2.item(i)
    #Produkte in Liste schreiben
    prodlarray.append(prodl)

# DEBUG: Produkte ausgeben
# print(prodlarray)
# Summe aus Produkten berechnen
resultl = sum(prodlarray)
print("Skalarprodukt:")
print(resultl)
print()

# Test mit NumPy Funktion
resultlnp = np.dot(lv1, lv2)
print("Skalarprodukt zum Vergleich mit np.dot berechnet:")
print(resultlnp)
print()

# Aufgabe 1l Test mit vorgegebenen Vektoren:
print("Aufgabe 1l - Test mit vorgegebenen Vektoren:")
lv1t = np.array([1, 2, 3, 4 , 5])
lv2t = np.array([-1, 9, 5, 3, 1])
print("Testvektor 1:")
print(lv1t)
print("Testvektor 2:")
print(lv2t)
print()
testprodlarray = []
for i in range(lv1t.size):
    #Produkte berechnen
    testprodl = lv1t.item(i) * lv2t.item(i)
    #Produkte in Liste schreiben
    testprodlarray.append(testprodl)

# DEBUG: Produkte ausgeben
# print(testprodlarray)
# Summe aus Produkten berechnen
testresultl = sum(testprodlarray)
print("Skalarprodukt aus vorgegebenen Vektoren")
print(testresultl)
print()

# Test mit NumPy Funktion
testresultlnp = np.dot(lv1t, lv2t)
print("Skalarprodukt zum Vergleich mit np.dot berechnet:")
print(testresultlnp)
print()
print()


# Aufgabe 1m)
print("Aufgabe 1m:")
v0 = np.matrix([[1], [1], [0]])
print("v0:")
print(v0)
#v0t = v0.reshape(1, v0.shape[0])
v0t = v0.T  # transposed
print()
print("v0t:")
print(v0t)
v1 = np.matrix([[-1], [2], [5]])
print()
print("v1:")
print(v1)
m = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]])
print()
print("m:")
print(m)
#x = v0t.dot(v1)
#print("(v0t*v1) = ", x)
#y = m.dot(v1)
#print("m*v1 = ", y)
#b = np.multiply(x, y)
#print("b:", b)
ak = np.multiply(v0t.dot(v1), m.dot(v1))
print("\nak = np.multiply(v0t.dot(v1), m.dot(v1))\n")
print("Ergebnis:")
print(ak)