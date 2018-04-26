import numpy as np
import math

print("Abgabe IS Aufgabe 1\n\n")

print("a) Erzeugen Sie einen Vektor mit Nullen der Länge 10 (10 Elemente) und setzen den Wert des 5.Elementes "
      "auf eine 1.\n")
print("a1 = np.zeros(10, dtype=int)\na1[4] = 1\n")
a1 = np.zeros(10, dtype=int)
a1[4] = 1
print("Ergebnis: ", a1)

print("\nb) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).\n")
print("a2 = np.arange(10, 50)\n")
a2 = np.arange(10, 50)
print("Ergebnis: ", a2)

print("\nc) Drehen Sie die Werte des Vektors um (geht in einer Zeile).\n")
print("a3 = a2[::-1]\n")
a3 = a2[::-1]
print("Ergebnis: ", a3)

print("\nd) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 bis 15 (links oben rechts unten).\n")
print("a4 = np.arange(16).reshape((4, 4))\n")
a4 = np.arange(16).reshape((4, 4))
print("Ergebnis: ", a4)

print("\ne) Erzeuge eine 8x8 Matrix mit Zufallswerte\n")
print("a5 = np.random.randint(101, size=(8, 8))\n")
a5 = np.random.randint(101, size=(8, 8))
print(a5)
print("\nund finde deren Maximum und Minimum\n")
print("min = a5.min()\nmax = a5.max()\n")
print("min: %s max: %s\n" % (a5.min(), a5.max()))
print("normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen")
print("ein Wert wird 1 (max) sein und einer 0 (min)).\n")
print("a5n = a5 / np.full(a5.shape, a5.max())\n")
a5n = a5 / np.full(a5.shape, a5.max())
print("Ergebnis: ", a5n)

print("\nf) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix\n")
print("a6a = np.random.randint(11, size=(4, 3))")
print("a6b = np.random.randint(11, size=(3, 2))")
print("a6 = np.matmul(a6a, a6b)\n")
a6a = np.random.randint(11, size=(4, 3))
a6b = np.random.randint(11, size=(3, 2))
a6 = np.matmul(a6a, a6b)
print("Ergebnis: \n", a6)

print("\ng) Erzeugen Sie ein 1D Array mit den Werte von 0 bis 20\n")
print("a7 = np.arange(21, dtype=int)\n")
a7 = np.arange(21, dtype=int)
print(a7)
print("\nund negieren Sie Werte zwischen 8 und 16 nachträglich.\n")
print("a7n = np.concatenate((a7[:9], np.negative(a7[9:16]), a7[16:]), axis=0)\n")
a7n = np.concatenate((a7[:9], np.negative(a7[9:16]), a7[16:]), axis=0)
print("Ergebnis: ", a7n)

print("\nh) Summieren Sie alle Werte in einem Array.\n")
print("a8 = np.random.randint(21, size=5)\n")
a8 = np.random.randint(21, size=5)
print(a8)
print("\na8sum = a8.sum()\n")
a8sum = a8.sum()
print("Ergebnis: ", a8sum)

print("\ni) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile aus.\n")
print("a9 = np.random.randint(21, size=(5, 5))")
a9 = np.random.randint(21, size=(5, 5))
print(a9)
print("\nEven (a9[1::2]):\n", a9[1::2])     # even [1:5:2]
print("\nOdd  (a9[::2]) :\n", a9[::2])      # odd  [0:5:2]

print("\nj) Erzeugen Sie eine Matrix M der Größe 4x3 und\n")
print("a10m = np.random.randint(11, size=(4, 3))\n")
a10m = np.random.randint(11, size=(4, 3))
print("matrix 4x3:\n", a10m)
print("\neinen Vektor v mit Länge 3.\n")
print("a10v = np.random.randint(11, size=3)\n")
a10v = np.random.randint(11, size=3)
print("vector len 3: ", a10v)
print("\nMultiplizieren Sie jeden Spalteneintrag aus v mit der kompletten Spalte aus M.")
print("Schauen Sie sich dafür an, was Broadcasting in Numpy bedeutet.\n")
print("a10 = np.multiply(a10m, a10v)")
a10 = np.multiply(a10m, a10v)
print("Ergebnis: \n", a10)

print("\nk) Erzeugen Sie eine Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten interpretieren")
print("können ([[x0, y0],[x1, y1],[x2, y2]]).\n")
print("a11m = np.random.randint(11, size=(10, 2))\n")
a11m = np.random.randint(11, size=(10, 2))
print(a11m)
print("\nKonvertieren Sie diese in Polarkoordinaten https://de.wikipedia.org/wiki/Polarkoordinaten.\n")
print("def cart2pol(a):\n\trho = np.sqrt(a[0]**2 + a[1]**2)\n\tphi = np.arctan2(a[1], a[0])\n\treturn(rho, phi)\n")


def cart2pol(a):
    rho = np.sqrt(a[0]**2 + a[1]**2)
    phi = np.arctan2(a[1], a[0])
    return rho, phi


print("a11 = np.apply_along_axis(cart2pol, 1, a11m)\n")
a11 = np.apply_along_axis(cart2pol, 1, a11m)
print("Ergebnis: \n", a11)

print("\nl) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlänge für Vektoren")
print("beliebiger Länge berechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen von NumPy. Testen ")
print("Sie Ihre Funktionen mit den gegebenen Vektoren:\n")
print("v1 = np.array([1, 2, 3, 4, 5])")
v1 = np.array([1, 2, 3, 4, 5])
print("v2 = np.array([-1, 9, 5, 3, 1])")
v2 = np.array([-1, 9, 5, 3, 1])
print("\ndef scalar_product(vector1, vector2):\n\treturn sum(x * y for x, y in zip(vector1, vector2))\n")


def scalar_product(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0
    return sum(x * y for x, y in zip(vector1, vector2))


print("scalar_product(v1, v2): ", scalar_product(v1, v2))
print("np.inner(v1, v2): ", np.inner(v1, v2))     # np.dot(v1, v2) geht auch
print("\ndef vector_length(vector):\n\treturn math.sqrt(sum(map(lambda x: pow(x, 2), vector)))\n")


def vector_length(vector):
    # return math.sqrt(scalar_product(vector, vector))  # same
    return math.sqrt(sum(map(lambda x: pow(x, 2), vector)))


print("vector_length(v1): ", vector_length(v1))
print("np.linalg.norm(v1): ", np.linalg.norm(v1))
print("vector_length(v2): ", vector_length(v2))
print("np.linalg.norm(v2): ", np.linalg.norm(v2))

print("\nl) Berechnen Sie (v0T v1)Mv0 unter der Nutzung von NumPy Operationen.")
print("Achten Sie darauf, dass hier v0,v1 Spaltenvektoren gegeben sind. v0T ist also ein Zeilenvektor.")
v0 = np.array([[1], [1], [0]])
print("v0:\n", v0)
v0t = v0.reshape(1, v0.shape[0])
print("v0t:", v0t)
v1 = np.array([[-1], [2], [5]])
print("v1:\n", v1)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]])
print("m:\n", m)

print("---")
x = (v0t * v1)
print(x)
print("---")
y = np.matmul(x, m)
print(y)
