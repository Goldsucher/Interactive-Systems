import numpy as np


# a)
a = np.zeros(10)
a[4] = 1
print("a): ")
print (a)

# b)
b = np.arange(10, 50)
print("b): ")
print (b)

# c)
c = b[::-1]
print("c): ")
print (c)

# d)
d = np.arange(0, 16).reshape(4, 4)
print("d): ")
print(d)

# e)
e = np.random.randint(100, size=(8, 8))
print("e): ")
print(e)
print("min: ", e.min())
print("max: ", e.max())
print("normalized: ", e / np.full(e.shape, e.max()))

# f)
print("f): ")
f4 = np.random.randint(100, size=(4, 3))
f3 = np.random.randint(100, size=(3, 2))
print("4x3:", f4)
print("3x2:", f3)
print("4x3 x 3+2:", np.matmul(f4, f3))

# g) - nicht fertig
print("g): ")
g = np.arange(0, 21)
print(g)

# h)
print("h):)")
h = np.sum(np.random.randint(100, size=(1, 20)))
print(h)

# i)
print("i):")
i = np.random.randint(100, size=(5, 5))
print(i)
print("ungerade Zeile:")
print(i[::2])
print("gerade Zeile:")
print(i[1::2])

# j)
print("j):")
jM = np.random.randint(10, size=(4, 3))
jV = np.random.randint(10, size=(1, 3))
print("Matrix:")
print(jM)
print("Vector:")
print(jV)
print("Result:")
print(np.multiply(jM, jV))

# k)
k = np.random.randint(10, size=(10, 2))
print(k)
