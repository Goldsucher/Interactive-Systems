import numpy as np
import cv2

img_grey = cv2.imread("Lenna.png", 0)

print("img_grey alt=")
print(img_grey)

#img_grey[1, 1] = img_grey[0, 0] * 0.0625 + img_grey[0, 1] * 0.125 + img_grey[0, 2] * 0.0625 + img_grey[1, 0] * 0.125 + img_grey[1, 1] * 0.25 + img_grey[1, 2] * 0.125 + img_grey[2, 0] * 0.0625 + img_grey[2, 1] * 0.125 + img_grey[2, 2] * 0.0625

img_size = len(img_grey)

#for y, x in range(1, img_size):
 #   img_grey[y, x] = img_grey[y-1, x-1] * 0.0625 + img_grey[y-1, x] * 0.125 + img_grey[y-1, x+1] * 0.0625 + img_grey[y, x-1] * 0.125 + img_grey[y, x] * 0.25 + img_grey[y, x+1] * 0.125 + img_grey[y+1, x-1] * 0.0625 + img_grey[y+1, x] * 0.125 + img_grey[y+1, x+1] * 0.0625

for y in range(1, img_size-1):
    for x in range(1, img_size-1):
        img_grey[y, x] = img_grey[y-1, x-1] * 0.0625 + img_grey[y-1, x] * 0.125 + img_grey[y-1, x+1] * 0.0625 + img_grey[y, x-1] * 0.125 + img_grey[y, x] * 0.25 + img_grey[y, x+1] * 0.125 + img_grey[y+1, x-1] * 0.0625 + img_grey[y+1, x] * 0.125 + img_grey[y+1, x+1] * 0.0625
        # img_grey[y, x] = img_grey[y-1, x-1] * 0.0625 + img_grey[y-1, x] * 0.125 + img_grey[y-1, x+1] * 0.0625 + img_grey[y, x-1] * 0.125 + img_grey[y, x] * 0.25 + img_grey[y, x+1] * 0.125 + img_grey[y+1, x-1] * 0.0625 + img_grey[y+1, x] * 0.125 + img_grey[y+1, x+1] * 0.0625
        # img_grey[y, x] = img_grey[y-1, x-1] * 0.0625 + img_grey[y-1, x] * 0.125 + img_grey[y-1, x+1] * 0.0625 + img_grey[y, x-1] * 0.125 + img_grey[y, x] * 0.25 + img_grey[y, x+1] * 0.125 + img_grey[y+1, x-1] * 0.0625 + img_grey[y+1, x] * 0.125 + img_grey[y+1, x+1] * 0.0625
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111
        img_grey[y, x] = img_grey[y - 1, x - 1] * 0.1111111111111 + img_grey[y - 1, x] * 0.1111111111111 + img_grey[y - 1, x + 1] * 0.1111111111111 + img_grey[y, x - 1] * 0.1111111111111 + img_grey[y, x] * 0.1111111111111 + img_grey[y, x + 1] * 0.1111111111111 + img_grey[y + 1, x - 1] * 0.1111111111111 + img_grey[y + 1, x] * 0.1111111111111 + img_grey[y + 1, x + 1] * 0.1111111111111

print("img_grey neu=")
print(img_grey)


cv2.imshow('result', img_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()


# print(img_grey.item(17))
# img_grey.item(17) = img_grey.item(1)*0.0625 + img_grey.item(2)*0.125 + img_grey.item(3)*0.0625 + img_grey.item(16)*0.125 + img_grey.item(17)*0.25 + img_grey.item(18)*0.125 + img_grey.item(31)*0.0625 + img_grey.item(32)*0.125 + img_grey.item(33)*0.0625
blur = []
blur = img_grey.item(1) + img_grey.item(2)

# print(e)

#print(img_grey.item(1, 1))  # 151 --> y,x - 1
#print(img_grey.item(1, 2))  # 114 --> y - 1
#print(img_grey.item(1, 3))  # 130 --> y - 1, x + 1
#print(img_grey.item(2, 1))  # 149 --> x - 1
#print(img_grey.item(2, 2))  # 111 -->
#print(img_grey.item(2, 3))  # 126 --> x + 1
#print(img_grey.item(3, 1))  # 150 --> y + 1, x - 1
#print(img_grey.item(3, 2))  # 111 --> y + 1
#print(img_grey.item(3, 3))  # 129 --> y,x + 1

# print(len(img_grey) * len(img_grey))


img_dimension = (len(img_grey), (len(img_grey)))
# result = []
nullen = np.zeros(img_dimension, dtype=int)

#for y in np.nditer(img_grey, op_flags=['readwrite']):
    #print(img_grey[y])
    #y[...] = img_grey

    #print(np.nditer.index)
    #y[...] = y + 5



#for y in range(1, img_size):
 #   new = img_grey[1, y] - img_grey[0, y - 1] * 0.0625 + img_grey[0, ]


#print("0")
#print(img_grey[1, 0])
#print()
#print(img_grey)



#for y in range(img_size):
 #   print(img_grey.item(y))
   # qw = sum(y)
    #qw.append([img_grey.item(0)])
    #nullen = nullen + img_grey.item(y)
    #for x in range(img_size, 15):
        #blur = np.zeros(15, 15)
        #print(img_grey.item(y, x))
        #result = np.append(img_grey.item(y, x))
        #result = nullen + img_grey.item(y, x)

print()
#print(qw)
#print(qw1)
print()
#print(nullen)

# print(blur)



# print(img_grey[::]+1)

#for i in np.ndindex(img_grey.shape[:2]):
  #  print(i)


# ssdd = img_grey.item(2, 2)

