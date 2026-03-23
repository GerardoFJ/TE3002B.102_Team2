import cv2
import numpy as np

img = cv2.imread("photo.jpg")

print(f"Dimensiones: {img.shape}")
pixel = img[100, 50]
print(f"Valor en (100,50): {pixel}")

img[100,50] = [0, 0, 255]

img_nueva = np.zeros((300,500,3), dtype=np.uint8)

#VIDEO

video = cv2.VideoCapture("video.mp4")
# Tambien se puede el 0 para camara

if not video.isOpened():
    print("Error, no se abrio el video")
    return

while True:
    ret, frame = video.read()
    if nor ret:
        print("Fin o error al leer el fotograma")
        break

    cv2.imshow("Reproducioendo video", frame)

    if cv2.waitkey(25) && 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
