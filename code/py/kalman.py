# kalman.py

# Librerias importadas
import cv2

# Video a analizar
cap = cv2.VideoCapture('../../../../video/unq/agora.mp4')
#cap = cv2.VideoCapture('../../../../video/balkon/balkonSummer.mp4')

# Creacion del sustractor de fondo MOG2
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0) #127
bs.setHistory(100) #500
#bs.setVarThreshold(36) #16
#bs.setBackgroundRatio(0.5) #0.9

bs1 = cv2.createBackgroundSubtractorMOG2()
bs1.setDetectShadows(True)
bs1.setShadowValue(0)
bs1.setHistory(100)
#bs1.setVarThreshold(36)
#bs1.setBackgroundRatio(0.5)

# Tiempo en ms a esperar entre frames, 0 = para siempre
tms = 10

# Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# Loop principal
while(cap.isOpened()):
  frame = cv2.resize(cap.read()[1],(480,480))
  # Se aplica el sustractor de fondo
  blur = cv2.GaussianBlur(frame,(11,11),0)
  msk = bs.apply(blur)
  # Se filtra el ruido
  frg = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
  # Se muestran los distintos pasos del algoritmo
#  cv2.imshow('1 - Frame',cv2.resize(frame,(500,500)))
  cv2.imshow('2 - FRG',frg)
  cv2.moveWindow('2 - FRG',0,0)
#  cv2.imshow('3 - FFRG',ffrg)
  # Botones de terminado, pausado, reanudar
  k = cv2.waitKey(tms) & 0xFF
  if k == ord('q'):
    break # Terminar
  elif k == ord('p'):
    tms = 0 #Pausar
  elif k == ord('f'):
    tms = 10 # Reanudar

# Liberar el video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
