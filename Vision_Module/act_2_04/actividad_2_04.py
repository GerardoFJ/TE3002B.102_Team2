import sys
import time
import math
import threading
import numpy as np
import cv2

# Agregar la ruta de los módulos gRPC del simulador
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "LinuxCtrl" / "v2"))

import grpc
import te3002b_pb2
import te3002b_pb2_grpc
import google.protobuf.empty_pb2


class CenterLineDetector:
    def __init__(self):
        self.cameraWidth = 320
        self.cameraHeight = 240
        self._prev_cx = None

    def detect_center_line(self, image):
        """
        Detecta la línea central de la pista en el 1/4 inferior de la imagen y devuelve las coordenadas del mejor candidato.
        :param image: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato en coordenadas de la imagen original, None si no se detecta.
        """
        h, w = image.shape[:2]

        if self._prev_cx is None:
            self._prev_cx = w // 2

        roi_y_start = 3 * h // 4
        roi = image[roi_y_start:h, :]
        roi_h = roi.shape[0]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Histograma de columnas: píxeles muy oscuros (líneas negras de la pista)
        dark_hist = np.sum(gray < 60, axis=0).astype(float)

        # Suavizar histograma
        ks = max(3, w // 30)
        smooth = np.convolve(dark_hist, np.ones(ks) / ks, mode='same')

        # Excluir márgenes de escena
        margin = w * 8 // 100
        smooth[:margin] = 0
        smooth[w - margin:] = 0

        # Encontrar picos locales por encima del 20% del máximo
        threshold = smooth.max() * 0.2
        peaks = []
        for x in range(1, w - 1):
            if smooth[x] >= threshold and smooth[x] >= smooth[x - 1] and smooth[x] >= smooth[x + 1]:
                peaks.append(x)

        if peaks:
            # Elegir el pico más cercano al cx anterior (rastreo temporal)
            cx = min(peaks, key=lambda x: abs(x - self._prev_cx))
        else:
            cx = self._prev_cx

        cx = max(0, min(w - 1, cx))
        self._prev_cx = cx
        cy = roi_y_start + roi_h // 2

        return (cx, cy)


class SimTester:
    def __init__(self):
        self.channel = grpc.insecure_channel('127.0.0.1:7072')
        self.stub = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.datacmd = te3002b_pb2.CommandData()
        self.dataconfig = te3002b_pb2.ConfigurationData()
        self.running = True
        self.timer_delta = 0.025
        self.detector = CenterLineDetector()

    def callback(self):
        self.dataconfig.resetRobot = True
        self.dataconfig.mode = 2
        self.dataconfig.cameraWidth = 360
        self.dataconfig.cameraHeight = 240
        self.dataconfig.resetCamera = False
        self.dataconfig.scene = 2026
        self.dataconfig.cameraLinear.x = 0
        self.dataconfig.cameraLinear.y = 0
        self.dataconfig.cameraLinear.z = 0
        self.dataconfig.cameraAngular.x = 0
        self.dataconfig.cameraAngular.y = 0
        self.dataconfig.cameraAngular.z = 0
        req = google.protobuf.empty_pb2.Empty()

        self.stub.SetConfiguration(self.dataconfig)
        self.dataconfig.resetRobot = False
        time.sleep(0.25)
        self.stub.SetConfiguration(self.dataconfig)

        while self.running:
            result = self.stub.GetImageFrame(req)
            img_buffer = np.frombuffer(result.data, np.uint8)
            img_in = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

            if img_in is None:
                continue

            img = cv2.resize(img_in, (320, 240), interpolation=cv2.INTER_LANCZOS4)

            center = self.detector.detect_center_line(img)

            vis = img.copy()
            if center is not None:
                cx, cy = center
                cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
                cv2.line(vis, (cx, 0), (cx, 240), (0, 255, 0), 1)
                cv2.putText(vis, f"cx={cx} cy={cy}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.line(vis, (0, 180), (320, 180), (0, 0, 255), 1)
            cv2.imshow('CenterLine Detector', vis)
            cv2.waitKey(1)

            self.datacmd.linear.x = 0.01
            self.datacmd.linear.y = 0.0
            self.datacmd.linear.z = 0.0
            self.datacmd.angular.x = 0.0
            self.datacmd.angular.y = 0.0
            self.datacmd.angular.z = -0.001
            self.stub.SetCommand(self.datacmd)

            time.sleep(self.timer_delta - 0.001)


def main():
    node = SimTester()
    thread = threading.Thread(target=node.callback)
    thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.running = False
        thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
