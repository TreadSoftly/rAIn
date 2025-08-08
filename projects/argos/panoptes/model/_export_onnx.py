from ultralytics import YOLO
import os, sys, glob, shutil
targets = [p for p in sys.argv[1:] if os.path.exists(p)]
print('Exporting to ONNX:', targets)
for p in targets:
    try:
        m = YOLO(p)
        out = m.export(format='onnx', dynamic=True, simplify=True, imgsz=640, opset=12, device='cpu')
        print('ONNX:', out)
    except Exception as e:
        print('FAILED export:', p, '->', e)
for onnx in glob.glob('runs/**/**/*.onnx', recursive=True):
    dst = os.path.join(os.getcwd(), os.path.basename(onnx))
    try:
        shutil.copy2(onnx, dst)
        print('copied:', onnx, '->', dst)
    except Exception as e:
        print('FAILED copy:', onnx, '->', e)
