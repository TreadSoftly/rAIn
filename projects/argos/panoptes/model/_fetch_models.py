from ultralytics import YOLO # type: ignore
import sys
for n in sys.argv[1:]:
    try:
        YOLO(n)  # downloads to CWD if not present
        print('ok:', n)
    except Exception as e:
        print('FAILED:', n, '->', e)
