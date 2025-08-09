# \rAIn\projects\argos\panoptes\model\_fetch_v12.py

from ultralytics import YOLO # type: ignore
import sys
for n in sys.argv[1:]:
    try:
        YOLO(n)  # downloads to CWD if missing
        print('ok:', n)
    except Exception as e:
        print('FAILED:', n, '->', e)
