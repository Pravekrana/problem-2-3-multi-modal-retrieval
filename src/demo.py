"""Make two distinguishable sample videos for an end-to-end retrieval smoke test."""
import cv2
import numpy as np
from pathlib import Path
folder = Path("demo_videos")
folder.mkdir(exist_ok=True)
for name,color in [("red_circle.avi",(0,0,255)),("green_square.avi",(0,255,0))]:
    path=folder/name
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),5,(160,120))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    for i in range(20):
        frame=np.zeros((120,160,3),dtype=np.uint8)
        if name.startswith("red"):
            cv2.circle(frame,(40+i*3,60),25,color,-1)
        else:
            cv2.rectangle(frame,(35+i*2,40),(75+i*2,80),color,-1)
        writer.write(frame)
    writer.release()
    print(path)
