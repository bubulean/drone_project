import pyhula
from hula_video import hula_video
import time
import cv2
import os

api = pyhula.UserApi()
SNAPSHOT_DIR = "photo"
DETECTION_LOG_FILE = "detections.txt"

if not api.connect("192.168.100.87"):
    print("connect error")
else:
    print('connection to station by wifi')
    # api.single_fly_lamplight(255, 0, 0, 5, 64)
    # api.plane_fly_generating(4, 3, 3)
    # time.sleep(1)
    api.plane_fly_generating(5, 10, 100)

    # time.sleep(5)
    # api.single_fly_takeoff()
    # api.single_fly_up(50) 
    # time.sleep(2)
    # api.single_fly_touchdown()

    # vid = hula_video(hula_api=api, display=False)
    # huladetector = onnxdetector(
    #     model='model/detect_3_object_12_11.onnx',
    #     label="model/object.txt",
    #     confidence_thres=0.4
    # )
    # vid.video_mode_on()
    # api.single_fly_takeoff()
    # time.sleep(0.5)

    # found = False
    # frame = vid.get_video()
    # print("[SCAN] Frame is", "OK" if frame is not None else "None")
    # if frame is not None:
    #     obj_found, frame_out = huladetector.detect(frame)

    #     if obj_found is not None:
    #         found = True
    #         label = obj_found.get("label", "object")

    #         # Ensure snapshot directory exists
    #         os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    #         snapshot_path = os.path.join(
    #             SNAPSHOT_DIR, f"{label}_{time.time()}.png"
    #         )

    #         # Prefer frame_out if available (may have bounding boxes drawn)
    #         img_to_save = frame_out if frame_out is not None else frame
    #         try:
    #             cv2.imwrite(snapshot_path, img_to_save)
    #             print(f"[SCAN] Saved snapshot: {snapshot_path}")
                
    #         except Exception as e:
    #             print(f"[SCAN] Could not save snapshot: {e}")
    #             snapshot_path = ""  # avoid confusing GUI with bad path

    #         # Log detection for the GUI
            
    #         try:
    #             with open(DETECTION_LOG_FILE, "a") as f:
    #                 f.write(f"{label} {snapshot_path}\n")
    #         except Exception as e:
    #             print(f"[SCAN] Could not write to {DETECTION_LOG_FILE}: {e}")

    
    
    # print("[SCAN] found =", found)
    
    # print("[Sniffer] >>> forward 30"); api.single_fly_forward(30); time.sleep(2)
    # print("[Sniffer] >>> back 30"); api.single_fly_back(30); time.sleep(2)
    # print("[Sniffer] >>> turn left 90"); api.single_fly_turnleft(90); time.sleep(2)
    # print("[Sniffer] >>> turn right 90"); api.single_fly_turnright(90); time.sleep(2)
    # print("[Sniffer] >>> up 20"); api.single_fly_up(20); time.sleep(2)
    # print("[Sniffer] >>> down 20"); api.single_fly_down(20); time.sleep(2)
    # print("[Sniffer] >>> hover 2s"); api.single_fly_hover_flight(2); time.sleep(3)
    
    # api.single_fly_touchdown()
    # vid.close()

if __name__ == "__main__":
    pass
    
