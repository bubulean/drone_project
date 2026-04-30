# -*- coding: utf-8 -*-
"""
hula_video.py
-------------
Video stream handler for the Hula drone.
Uses the original ffmpeg-lib.dll via ctypes (Python 3.6 + pyhula).

Bug fixes over the original:
- Fixed video_queue.empty bug (was a property ref, never called)
- Added threading.Lock for thread-safe queue access
- Removed unused detect/detecting state
- close() is idempotent
"""

import threading
import time
import os
import cv2
import ctypes
import numpy as np
from collections import deque


class hula_video(object):

    def __init__(self, hula_api, hula_ip="0.0.0.0", display=False):
        self.uapi        = hula_api
        self.hula_ip     = hula_ip
        self.display     = display
        self.live        = display
        self.video_port  = 9000 + (self.uapi.get_plane_id() * 2)

        self._queue      = deque()
        self._lock       = threading.Lock()
        self.stopApp     = False
        self.record      = False
        self.photo_filename = "photo"
        self.photo_index    = 0
        self.buffer_size    = 25

        self.savepath = os.path.join(os.getcwd(), "photo")
        if not os.path.exists(self.savepath):
            print("creating save directory")
            os.makedirs(self.savepath)

        self.lib = ctypes.cdll.LoadLibrary("ffmpeg-lib.dll")
        self._setup_lib()

        self.videothread = threading.Thread(
            target=self._receive_video_data, daemon=True
        )

    # ── Backwards-compat ─────────────────────────────────────────────────────
    @property
    def video_queue(self):
        return self._queue

    def _setup_lib(self):
        lib = self.lib
        lib.init_video.argtypes          = [ctypes.c_wchar_p, ctypes.c_int]
        lib.init_video.restype           = ctypes.c_int
        lib.get_rgb_datas_length.argtypes = []
        lib.get_rgb_datas_length.restype  = ctypes.c_int
        lib.get_width.argtypes            = []
        lib.get_width.restype             = ctypes.c_int
        lib.get_height.argtypes           = []
        lib.get_height.restype            = ctypes.c_int
        lib.get_rgb_datas.argtypes        = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        lib.get_rgb_datas.restype         = None
        lib.get_rgb_ptr.argtypes          = []
        lib.get_rgb_ptr.restype           = ctypes.POINTER(ctypes.c_uint8)
        lib.should_update_frame.argtypes  = []
        lib.should_update_frame.restype   = ctypes.c_bool
        lib.has_updated_frame.argtypes    = []
        lib.has_updated_frame.restype     = None

    # ── Public API ────────────────────────────────────────────────────────────

    def video_mode_on(self):
        """Turn on the drone's RTP stream and start the receiver thread."""
        self.stopApp = False
        self.uapi.Plane_cmd_swith_rtp(0)
        print("Starting video stream. Please wait.")
        time.sleep(1)
        self.lib.init_video(self.hula_ip, self.video_port)
        while self.lib.get_rgb_datas_length() <= 0:
            time.sleep(0.05)
        print("Stream started.")
        self.videothread.start()

    def get_video(self, get_latest=True, keep_getting=True, timeout=0.1):
        """
        Return a BGR numpy frame or None.
        Returns within timeout seconds so callers stay responsive.
        """
        deadline = time.time() + timeout
        while not self.stopApp:
            with self._lock:
                if self._queue:
                    return self._queue.pop() if get_latest else self._queue.popleft()
            if not keep_getting or time.time() >= deadline:
                return None
            time.sleep(0.005)
        return None

    def get_frame(self, latest=True, block=True):
        return self.get_video(get_latest=latest, keep_getting=block)

    def get_image_size(self):
        return (self.lib.get_height(), self.lib.get_width())

    def startrecording(self, filename="photo"):
        self.record = True
        self.photo_filename = filename
        self.photo_index = 0

    def stoprecording(self):
        self.record = False

    def stop_live(self):
        self.live = False
        with self._lock:
            self._queue.clear()

    def close(self):
        if not self.stopApp:
            self.stopApp = True
            try:
                self.uapi.Plane_cmd_swith_rtp(1)
            except Exception:
                pass
            if self.display:
                try:
                    cv2.destroyWindow("HulaVideo")
                except Exception:
                    pass
            print("Video stream closed")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _receive_video_data(self):
        print("Video Receiver started on port %d" % self.video_port)
        length   = self.lib.get_rgb_datas_length()
        vid_w    = self.lib.get_width()
        vid_h    = self.lib.get_height()

        if self.display:
            cv2.namedWindow("HulaVideo", cv2.WINDOW_NORMAL)

        while not self.stopApp:
            if not self.lib.should_update_frame():
                time.sleep(0.005)
                continue

            ptr   = self.lib.get_rgb_ptr()
            data  = np.ctypeslib.as_array(ptr, shape=(length,))[:length]
            frame = np.frombuffer(data, dtype=np.uint8).reshape((vid_h, vid_w, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with self._lock:
                if len(self._queue) >= self.buffer_size:
                    self._queue.popleft()
                self._queue.append(frame)

            if self.record:
                path = os.path.join(
                    self.savepath,
                    "%s%d.jpg" % (self.photo_filename, self.photo_index)
                )
                cv2.imwrite(path, frame)
                self.photo_index += 1

            if self.display:
                cv2.imshow("HulaVideo", frame)
                cv2.waitKey(1)

        if self.display:
            cv2.destroyWindow("HulaVideo")

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.video_mode_on()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


HulaVideo = hula_video