"""
sniffer.py
----------
Run this with Python 3.6 + pyhula installed while connected to the drone.
It monkey-patches the UDP socket to intercept every raw MAVLink packet
sent and received, decodes them, and logs them to sniffer_log.json.

Usage (in your 3.6 venv):
    py -3.6 sniffer.py

Then in a SEPARATE terminal, run your original working main.py so it
exercises all the commands (takeoff, move, land, etc).
This script will capture everything and write sniffer_log.json.
Send that file back and we can build the full rewrite from it.
"""

import socket
import json
import time
import threading
import struct
import os
import sys

LOG_FILE = "sniffer_log.json"
DRONE_IP = "192.168.100.87"  # change if different
LISTEN_PORT = 14550           # standard MAVLink UDP port — adjust if needed

packets = []
lock = threading.Lock()


# ── MAVLink minimal parser ────────────────────────────────────────────────────
# We parse the header ourselves so we don't need pymavlink here.
# MAVLink v1: STX=0xFE, len, seq, sysid, compid, msgid, payload..., crc(2)
# MAVLink v2: STX=0xFD, len, incompat, compat, seq, sysid, compid, msgid(3), payload..., crc(2)

def parse_mavlink(data: bytes) -> dict:
    if len(data) < 6:
        return {"raw": data.hex(), "version": "unknown"}

    stx = data[0]
    if stx == 0xFE:  # MAVLink v1
        if len(data) < 8:
            return {"raw": data.hex(), "version": "v1_short"}
        payload_len = data[1]
        seq     = data[2]
        sysid   = data[3]
        compid  = data[4]
        msgid   = data[5]
        payload = data[6:6 + payload_len]
        return {
            "version":     "v1",
            "seq":         seq,
            "sysid":       sysid,
            "compid":      compid,
            "msgid":       msgid,
            "payload_hex": payload.hex(),
            "payload_len": payload_len,
            "raw":         data.hex(),
        }
    elif stx == 0xFD:  # MAVLink v2
        if len(data) < 10:
            return {"raw": data.hex(), "version": "v2_short"}
        payload_len  = data[1]
        seq          = data[4]
        sysid        = data[5]
        compid       = data[6]
        msgid        = data[7] | (data[8] << 8) | (data[9] << 16)
        payload      = data[10:10 + payload_len]
        return {
            "version":     "v2",
            "seq":         seq,
            "sysid":       sysid,
            "compid":      compid,
            "msgid":       msgid,
            "payload_hex": payload.hex(),
            "payload_len": payload_len,
            "raw":         data.hex(),
        }
    else:
        return {"raw": data.hex(), "version": "unknown_stx", "stx": hex(stx)}


def log_packet(direction: str, addr, data: bytes, label: str = ""):
    parsed = parse_mavlink(data)
    entry = {
        "t":         time.time(),
        "direction": direction,   # "SEND" or "RECV"
        "addr":      str(addr),
        "label":     label,
        "parsed":    parsed,
    }
    with lock:
        packets.append(entry)
        print(f"[{direction}] msgid={parsed.get('msgid','?'):>6}  "
              f"len={parsed.get('payload_len','?'):>4}  "
              f"addr={addr}  {label}")


# ── Intercept socket ──────────────────────────────────────────────────────────
_real_socket = socket.socket

class SniffingSocket(_real_socket):
    """Wraps every UDP socket to log all send/recv traffic."""

    def sendto(self, data, *args):
        addr = args[-1] if args else "?"
        log_packet("SEND", addr, bytes(data))
        return super().sendto(data, *args)

    def send(self, data, flags=0):
        log_packet("SEND", "connected", bytes(data))
        return super().send(data, flags)

    def recvfrom(self, bufsize, flags=0):
        data, addr = super().recvfrom(bufsize, flags)
        if data:
            log_packet("RECV", addr, bytes(data))
        return data, addr

    def recv(self, bufsize, flags=0):
        data = super().recv(bufsize, flags)
        if data:
            log_packet("RECV", "connected", bytes(data))
        return data

socket.socket = SniffingSocket
print("[Sniffer] Socket patched — all MAVLink traffic will be logged.")


# ── Auto-save loop ────────────────────────────────────────────────────────────
def save_loop():
    while True:
        time.sleep(3)
        with lock:
            snapshot = list(packets)
        with open(LOG_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)

threading.Thread(target=save_loop, daemon=True).start()
print(f"[Sniffer] Logging to {LOG_FILE} — now run your main.py in another terminal.")
print("[Sniffer] Press Ctrl+C to stop and do a final save.\n")

try:
    # Keep alive — the patched socket module stays active for the lifetime
    # of this process. But since main.py runs in a SEPARATE process, we
    # need a different approach: import and run it directly here.
    #
    # OPTION A (recommended): import your original script directly so
    # everything runs in the same process and the patch applies.
    #
    # Uncomment the block below and set the path to your original main.py:
    # -------------------------------------------------------------------------
    # import importlib.util
    # spec = importlib.util.spec_from_file_location(
    #     "original_main",
    #     r"C:\path\to\your\original_main.py"   # ← set this
    # )
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # -------------------------------------------------------------------------
    #
    # OPTION B: just keep this process alive and manually call pyhula here.
    # This is simpler — edit the block below with your drone IP:

    import pyhula
    from hula_video import hula_video
    import cv2

    api = pyhula.UserApi()
    if not api.connect(DRONE_IP):
        print("[Sniffer] Connection failed — check DRONE_IP at top of file")
        sys.exit(1)

    print("[Sniffer] Connected! Running command sequence...")

    vid = hula_video(hula_api=api, display=False)
    vid.video_mode_on()

    print("[Sniffer] >>> takeoff"); api.single_fly_takeoff(); time.sleep(2)
    print("[Sniffer] >>> forward 30"); api.single_fly_forward(30); time.sleep(2)
    print("[Sniffer] >>> back 30"); api.single_fly_back(30); time.sleep(2)
    print("[Sniffer] >>> turn left 90"); api.single_fly_turnleft(90); time.sleep(2)
    print("[Sniffer] >>> turn right 90"); api.single_fly_turnright(90); time.sleep(2)
    print("[Sniffer] >>> up 20"); api.single_fly_up(20); time.sleep(2)
    print("[Sniffer] >>> down 20"); api.single_fly_down(20); time.sleep(2)
    print("[Sniffer] >>> hover 2s"); api.single_fly_hover_flight(2); time.sleep(3)
    print("[Sniffer] >>> land"); api.single_fly_touchdown(); time.sleep(2)

    vid.close()
    print("[Sniffer] Command sequence complete.")

except KeyboardInterrupt:
    print("\n[Sniffer] Stopped by user.")
finally:
    with lock:
        snapshot = list(packets)
    with open(LOG_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[Sniffer] Final save → {LOG_FILE}  ({len(snapshot)} packets)")
    print("Send sniffer_log.json back and we'll build the rewrite from it.")
