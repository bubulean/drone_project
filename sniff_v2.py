"""
sniffer_v2.py
-------------
Network-level packet capture using scapy.
This captures ALL UDP traffic to/from the drone at the OS level,
so it works even when pyhula calls the socket API directly from C.

Requirements (run in your 3.6 venv OR any Python):
    pip install scapy

On Windows, scapy also requires Npcap:
    Download and install from: https://npcap.com/#download
    During install: check "Install Npcap in WinPcap API-compatible Mode"

Usage:
    1. Connect to drone WiFi
    2. Run this script as Administrator (required for raw packet capture)
    3. In a separate terminal, run your original working main.py
    4. This script captures everything and saves to sniffer_v2_log.json
    5. Send back sniffer_v2_log.json

Run as admin:
    Right-click terminal → "Run as Administrator"
    cd to your project folder
    py -3.6 sniffer_v2.py
"""

import json
import time
import threading

DRONE_IP  = "192.168.100.87"   # ← your drone's IP
LOG_FILE  = "sniffer_v2_log.json"

packets   = []
lock      = threading.Lock()

def parse_mavlink(data: bytes) -> dict:
    """Minimal MAVLink v1/v2 header parser."""
    if not data or len(data) < 6:
        return {"raw": data.hex() if data else "", "note": "too_short"}
    stx = data[0]
    if stx == 0xFE:  # v1
        if len(data) < 8:
            return {"raw": data.hex(), "version": "v1_short"}
        return {
            "version":     "v1",
            "seq":         data[2],
            "sysid":       data[3],
            "compid":      data[4],
            "msgid":       data[5],
            "payload_len": data[1],
            "payload_hex": data[6:6 + data[1]].hex(),
            "raw":         data.hex(),
        }
    elif stx == 0xFD:  # v2
        if len(data) < 10:
            return {"raw": data.hex(), "version": "v2_short"}
        return {
            "version":     "v2",
            "seq":         data[4],
            "sysid":       data[5],
            "compid":      data[6],
            "msgid":       data[7] | (data[8] << 8) | (data[9] << 16),
            "payload_len": data[1],
            "payload_hex": data[10:10 + data[1]].hex(),
            "raw":         data.hex(),
        }
    return {"raw": data.hex(), "note": f"unknown_stx_{hex(stx)}"}


def handle_packet(pkt):
    from scapy.all import UDP, Raw, IP
    if not pkt.haslayer(UDP) or not pkt.haslayer(Raw):
        return

    src = pkt[IP].src
    dst = pkt[IP].dst
    direction = "SEND" if dst == DRONE_IP else "RECV"

    payload = bytes(pkt[Raw].load)
    parsed  = parse_mavlink(payload)

    entry = {
        "t":         time.time(),
        "direction": direction,
        "src":       src,
        "dst":       dst,
        "sport":     pkt[UDP].sport,
        "dport":     pkt[UDP].dport,
        "parsed":    parsed,
    }

    with lock:
        packets.append(entry)

    msgid = parsed.get("msgid", "?")
    print(f"[{direction}]  {src}:{pkt[UDP].sport} → {dst}:{pkt[UDP].dport}"
          f"  msgid={msgid:<6}  len={parsed.get('payload_len','?')}")


def save_loop():
    while True:
        time.sleep(3)
        with lock:
            snap = list(packets)
        with open(LOG_FILE, "w") as f:
            json.dump(snap, f, indent=2)


if __name__ == "__main__":
    from scapy.all import sniff, conf

    print("[Sniffer v2] Starting network-level capture...")
    print(f"[Sniffer v2] Watching all UDP traffic to/from {DRONE_IP}")
    print(f"[Sniffer v2] Saving to {LOG_FILE}")
    print("[Sniffer v2] Now run your original main.py in another terminal.")
    print("[Sniffer v2] Press Ctrl+C to stop.\n")

    threading.Thread(target=save_loop, daemon=True).start()

    try:
        from scapy.all import dev_from_index
        target_iface = dev_from_index(6) 
        sniff(
            iface = target_iface,
            filter=f"udp and host {DRONE_IP}",
            prn=handle_packet,
            store=False,
            timeout = 80
        )
    except KeyboardInterrupt:
        print("\n[Sniffer v2] Stopped.")
    finally:
        with lock:
            snap = list(packets)
        with open(LOG_FILE, "w") as f:
            json.dump(snap, f, indent=2)
        sends = sum(1 for p in snap if p["direction"] == "SEND")
        recvs = sum(1 for p in snap if p["direction"] == "RECV")
        print(f"[Sniffer v2] Saved {len(snap)} packets "
              f"({sends} SEND, {recvs} RECV) → {LOG_FILE}")
        print("Send back sniffer_v2_log.json and we'll build the rewrite.")