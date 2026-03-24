"""
bridge_server.py  —  run with Python 3.6 + pyhula
--------------------------------------------------
Listens on localhost:7625 for JSON commands, executes them via pyhula,
and sends back JSON results. Keeps running until told to quit.

Start it ONCE before your main.py:
    py -3.6 bridge_server.py

Then main.py (Python 3.11) uses BridgeClient to talk to it.
"""

import socket
import json
import traceback
import pyhula

HOST = "127.0.0.1"
PORT = 7625

api = None       # initialised on first "connect" command
connected = False


def handle(cmd, args):
    global api, connected

    name = cmd

    # ── Connection ────────────────────────────────────────────────────────────
    if name == "connect":
        # Reuse existing connection — pyhula cannot rebind the same UDP port
        # twice in the same process, so we must never create a second UserApi.
        if connected and api is not None:
            print("[Bridge] Already connected, reusing existing session.")
            return {"ok": True, "result": True}
        api = pyhula.UserApi()
        ok = api.connect(args["ip"])
        if ok:
            connected = True
        return {"ok": bool(ok), "result": bool(ok)}

    if api is None or not connected:
        return {"ok": False, "error": "not_connected"}

    # ── Flight ────────────────────────────────────────────────────────────────
    if name == "takeoff":
        return {"ok": True, "result": api.single_fly_takeoff(args.get("led", 0))}
    if name == "land":
        return {"ok": True, "result": api.single_fly_touchdown(args.get("led", 0))}
    if name == "hover":
        return {"ok": True, "result": api.single_fly_hover_flight(args["duration"], args.get("led", 0))}
    if name == "arm":
        return {"ok": True, "result": api.plane_fly_arm()}
    if name == "disarm":
        return {"ok": True, "result": api.plane_fly_disarm()}

    # ── Movement ──────────────────────────────────────────────────────────────
    if name == "move_forward":
        return {"ok": True, "result": api.single_fly_forward(args["distance"], args.get("speed", 100), args.get("led", 0))}
    if name == "move_back":
        return {"ok": True, "result": api.single_fly_back(args["distance"], args.get("speed", 100), args.get("led", 0))}
    if name == "move_left":
        return {"ok": True, "result": api.single_fly_left(args["distance"], args.get("speed", 100), args.get("led", 0))}
    if name == "move_right":
        return {"ok": True, "result": api.single_fly_right(args["distance"], args.get("speed", 100), args.get("led", 0))}
    if name == "move_up":
        return {"ok": True, "result": api.single_fly_up(args["height"], args.get("speed", 100), args.get("led", 0))}
    if name == "move_down":
        return {"ok": True, "result": api.single_fly_down(args["height"], args.get("speed", 100), args.get("led", 0))}
    if name == "turn_left":
        return {"ok": True, "result": api.single_fly_turnleft(args["angle"], args.get("led", 0))}
    if name == "turn_right":
        return {"ok": True, "result": api.single_fly_turnright(args["angle"], args.get("led", 0))}
    if name == "fly_straight":
        return {"ok": True, "result": api.single_fly_straight_flight(args["x"], args["y"], args["z"], args.get("speed", 100), args.get("led", 0))}
    if name == "fly_curve":
        return {"ok": True, "result": api.single_fly_curvilinearFlight(args["x"], args["y"], args["z"], not args.get("clockwise", False), args.get("speed", 100), args.get("led", 0))}
    if name == "orbit":
        return {"ok": True, "result": api.single_fly_radius_around(args["radius"], args.get("led", 0))}
    if name == "spin":
        return {"ok": True, "result": api.single_fly_autogyration360(args["rotations"], args.get("led", 0))}
    if name == "bounce":
        return {"ok": True, "result": api.single_fly_bounce(args["count"], args["height"], args.get("led", 0))}
    if name == "somersault":
        return {"ok": True, "result": api.single_fly_somersault(args["direction"], args.get("led", 0))}

    # ── Telemetry ─────────────────────────────────────────────────────────────
    if name == "get_battery":
        return {"ok": True, "result": api.get_battery()}
    if name == "get_coordinate":
        return {"ok": True, "result": api.get_coordinate()}
    if name == "get_yaw":
        return {"ok": True, "result": api.get_yaw()}
    if name == "get_acceleration":
        return {"ok": True, "result": api.get_accelerated_speed()}
    if name == "get_velocity":
        return {"ok": True, "result": api.get_plane_speed()}
    if name == "get_altitude":
        return {"ok": True, "result": api.get_plane_distance()}
    if name == "get_plane_id":
        return {"ok": True, "result": api.get_plane_id()}
    if name == "get_obstacles":
        return {"ok": True, "result": api.Plane_getBarrier()}

    # ── Camera / video ────────────────────────────────────────────────────────
    if name == "stream_on":
        return {"ok": True, "result": api.Plane_cmd_swith_rtp(0)}
    if name == "stream_off":
        return {"ok": True, "result": api.Plane_cmd_swith_rtp(1)}
    if name == "take_photo":
        return {"ok": True, "result": api.Plane_fly_take_photo()}
    if name == "record_start":
        return {"ok": True, "result": api.Plane_cmd_switch_video(0)}
    if name == "record_stop":
        return {"ok": True, "result": api.Plane_cmd_switch_video(1)}
    if name == "set_camera_angle":
        return {"ok": True, "result": api.Plane_cmd_camera_angle(args["direction"], args["angle"])}
    if name == "flip_stream":
        api.single_fly_flip_rtp()
        return {"ok": True}

    # ── Obstacle avoidance ────────────────────────────────────────────────────
    if name == "set_obstacle_avoidance":
        return {"ok": True, "result": api.single_fly_barrier_aircraft(args["enabled"])}

    # ── Vision / QR ──────────────────────────────────────────────────────────
    if name == "follow_line":
        return {"ok": True, "result": api.single_fly_Line_walking(args["fun_id"], args["distance"], args["line_color"])}
    if name == "identify_tag":
        return {"ok": True, "result": api.single_fly_AiIdentifies(args["mode"])}
    if name == "align_qr_optical":
        return {"ok": True, "result": api.single_fly_Optical_flow_alignment(args["qr_id"], args.get("qr_size", 20), args.get("angle", 0))}
    if name == "recognize_qr_optical":
        return {"ok": True, "result": api.single_fly_Optical_flow_recognition(args["qr_id"], args.get("qr_size", 20))}
    if name == "recognize_qr_front":
        return {"ok": True, "result": api.single_fly_Anticipatory_recognition(args["qr_id"])}
    if name == "align_qr_front":
        return {"ok": True, "result": api.single_fly_Proactive_alignment(args["qr_id"])}
    if name == "track_qr":
        return {"ok": True, "result": api.single_fly_track_Qrcode(args["qr_id"], args["duration"])}
    if name == "get_color":
        return {"ok": True, "result": api.single_fly_getColor(args.get("mode", 1))}

    # ── Peripherals ───────────────────────────────────────────────────────────
    if name == "set_led":
        return {"ok": True, "result": api.single_fly_lamplight(args["r"], args["g"], args["b"], args["duration"], args["mode"])}
    if name == "set_electromagnet":
        return {"ok": True, "result": api.Plane_cmd_electromagnet(2 if args["attach"] else 3)}
    if name == "set_clamp":
        return {"ok": True, "result": api.Plane_cmd_clamp(args["action"], args.get("angle", 0))}
    if name == "set_qr_positioning":
        return {"ok": True, "result": api.Plane_cmd_switch_QR(0 if args["enabled"] else 1)}
    if name == "fire_laser":
        return {"ok": True, "result": api.plane_fly_generating(args["fire_type"], args.get("frequency", 10), args.get("ammo", 100))}
    if name == "laser_hit":
        return {"ok": True, "result": api.plane_fly_laser_receiving()}

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    if name == "quit":
        return {"ok": True, "quit": True}

    return {"ok": False, "error": "unknown_command: " + name}


def run():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("[Bridge] Listening on {}:{} — waiting for main.py".format(HOST, PORT))

    while True:
        conn, addr = server.accept()
        print("[Bridge] Connected from {}".format(addr))
        buf = b""
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                # Messages are newline-delimited JSON
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode())
                        cmd  = msg.get("cmd", "")
                        args = msg.get("args", {})
                        result = handle(cmd, args)
                    except Exception:
                        result = {"ok": False, "error": traceback.format_exc()}

                    resp = json.dumps(result) + "\n"
                    conn.sendall(resp.encode())

                    if result.get("quit"):
                        print("[Bridge] Quit command received.")
                        conn.close()
                        server.close()
                        return
        except Exception:
            print("[Bridge] Connection error:\n" + traceback.format_exc())
        finally:
            conn.close()
        print("[Bridge] Client disconnected — waiting for next connection.")


if __name__ == "__main__":
    run()