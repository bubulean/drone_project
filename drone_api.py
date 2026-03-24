"""
drone_api.py  —  run with Python 3.11
--------------------------------------
Identical public interface to before — same method names, same validation,
same exceptions. Internally talks to bridge_server.py over a local TCP
socket instead of importing pyhula directly.

Usage
-----
    # Terminal 1 (Python 3.6 venv):
    py -3.6 bridge_server.py

    # Terminal 2 (Python 3.11 venv) — your normal code:
    from drone_api import DroneAPI
    api = DroneAPI()
    api.connect("192.168.100.87")
    api.takeoff()
    ...
"""

import socket
import json

HOST = "127.0.0.1"
PORT = 7625


class DroneAPIError(RuntimeError):
    pass


class DroneAPI:
    def __init__(self, bridge_host=HOST, bridge_port=PORT):
        self._host = bridge_host
        self._port = bridge_port
        self._sock = None
        self._buf  = b""
        self._connect_bridge()

    def _connect_bridge(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._sock.connect((self._host, self._port))
            self._sock.settimeout(30.0)
        except ConnectionRefusedError:
            raise DroneAPIError(
                f"Cannot connect to bridge_server.py at {self._host}:{self._port}.\n"
                "Start it first:  py -3.6 bridge_server.py"
            )

    def _call(self, cmd, **kwargs):
        msg = json.dumps({"cmd": cmd, "args": kwargs}) + "\n"
        self._sock.sendall(msg.encode())
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise DroneAPIError("Bridge server closed the connection.")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        resp = json.loads(line.decode())
        if not resp.get("ok"):
            raise DroneAPIError(f"Bridge error for '{cmd}': {resp.get('error', resp)}")
        return resp.get("result")

    def close(self):
        try:
            self._call("quit")
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def connect(self, ip):
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError(f"ip must be a non-empty string, got: {ip!r}")
        result = self._call("connect", ip=ip)
        if not result:
            raise ConnectionError(f"Failed to connect to drone at {ip}")
        return True

    def get_battery(self):          return self._call("get_battery")
    def get_coordinate(self):       return self._call("get_coordinate")
    def get_yaw(self):              return self._call("get_yaw")
    def get_acceleration(self):     return self._call("get_acceleration")
    def get_velocity(self):         return self._call("get_velocity")
    def get_altitude(self):         return self._call("get_altitude")
    def get_plane_id(self):         return self._call("get_plane_id")
    def get_obstacles(self):        return self._call("get_obstacles")

    def takeoff(self, led=0):       return self._call("takeoff", led=led)
    def land(self, led=0):          return self._call("land", led=led)
    def arm(self):                  return self._call("arm")
    def disarm(self):               return self._call("disarm")

    def hover(self, duration_s, led=0):
        if duration_s <= 0: raise ValueError(f"duration_s must be positive")
        return self._call("hover", duration=duration_s, led=led)

    def _validate_move(self, distance, speed):
        if distance <= 0: raise ValueError(f"distance must be positive, got {distance}")
        if not (0 <= speed <= 100): raise ValueError(f"speed must be 0-100, got {speed}")

    def move_forward(self, distance, speed=100, led=0):
        self._validate_move(distance, speed)
        return self._call("move_forward", distance=distance, speed=speed, led=led)

    def move_back(self, distance, speed=100, led=0):
        self._validate_move(distance, speed)
        return self._call("move_back", distance=distance, speed=speed, led=led)

    def move_left(self, distance, speed=100, led=0):
        self._validate_move(distance, speed)
        return self._call("move_left", distance=distance, speed=speed, led=led)

    def move_right(self, distance, speed=100, led=0):
        self._validate_move(distance, speed)
        return self._call("move_right", distance=distance, speed=speed, led=led)

    def move_up(self, height, speed=100, led=0):
        self._validate_move(height, speed)
        return self._call("move_up", height=height, speed=speed, led=led)

    def move_down(self, height, speed=100, led=0):
        self._validate_move(height, speed)
        return self._call("move_down", height=height, speed=speed, led=led)

    def turn_left(self, angle, led=0):
        if angle <= 0: raise ValueError(f"angle must be positive, got {angle}")
        return self._call("turn_left", angle=angle, led=led)

    def turn_right(self, angle, led=0):
        if angle <= 0: raise ValueError(f"angle must be positive, got {angle}")
        return self._call("turn_right", angle=angle, led=led)

    def fly_straight(self, x, y, z, speed=100, led=0):
        if not (0 <= speed <= 100): raise ValueError(f"speed must be 0-100")
        return self._call("fly_straight", x=x, y=y, z=z, speed=speed, led=led)

    def fly_curve(self, x, y, z, clockwise=False, speed=100, led=0):
        if not (0 <= speed <= 100): raise ValueError(f"speed must be 0-100")
        return self._call("fly_curve", x=x, y=y, z=z, clockwise=clockwise, speed=speed, led=led)

    def orbit(self, radius, led=0):    return self._call("orbit", radius=radius, led=led)
    def spin(self, rotations, led=0):  return self._call("spin", rotations=rotations, led=led)

    def bounce(self, count, height, led=0):
        if count <= 0:  raise ValueError(f"count must be positive")
        if height <= 0: raise ValueError(f"height must be positive")
        return self._call("bounce", count=count, height=height, led=led)

    def somersault(self, direction, led=0):
        if direction not in (0,1,2,3): raise ValueError(f"direction must be 0-3")
        return self._call("somersault", direction=direction, led=led)

    def set_obstacle_avoidance(self, enabled):
        return self._call("set_obstacle_avoidance", enabled=enabled)

    def follow_line(self, fun_id, distance, line_color):
        return self._call("follow_line", fun_id=fun_id, distance=distance, line_color=line_color)

    def identify_tag(self, mode):       return self._call("identify_tag", mode=mode)

    def align_qr_optical(self, qr_id, qr_size=20, angle=0):
        if not (6 <= qr_size <= 30): raise ValueError(f"qr_size must be 6-30")
        return self._call("align_qr_optical", qr_id=qr_id, qr_size=qr_size, angle=angle)

    def recognize_qr_optical(self, qr_id, qr_size=20):
        if not (6 <= qr_size <= 30): raise ValueError(f"qr_size must be 6-30")
        return self._call("recognize_qr_optical", qr_id=qr_id, qr_size=qr_size)

    def recognize_qr_front(self, qr_id):   return self._call("recognize_qr_front", qr_id=qr_id)
    def align_qr_front(self, qr_id):       return self._call("align_qr_front", qr_id=qr_id)

    def track_qr(self, qr_id, duration_s):
        if duration_s <= 0: raise ValueError(f"duration_s must be positive")
        return self._call("track_qr", qr_id=qr_id, duration=duration_s)

    def get_color(self, mode=1):        return self._call("get_color", mode=mode)
    def stream_on(self):                return self._call("stream_on")
    def stream_off(self):               return self._call("stream_off")
    def take_photo(self):               return self._call("take_photo")
    def record_start(self):             return self._call("record_start")
    def record_stop(self):              return self._call("record_stop")
    def flip_stream(self):              return self._call("flip_stream")

    def set_camera_angle(self, direction, angle):
        if not (0 <= angle <= 90): raise ValueError(f"angle must be 0-90")
        return self._call("set_camera_angle", direction=direction, angle=angle)

    def set_led(self, r, g, b, duration_s, mode):
        for val, name in ((r,"r"),(g,"g"),(b,"b")):
            if not (0 <= val <= 255): raise ValueError(f"{name} must be 0-255")
        return self._call("set_led", r=r, g=g, b=b, duration=duration_s, mode=mode)

    def set_electromagnet(self, attach):    return self._call("set_electromagnet", attach=attach)

    def set_clamp(self, action, angle=0):
        if action == 2 and not (0 <= angle <= 180): raise ValueError(f"angle must be 0-180")
        return self._call("set_clamp", action=action, angle=angle)

    def set_qr_positioning(self, enabled):  return self._call("set_qr_positioning", enabled=enabled)

    def fire_laser(self, fire_type, frequency=10, ammo=100):
        if fire_type not in range(6):   raise ValueError(f"fire_type must be 0-5")
        if not (1 <= frequency <= 14):  raise ValueError(f"frequency must be 1-14")
        if not (1 <= ammo <= 255):      raise ValueError(f"ammo must be 1-255")
        return self._call("fire_laser", fire_type=fire_type, frequency=frequency, ammo=ammo)

    def laser_hit(self):    return self._call("laser_hit")

    def __enter__(self):    return self
    def __exit__(self, *_): self.close()