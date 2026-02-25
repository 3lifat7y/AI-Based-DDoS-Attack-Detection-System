import pyshark
import json
import time
import numpy as np
from collections import defaultdict
import argparse
import os

CAPTURE_INTERFACE = "Wi-Fi" 
# CAPTURE_INTERFACE = "Ethernet" 
SEND_INTERVAL = .5

def parse_args():
    p = argparse.ArgumentParser(description='Feature extraction capture')
    p.add_argument('--interval', '-i', type=float, help='Seconds between feature writes (overrides SEND_INTERVAL)')
    p.add_argument('--iface', '-f', type=str, help='Capture interface (overrides CAPTURE_INTERFACE)')
    return p.parse_args()

args = None
try:
    args = parse_args()
except SystemExit:
    args = None

if args:
    if args.interval:
        SEND_INTERVAL = args.interval
    if args.iface:
        CAPTURE_INTERFACE = args.iface
else:
    try:
        env_i = os.environ.get('FEATURE_SEND_INTERVAL')
        if env_i:
            v = float(env_i)
            if v > 0:
                SEND_INTERVAL = v
    except Exception:
        pass

print(f"\nUsing interface: {CAPTURE_INTERFACE}")
print("="*50 + "\n")

def reset_stats():
    return {
        "flow_id": None,
        "src_ip": None,
        "src_port": None,
        "dst_ip": None,
        "dst_port": None,
        "protocol": None,
        "start_time": time.time(),
        "end_time": None,
        "packet_sizes": [],
        "fwd_packet_sizes": [],
        "bwd_packet_sizes": [],
        "packet_times": [],
        "fwd_packet_times": [],
        "bwd_packet_times": [],
        "fwd_packets": 0,
        "bwd_packets": 0,
        "fwd_bytes": 0,
        "bwd_bytes": 0,
        "flags": defaultdict(int),
        "header_lengths": {"fwd": 0, "bwd": 0},
    }

stats = reset_stats()

def process_packet(packet):
    try:
        ts = time.time()
        stats["packet_times"].append(ts)

        if hasattr(packet, 'ip'):
            src = packet.ip.src
            dst = packet.ip.dst
            proto = packet.highest_layer

            if stats["src_ip"] is None:
                stats["src_ip"], stats["dst_ip"], stats["protocol"] = src, dst, proto
                stats["flow_id"] = f"{src}-{dst}-{proto}"

            direction = "fwd" if src == stats["src_ip"] else "bwd"

            length = int(packet.length)
            stats["packet_sizes"].append(length)

            if direction == "fwd":
                stats["fwd_packets"] += 1
                stats["fwd_bytes"] += length
                stats["fwd_packet_sizes"].append(length)
                stats["fwd_packet_times"].append(ts)
            else:
                stats["bwd_packets"] += 1
                stats["bwd_bytes"] += length
                stats["bwd_packet_sizes"].append(length)
                stats["bwd_packet_times"].append(ts)

        if hasattr(packet, 'tcp'):
            flags = packet.tcp.flags
            if "0x0001" in flags: stats["flags"]["FIN"] += 1
            if "0x0002" in flags: stats["flags"]["SYN"] += 1
            if "0x0004" in flags: stats["flags"]["RST"] += 1
            if "0x0008" in flags: stats["flags"]["PSH"] += 1
            if "0x0010" in flags: stats["flags"]["ACK"] += 1
            if "0x0020" in flags: stats["flags"]["URG"] += 1
            if "0x0040" in flags: stats["flags"]["CWE"] += 1
            if "0x0080" in flags: stats["flags"]["ECE"] += 1

    except Exception as e:
        print("Error processing packet:", e)


def send_features():
    stats["end_time"] = time.time()
    duration = max(stats["end_time"] - stats["start_time"], 1e-6)

    def safe_stats(data):
        if not data: return {"max": 0, "min": 0, "mean": 0, "std": 0, "var": 0}
        arr = np.array(data, dtype=float)
        return {
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "var": float(np.var(arr))
        }

    pkt_stats = safe_stats(stats["packet_sizes"])
    fwd_stats = safe_stats(stats["fwd_packet_sizes"])
    bwd_stats = safe_stats(stats["bwd_packet_sizes"])

    def iats(times):
        if len(times) < 2: return {"mean": 0, "std": 0, "max": 0, "min": 0, "total": 0}
        diffs = np.diff(times)
        return {
            "total": float(np.sum(diffs)),
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs)),
            "max": float(np.max(diffs)),
            "min": float(np.min(diffs))
        }

    flow_iat = iats(stats["packet_times"])
    fwd_iat = iats(stats["fwd_packet_times"])
    bwd_iat = iats(stats["bwd_packet_times"])

    features = {
        "Flow ID": stats["flow_id"],
        "Source IP": stats["src_ip"],
        "Source Port": stats.get("src_port", 0),
        "Destination IP": stats["dst_ip"],
        "Destination Port": stats.get("dst_port", 0),
        "Protocol": stats["protocol"],
        "Timestamp": int(time.time()),
        "Flow Duration": round(duration, 6),

        "Total Fwd Packets": stats["fwd_packets"],
        "Total Backward Packets": stats["bwd_packets"],
        "Total Length of Fwd Packets": stats["fwd_bytes"],
        "Total Length of Bwd Packets": stats["bwd_bytes"],

        "Fwd Packet Length Max": fwd_stats["max"],
        "Fwd Packet Length Min": fwd_stats["min"],
        "Fwd Packet Length Mean": fwd_stats["mean"],
        "Fwd Packet Length Std": fwd_stats["std"],

        "Bwd Packet Length Max": bwd_stats["max"],
        "Bwd Packet Length Min": bwd_stats["min"],
        "Bwd Packet Length Mean": bwd_stats["mean"],
        "Bwd Packet Length Std": bwd_stats["std"],

        "Flow Bytes/s": round((stats["fwd_bytes"] + stats["bwd_bytes"]) / duration, 3),
        "Flow Packets/s": round((stats["fwd_packets"] + stats["bwd_packets"]) / duration, 3),

        "Flow IAT Mean": flow_iat["mean"],
        "Flow IAT Std": flow_iat["std"],
        "Flow IAT Max": flow_iat["max"],
        "Flow IAT Min": flow_iat["min"],

        "Fwd IAT Total": fwd_iat["total"],
        "Fwd IAT Mean": fwd_iat["mean"],
        "Fwd IAT Std": fwd_iat["std"],
        "Fwd IAT Max": fwd_iat["max"],
        "Fwd IAT Min": fwd_iat["min"],

        "Bwd IAT Total": bwd_iat["total"],
        "Bwd IAT Mean": bwd_iat["mean"],
        "Bwd IAT Std": bwd_iat["std"],
        "Bwd IAT Max": bwd_iat["max"],
        "Bwd IAT Min": bwd_iat["min"],

        "FIN Flag Count": stats["flags"]["FIN"],
        "SYN Flag Count": stats["flags"]["SYN"],
        "RST Flag Count": stats["flags"]["RST"],
        "PSH Flag Count": stats["flags"]["PSH"],
        "ACK Flag Count": stats["flags"]["ACK"],
        "URG Flag Count": stats["flags"]["URG"],
        "CWE Flag Count": stats["flags"]["CWE"],
        "ECE Flag Count": stats["flags"]["ECE"],

        "Average Packet Size": pkt_stats["mean"],
        "Avg Fwd Segment Size": fwd_stats["mean"],
        "Avg Bwd Segment Size": bwd_stats["mean"],
        "Down/Up Ratio": round(stats["bwd_packets"] / max(stats["fwd_packets"], 1), 3)
    }

    json_string = json.dumps(features, indent=4)
    try:
        with open("extracted_features.json", "w", encoding="utf-8") as f:
            f.write(json_string + "\n") 
        print("\nUpdated extracted_features.json:")
        print(json_string)  
    except Exception as e:
        print("Failed to write JSON:", e)


print(f"Starting live capture from {CAPTURE_INTERFACE} (updating every {SEND_INTERVAL} seconds)\n")

capture = pyshark.LiveCapture(interface=CAPTURE_INTERFACE)
last_send_time = time.time()

try:
    for packet in capture.sniff_continuously():
        process_packet(packet)

        if time.time() - last_send_time >= SEND_INTERVAL:
            send_features()
            stats = reset_stats()
            last_send_time = time.time()

except KeyboardInterrupt:
    print("\nCapture stopped.")
finally:
    capture.close()
    