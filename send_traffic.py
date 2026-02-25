import json
import random
import time
import requests
import argparse
import sys

TEST_SET_PATH = r"D:\university\LEVEL 3\semster 1\coding\Work-based\features_by_label.json"
TARGET_SERVER = "http://192.168.1.11:5000"
SEND_INTERVAL = 0.5

def parse_args():
    p = argparse.ArgumentParser(description='Send traffic simulator')
    p.add_argument('--interval', '-i', type=float, help='Seconds between sends (overrides SEND_INTERVAL)')
    p.add_argument('--target', '-t', type=str, help='Target server URL (overrides TARGET_SERVER)')
    p.add_argument('--count', '-c', type=int, help='Number of packets to send (default: infinite)')
    p.add_argument('--list', action='store_true', help='List available attack types then exit')
    return p.parse_args()

def load_dataset():
    with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)["all_data"]

def main():
    args = parse_args()
    global SEND_INTERVAL, TARGET_SERVER
    if args.interval:
        SEND_INTERVAL = args.interval
    if args.target:
        TARGET_SERVER = args.target

    print(f"Loading attack packets... (send interval={SEND_INTERVAL}s, target={TARGET_SERVER})")
    dataset = load_dataset()
    print(f"Loaded {len(dataset):,} packets\n")

    labels = {}
    for p in dataset:
        lbl = p.get("Label", "Unknown")
        labels[lbl] = labels.get(lbl, 0) + 1

    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)

    print("="*65)
    print("           AVAILABLE ATTACK TYPES")
    print("="*65)
    for i, (label, count) in enumerate(sorted_labels, 1):
        print(f"  {i}. {label:<18} → {count:>5} samples")
    print("="*65)

    if args.list:
        print("Attack types listed above.")
        sys.exit(0)

    while True:
        choice_raw = input(f"\nSelect attack by number or name (1-{len(sorted_labels)}): ")
        if not choice_raw:
            print("Please choose a valid number or label name.")
            continue
        try:
            choice = int(choice_raw)
            if 1 <= choice <= len(sorted_labels):
                selected = sorted_labels[choice-1][0]
                break
            else:
                print(f"Invalid number: choose 1-{len(sorted_labels)}")
                continue
        except ValueError:
            matches = [lbl for lbl, _cnt in sorted_labels if lbl.lower() == choice_raw.lower()]
            if len(matches) == 1:
                selected = matches[0]
                break
            else:
                print("No matching label found. Try exact label name or choose a number.")
                continue

    packets = [{k: v for k, v in p.items() if k != "Label"}
            for p in dataset if p.get("Label") == selected]

    if len(packets) == 0:
        print(f"\nNo packets for selected label ({selected}).")
        sys.exit(1)

    print(f"\nStarting {selected} attack - {len(packets)} packets ready")
    print(f"Sending one packet every {SEND_INTERVAL} seconds → Press Ctrl+C to stop\n")

    sent = 0
    try:
        while True:
            packet = random.choice(packets)
            try:
                requests.post(f"{TARGET_SERVER}/api/receive-packet", json=packet, timeout=3)
                print(f"[{time.strftime('%H:%M:%S')}] Sent → {selected}")
            except Exception:
                print(f"[{time.strftime('%H:%M:%S')}] Failed")
            sent += 1
            if args.count and sent >= args.count:
                print(f"\nSent {sent} packets as requested. Stopping.")
                break
            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("\n\nAttack stopped.")

if __name__ == "__main__":
    main()