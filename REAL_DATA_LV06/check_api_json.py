import requests
import json

IP = "192.168.0.15"

def check():
    headers = {"Accept": "application/json"}
    try:
        r = requests.get(f"http://{IP}/api/system", headers=headers, timeout=5)
        print(f"Status: {r.status_code}")
        print(f"Headers: {r.headers.get('Content-Type')}")
        if 'json' in r.headers.get('Content-Type', '').lower():
            print(f"JSON: {json.dumps(r.json(), indent=2)}")
        else:
            print(f"RAW: {r.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
