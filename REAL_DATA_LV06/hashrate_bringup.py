import sys
import os
import time

# Add the SDK path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "LV06_Universal_Config_Guide")))
from universal_lv06_driver import LV06StratumServer

HOST = "0.0.0.0"
PORT = 3333

def bringup():
    print("BRINGUP: Testing steady-state hashrate at D=512")
    server = LV06StratumServer(HOST, PORT)
    server.daemon = True
    server.start()
    
    print("BRINGUP: Waiting for connection...")
    while not server.connection_active:
        time.sleep(1)
    
    print("BRINGUP: Connection active. Waiting for handshake...")
    if not server.handshake_complete.wait(timeout=30):
        print("BRINGUP: Handshake TIMEOUT")
        return
    
    print("BRINGUP: Handshake OK. Sending Job D=512.0")
    # Using the trusted inject_rate but with constant diff
    # We'll use u=0.5 so D = 512 / (0.5+0.05) ~= 930
    # Actually let's just use 512 directly
    server.set_difficulty(512.0)
    
    # Send one job and WAIT
    from universal_lv06_driver import time as sdk_time
    header = "01000000" + "01" + "00" * 32 + "ffffffff" + "10" 
    coinb1 = header + "04" + "00"*8 + "0a" + "00"*10
    coinb2 = "ffffffff" + "01" + "00f2052a01000000" + "00"*8
    ntime = hex(int(time.time()))[2:].zfill(8)
    
    params = ["1", "0"*64, coinb1, coinb2, [], "20000000", "1f00ffff", ntime, True]
    server._send(server.client_conn, {"id": None, "method": "mining.notify", "params": params})
    
    print("BRINGUP: Job sent. Monitoring for 60 seconds...")
    start = time.time()
    while time.time() - start < 60:
        shares = server.harvest_state()
        if shares:
            print(f"BRINGUP: RECEIVED {len(shares)} SHARES!")
        time.sleep(1)
        
    server.stop()

if __name__ == "__main__":
    bringup()
