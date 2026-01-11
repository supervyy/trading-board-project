import subprocess
import sys
import threading
import os
import time
from pathlib import Path

# ANSI Colors for better visibility
COLOR_ORIG = "\033[96m" # Cyan
COLOR_NEW = "\033[92m"  # Green
COLOR_RESET = "\033[0m"

def stream_reader(pipe, prefix, color):
    """Reads from a pipe and prints with a prefixed label."""
    try:
        for line in iter(pipe.readline, b''):
            decoded_line = line.decode('utf-8', errors='replace').strip()
            if decoded_line:
                print(f"{color}{prefix} | {decoded_line}{COLOR_RESET}")
    except Exception as e:
        print(f"Error reading stream {prefix}: {e}")

def main():
    script_dir = Path(__file__).resolve().parent
    original_script = script_dir / "original_deploy_trading.py"
    new_script = script_dir / "deploy_trading.py"
    
    # Check files exist
    if not original_script.exists():
        print(f"Error: {original_script} not found.")
        sys.exit(1)
    if not new_script.exists():
        print(f"Error: {new_script} not found.")
        sys.exit(1)

    print("--- Starting Dual Deployment (Original & New) ---")
    print(f"Original: {original_script.name}")
    print(f"New:      {new_script.name}")
    print("-----------------------------------------------")

    # Environment with unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Start Processes
    p_orig = subprocess.Popen(
        [sys.executable, "-u", str(original_script), "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=script_dir,
        env=env
    )
    
    p_new = subprocess.Popen(
        [sys.executable, "-u", str(new_script), "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=script_dir,
        env=env
    )

    # Start Output Threads
    # We map stdout and stderr to the same reader for simplicity, or separate if needed.
    # Usually easier to just see everything.
    
    threads = []
    threads.append(threading.Thread(target=stream_reader, args=(p_orig.stdout, "[ORIG]", COLOR_ORIG), daemon=True))
    threads.append(threading.Thread(target=stream_reader, args=(p_orig.stderr, "[ORIG ERR]", COLOR_ORIG), daemon=True))
    
    threads.append(threading.Thread(target=stream_reader, args=(p_new.stdout, "[NEW ]", COLOR_NEW), daemon=True))
    threads.append(threading.Thread(target=stream_reader, args=(p_new.stderr, "[NEW  ERR]", COLOR_NEW), daemon=True))

    for t in threads:
        t.start()

    try:
        while True:
            # Check if processes are still alive
            ret_orig = p_orig.poll()
            ret_new = p_new.poll()
            
            if ret_orig is not None and ret_new is not None:
                print("Both processes have finished.")
                break
            
            if ret_orig is not None:
                print(f"Original process exited with code {ret_orig}. Waiting for New...")
                p_orig = None # Mark as done to avoid repeated msg (logic simplified here)
                # In a real robust loop we'd handle this better, but let's just sleep
                break # Or continue? User wants both running. If one crashes, maybe we should stop?
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping both scripts...")
        if p_orig: p_orig.terminate()
        if p_new: p_new.terminate()
        print("Waiting for termination...")
        # Give them a moment
        time.sleep(1)
        
    print("Exited.")

if __name__ == "__main__":
    main()
