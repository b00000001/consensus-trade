import socket
import os
from dotenv import load_dotenv

load_dotenv()

def check_conn(host, port, name):
    print(f"Checking {name} at {host}:{port}...")
    try:
        with socket.create_connection((host, int(port)), timeout=3):
            print(f"SUCCESS: {name} is reachable.")
            return True
    except Exception as e:
        print(f"FAILED: {name} is unreachable. Error: {e}")
        return False

redis_host = os.getenv("REDIS_HOST", "192.168.1.22")
redis_port = os.getenv("REDIS_PORT", "6379")

ollama_url = os.getenv("OLLAMA_BASE", "http://192.168.1.207:11434")
# Parse host and port from URL
from urllib.parse import urlparse
parsed = urlparse(ollama_url)
ollama_host = parsed.hostname or "192.168.1.207"
ollama_port = parsed.port or 11434

check_conn(redis_host, redis_port, "Redis")
check_conn(ollama_host, ollama_port, "Ollama")
