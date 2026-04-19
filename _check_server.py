import sys
sys.path.insert(0, "lib")
from osaurus_lib import is_server_running
print("Server running:", is_server_running())