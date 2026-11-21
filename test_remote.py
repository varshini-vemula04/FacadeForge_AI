
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

values = {}

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

url = values["FACADEFORGE_REMOTE_URL"].rstrip("/") + "/health"
token = values["FACADEFORGE_REMOTE_TOKEN"]

request = Request(
    url,
    headers={"Authorization": f"Bearer {token}"}
)

try:
    with urlopen(request, timeout=25) as response:
        print("HTTP:", response.status)
        print("Response:", response.read().decode())

except HTTPError as e:
    print("HTTP:", e.code)
    print("Response:", e.read().decode())

except Exception as e:
    print("Error:", type(e).__name__, str(e))
