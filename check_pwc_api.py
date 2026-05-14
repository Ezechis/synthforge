import requests

# Test the API and show exactly what we get back
url = "https://paperswithcode.com/api/v1/papers/"
params = {"q": "prompt engineering", "page": 1, "items_per_page": 5}
headers = {"User-Agent": "PromptForge-Research-Bot/1.0", "Accept": "application/json"}

try:
    r = requests.get(url, params=params, headers=headers, timeout=20)
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('content-type', 'unknown')}")
    print(f"Body preview: {r.text[:500]}")
except Exception as exc:
    print(f"Error: {exc}")