import stun
import requests

nat_type, external_ip, external_port = stun.get_ip_info()
print(f"STUN IP: {external_ip}")

# Simplified headers to avoid Cloudflare detection
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    'Referer': 'https://ipapi.co/',
    'Origin': 'https://ipapi.co'
}

try:
    response = requests.get("https://ipapi.co/json/", headers=headers, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Text (first 200 chars): {response.text[:200]}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Country: {data.get('country', 'Unknown')}")
        print(f"Country Code: {data.get('country_code', 'Unknown')}")
    else:
        print(f"Error: HTTP {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")