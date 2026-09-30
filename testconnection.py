"""Standalone connectivity test - run this directly, no Streamlit involved.
Usage: python test_connection.py YOUR_GROQ_KEY
"""
import sys

if len(sys.argv) < 2:
    print("Usage: python test_connection.py YOUR_GROQ_API_KEY")
    sys.exit(1)

api_key = sys.argv[1]

print("1. Testing basic internet access...")
try:
    import urllib.request
    urllib.request.urlopen("https://www.google.com", timeout=5)
    print("   OK - internet reachable")
except Exception as e:
    print(f"   FAILED: {e}")
    print("   -> No general internet access from this machine/terminal.")
    sys.exit(1)

print("2. Testing DNS + HTTPS reach to Groq API host...")
try:
    import urllib.request
    try:
        urllib.request.urlopen("https://api.groq.com", timeout=5)
        print("   OK - api.groq.com reachable")
    except urllib.error.HTTPError as he:
        # Any HTTP response (even 403/404) proves the connection itself succeeded -
        # TLS handshake completed and the server replied. Only a timeout/refused
        # connection at this stage is an actual network block.
        print(f"   OK - api.groq.com reachable (got HTTP {he.code}, which is expected for a bare root request)")
except Exception as e:
    print(f"   FAILED: {e}")
    print("   -> api.groq.com is being blocked (firewall/proxy/antivirus/VPN).")
    sys.exit(1)

print("3. Testing actual Groq API call with your key...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say OK"}],
    )
    print("   OK - Groq responded:", resp.choices[0].message.content)
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")