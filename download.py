import requests
import os

os.makedirs("data", exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/pdf",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.justice.gov/"
}

session = requests.Session()
session.headers.update(headers)

with open("links.txt") as f:
    links = f.read().splitlines()

print(f"Downloading {len(links)} files...")

for i, url in enumerate(links):
    try:
        print(f"{i+1}/{len(links)}")

        r = session.get(url, allow_redirects=True)

        # check if actual PDF
        if b"%PDF" not in r.content[:5]:
            print("❌ Not PDF (blocked), retrying...")

            # try again without session (fallback)
            r = requests.get(url, headers=headers)

            if b"%PDF" not in r.content[:5]:
                print("❌ Still not PDF, skipping")
                continue

        filename = f"data/file_{i}.pdf"

        with open(filename, "wb") as f:
            f.write(r.content)

    except Exception as e:
        print("Error:", e)

print("Done")
