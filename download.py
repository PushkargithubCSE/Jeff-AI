import requests
import os

os.makedirs("data", exist_ok=True)

with open("links.txt") as f:
    links = f.read().splitlines()

print(f"Downloading {len(links)} PDFs")

for i, url in enumerate(links):
    try:
        print(f"{i+1}/{len(links)}")

        r = requests.get(url)

        filename = f"data/file_{i}.pdf"

        with open(filename, "wb") as f:
            f.write(r.content)

    except Exception as e:
        print("Error:", e)

print("Done")
