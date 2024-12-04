import requests
import os

# Function to fetch TLE data from a URL
def fetch_tle_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        tle_data = response.text.splitlines()
        return [tle_data[i:i + 3] for i in range(0, len(tle_data), 3)]  # Group into 3 (Name, Line1, Line2)
    else:
        raise Exception(f"Error fetching TLE data: {response.status_code}")

# Function to save TLE data to a file
def save_tle_data(tle_data, filename):
    with open(filename, 'w') as f:
        for tle_group in tle_data:
            f.write('\n'.join(tle_group) + '\n')

# Fetch TLE data from the provided URLs
tle_urls = {
    "Last 30 Days' Launches": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle',
    "Active Satellites": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
    "Russian ASAT Test Debris (COSMOS 1408)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle',
    "Chinese ASAT Test Debris (FENGYUN 1C)": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle',
    "IRIDIUM 33 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle',
    "COSMOS 2251 Debris": 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle'
}

# Directory to save the TLE files
output_dir = "tle_data"
os.makedirs(output_dir, exist_ok=True)

# Fetch and save TLE data
for name, url in tle_urls.items():
    try:
        tle_data = fetch_tle_data(url)
        filename = name.replace(' ', '_').replace("'", "")
        save_tle_data(tle_data, os.path.join(output_dir, f"{filename}.tle"))
        print(f"Saved TLE data for {name} to {output_dir}/{filename}.tle")
    except Exception as e:
        print(e)