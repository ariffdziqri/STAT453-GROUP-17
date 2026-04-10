import requests

# Step 1: Get company's CIK number (unique ID for every company)
def get_cik(ticker):
    url = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&enddt=2024-01-01&forms=10-K".format(ticker)
    
    # Simpler approach: use the ticker->CIK mapping file
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(tickers_url, headers={"User-Agent": "Group 17 aismadi@wisc.edu"})
    data = response.json()
    
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            # CIK must be zero-padded to 10 digits
            return str(entry["cik_str"]).zfill(10)
    return None

# Step 2: Get list of filings for that company
def get_filings(cik):
    url = f"https://data.sec.gov/submissions/{cik}.json"
    response = requests.get(url, headers={"User-Agent": "Group 17 aismadi@wisc.edu"})
    return response.json()

# Step 3: Find the latest 10-K accession number
def get_latest_10k(filings):
    recent = filings["filings"]["recent"]
    forms = recent["form"]
    accession_numbers = recent["accessionNumber"]
    
    for i, form in enumerate(forms):
        if form == "10-K":
            return accession_numbers[i].replace("-", "")  # clean format

# Step 4: Fetch the actual document
def fetch_10k_text(cik, accession_number):
    # Get the filing index first
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{accession_number}-index.htm"
    response = requests.get(index_url, headers={"User-Agent": "Group 17 aismadi@wisc.edu"})
    return response.text

# --- Put it all together ---
cik = get_cik("TSLA")
filings = get_filings(cik)
accession = get_latest_10k(filings)
text = fetch_10k_text(cik, accession)
print(text[:500])  # preview first 500 chars