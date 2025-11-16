"this one has to go to the scripts"

def process_all_pdfs_in_directory(folder_path, pdf_folder, index):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file_name in tqdm(json_files, desc="📦 Processing JSON files", unit="file"):
        json_path = os.path.join(folder_path, json_file_name)
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check if JSON is a list of entries
        if not isinstance(data, list):
            tqdm.write(f"⚠️ Skipping {json_file_name}: JSON is not a list")
            continue

        updated = False

        for entry in tqdm(data, desc=f"📝 {json_file_name}", leave=False):
            filename = entry.get('file')
            if not filename:
                continue

            pdf_path = os.path.join(pdf_folder, filename)
            text = parse_pdf_first_page(pdf_path)

            if not text or "Error reading PDF" in text:
                tqdm.write(f"❌ Failed to extract text from: {filename}")
                continue

            if index == 'SP500':
                date, quarter, year, ticker, company_name = extract_company_ticker_date_US(filename, text)
            elif index == 'STOXX600':
                date, quarter, year, ticker, company_name = extract_company_ticker_date_EU(filename, text)

            # Update the entry with new metadata
            entry['date'] = str(date) if date else None
            entry['quarter'] = quarter
            entry['year'] = year
            entry['ticker'] = ticker
            entry['company_name'] = company_name
            updated = True

        if updated:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)