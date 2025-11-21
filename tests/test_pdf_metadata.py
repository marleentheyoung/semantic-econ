#!/usr/bin/env python3
"""
Comprehensive test of PDF metadata extraction functions.

Tests both filename parsing and PDF first-page parsing for the RELL file.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_measurement.data.pdf_metadata import (
    parse_pdf_first_page,
    extract_metadata_from_pdf,
    _extract_company_ticker_date_eu,
    build_metadata_lookup,
)

# The specific file that's causing errors
TEST_FILENAME = "RELL - Q4 2010 Earnings Call 17February2011 400 AM ET CORRECTED TRANSCRIPT - 17-Feb-11.pdf"
TEST_PDF_PATH = "/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600/" + TEST_FILENAME

print("="*80)
print("COMPREHENSIVE TEST: PDF Metadata Extraction")
print("="*80)

print(f"\n📄 Testing file:")
print(f"   {TEST_FILENAME}")
print(f"\n📁 Full path:")
print(f"   {TEST_PDF_PATH}")

# ============================================================================
# TEST 1: Check if file exists
# ============================================================================
print("\n" + "="*80)
print("TEST 1: File Existence")
print("="*80)

pdf_path = Path(TEST_PDF_PATH)
if pdf_path.exists():
    print(f"✅ File exists")
    print(f"   Size: {pdf_path.stat().st_size:,} bytes")
else:
    print(f"❌ File NOT found!")
    print(f"\n⚠️  Cannot continue with PDF parsing tests.")
    print(f"   Please verify the path is correct.")
    sys.exit(1)

# ============================================================================
# TEST 2: Extract first page text
# ============================================================================
print("\n" + "="*80)
print("TEST 2: PDF First Page Extraction")
print("="*80)

print("\nCalling: parse_pdf_first_page()")

try:
    first_page_text = parse_pdf_first_page(TEST_PDF_PATH)
    
    if first_page_text is None:
        print("❌ Function returned None - PDF could not be read")
    elif len(first_page_text) == 0:
        print("❌ Function returned empty string")
    else:
        print(f"✅ Successfully extracted text")
        print(f"   Length: {len(first_page_text):,} characters")
        print(f"\n   First 500 characters:")
        print("   " + "-"*76)
        print("   " + first_page_text[:500].replace("\n", "\n   "))
        print("   " + "-"*76)
        print("   ...")
        
except Exception as e:
    print(f"❌ Exception occurred: {e}")
    first_page_text = None

# ============================================================================
# TEST 3: Internal EU extraction function
# ============================================================================
print("\n" + "="*80)
print("TEST 3: _extract_company_ticker_date_eu() - Internal Function")
print("="*80)

if first_page_text is not None:
    print("\nCalling: _extract_company_ticker_date_eu(filename, text)")
    
    try:
        date, quarter, year, ticker, company_name = _extract_company_ticker_date_eu(
            TEST_FILENAME, 
            first_page_text
        )
        
        print("\n✅ Function returned (no crash)")
        print("\nExtracted values:")
        print(f"   date:         {repr(date)}")
        print(f"   quarter:      {repr(quarter)}")
        print(f"   year:         {repr(year)}")
        print(f"   ticker:       {repr(ticker)}")
        print(f"   company_name: {repr(company_name)}")
        
        # Check for None values
        print("\nNone value check:")
        none_fields = []
        for field_name, value in [
            ("date", date),
            ("quarter", quarter),
            ("year", year),
            ("ticker", ticker),
            ("company_name", company_name),
        ]:
            is_none = value is None
            symbol = "⚠️ " if is_none else "✅"
            print(f"   {symbol} {field_name}: {is_none}")
            if is_none:
                none_fields.append(field_name)
        
        if none_fields:
            print(f"\n⚠️  WARNING: {len(none_fields)} field(s) are None: {', '.join(none_fields)}")
            print(f"   These could cause .upper() errors if not handled!")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Skipping - no PDF text available")

# ============================================================================
# TEST 4: Public API function
# ============================================================================
print("\n" + "="*80)
print("TEST 4: extract_metadata_from_pdf() - Public API")
print("="*80)

print("\nCalling: extract_metadata_from_pdf(pdf_path, index='STOXX600')")

try:
    metadata = extract_metadata_from_pdf(TEST_PDF_PATH, index="STOXX600")
    
    print("\n✅ Function returned (no crash)")
    print(f"\nResult type: {type(metadata)}")
    print(f"\nMetadata dictionary:")
    
    for key, value in metadata.items():
        value_repr = repr(value)
        is_none = value is None
        symbol = "⚠️ " if is_none else "✅"
        print(f"   {symbol} {key}: {value_repr}")
    
    # Count None values
    none_count = sum(1 for v in metadata.values() if v is None)
    print(f"\n   Total fields: {len(metadata)}")
    print(f"   None values: {none_count}")
    
    if none_count > 0:
        print(f"\n⚠️  WARNING: {none_count} field(s) are None")
        
except Exception as e:
    print(f"❌ Exception occurred: {e}")
    print(f"\n   Full traceback:")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: Test .upper() on each field
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Simulate .upper() Calls")
print("="*80)

print("\nSimulating what happens if code calls .upper() on each field:")

if 'metadata' in locals():
    for key, value in metadata.items():
        print(f"\n   Testing: {key}.upper()")
        
        if value is None:
            print(f"      ❌ Value is None - calling .upper() would crash!")
            try:
                result = value.upper()
            except AttributeError as e:
                print(f"      💥 Confirmed crash: {e}")
        elif isinstance(value, str):
            try:
                result = value.upper()
                print(f"      ✅ Success: {repr(result)}")
            except Exception as e:
                print(f"      ❌ Unexpected error: {e}")
        else:
            print(f"      ℹ️  Not a string (type: {type(value).__name__}), skipping")

# ============================================================================
# TEST 6: build_metadata_lookup() for the folder
# ============================================================================
print("\n" + "="*80)
print("TEST 6: build_metadata_lookup() - Folder-Level Function")
print("="*80)

print("\nCalling: build_metadata_lookup(folder, index='STOXX600')")
print("⚠️  This will process ALL PDFs in the folder - may take time!")

pdf_folder = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600")

try:
    # Limit to just our test file for speed
    print(f"\nProcessing folder: {pdf_folder}")
    print(f"Note: This processes ALL PDFs in the folder")
    
    import time
    start = time.time()
    
    metadata_lookup = build_metadata_lookup(str(pdf_folder), index="STOXX600")
    
    elapsed = time.time() - start
    
    print(f"\n✅ Function completed in {elapsed:.1f} seconds")
    print(f"\n   Total PDFs processed: {len(metadata_lookup)}")
    
    # Check our specific file
    if TEST_FILENAME in metadata_lookup:
        print(f"\n   ✅ Found our test file in lookup!")
        our_metadata = metadata_lookup[TEST_FILENAME]
        print(f"\n   Metadata for {TEST_FILENAME}:")
        for key, value in our_metadata.items():
            is_none = value is None
            symbol = "⚠️ " if is_none else "✅"
            print(f"      {symbol} {key}: {repr(value)}")
    else:
        print(f"\n   ❌ Test file NOT found in lookup!")
        print(f"\n   Available files (first 10):")
        for i, filename in enumerate(list(metadata_lookup.keys())[:10]):
            print(f"      {i+1}. {filename}")
    
    # Statistics on None values
    print(f"\n   Statistics across all files:")
    none_stats = {
        "ticker": 0,
        "quarter": 0,
        "year": 0,
        "company_name": 0,
        "date": 0,
    }
    
    for file_meta in metadata_lookup.values():
        for field in none_stats.keys():
            if file_meta.get(field) is None:
                none_stats[field] += 1
    
    print(f"\n   None value counts:")
    total_files = len(metadata_lookup)
    for field, count in none_stats.items():
        pct = (count / total_files * 100) if total_files > 0 else 0
        symbol = "⚠️ " if count > 0 else "✅"
        print(f"      {symbol} {field}: {count:,} / {total_files:,} ({pct:.1f}%)")
    
except Exception as e:
    print(f"❌ Exception occurred: {e}")
    print(f"\n   Full traceback:")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n📊 Test Results:")
print(f"   ✅ File exists: {pdf_path.exists()}")

if 'first_page_text' in locals() and first_page_text:
    print(f"   ✅ PDF text extracted: {len(first_page_text):,} chars")
else:
    print(f"   ❌ PDF text extraction failed")

if 'metadata' in locals():
    print(f"   ✅ Metadata extraction succeeded")
    
    required_fields = ['ticker', 'quarter', 'year']
    required_present = all(metadata.get(f) is not None for f in required_fields)
    
    if required_present:
        print(f"   ✅ All required fields present (ticker, quarter, year)")
    else:
        print(f"   ⚠️  Some required fields are None")
    
    optional_fields = ['company_name', 'date']
    optional_none = [f for f in optional_fields if metadata.get(f) is None]
    
    if optional_none:
        print(f"   ⚠️  Optional fields are None: {', '.join(optional_none)}")
        print(f"      → These need default values to prevent .upper() errors!")
else:
    print(f"   ❌ Metadata extraction failed")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("\n💡 To prevent 'NoneType has no attribute upper' errors:")
print("   Ensure structure_transcripts.py uses:")
print()
print("   company_name = meta.get('company_name') or ''")
print("   region = meta.get('region') or 'UNKNOWN'")
print("   date = meta.get('date') or ''")
print()
print("   This guarantees all string fields have default values!")

print("\n" + "="*80)