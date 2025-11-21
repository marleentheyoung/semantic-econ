#!/usr/bin/env python3
"""
Test to verify metadata extraction flow for RELL file.

This tests the EXACT flow that happens in the pipeline:
1. extract_metadata_from_pdf() is called
2. Returns a dict with metadata
3. Check if 'region' field is present
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_measurement.data.pdf_metadata import extract_metadata_from_pdf

# The specific file
TEST_FILENAME = "RELL - Q4 2010 Earnings Call 17February2011 400 AM ET CORRECTED TRANSCRIPT - 17-Feb-11.pdf"
TEST_PDF_PATH = "/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600/" + TEST_FILENAME

print("="*80)
print("TEST: Verify extract_metadata_from_pdf() Returns Region")
print("="*80)

print(f"\n📄 Testing file: {TEST_FILENAME}")
print(f"📁 Index: STOXX600")

# ============================================================================
# TEST 1: Call the function exactly as pipeline does
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Call extract_metadata_from_pdf()")
print("="*80)

print("\nCalling: extract_metadata_from_pdf(pdf_path, index='STOXX600')")

try:
    metadata = extract_metadata_from_pdf(TEST_PDF_PATH, index="STOXX600")
    
    print("\n✅ Function returned successfully")
    print(f"\nResult type: {type(metadata)}")
    print(f"Result keys: {list(metadata.keys())}")
    
except Exception as e:
    print(f"\n❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: Check each field
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Inspect Returned Fields")
print("="*80)

print("\nReturned metadata:")
for key, value in metadata.items():
    is_none = value is None
    symbol = "⚠️ " if is_none else "✅"
    print(f"   {symbol} {key}: {repr(value)}")

# ============================================================================
# TEST 3: Check for 'region' field specifically
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Check for 'region' Field")
print("="*80)

if "region" in metadata:
    region_value = metadata["region"]
    if region_value is None:
        print("❌ 'region' key EXISTS but value is None")
        print("   This will cause .upper() error!")
    else:
        print(f"✅ 'region' key EXISTS with value: {repr(region_value)}")
else:
    print("❌ 'region' key DOES NOT EXIST in returned dict")
    print("   This means it's missing from the function!")

# ============================================================================
# TEST 4: Simulate what structure_transcripts does
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Simulate structure_transcripts Usage")
print("="*80)

print("\nSimulating: structure_transcripts receives this metadata")
print("Then tries to access fields...")

# Simulate OLD code (what was causing the crash)
print("\n--- OLD CODE (causes crash) ---")
ticker = metadata.get("ticker")
quarter = metadata.get("quarter")
year = metadata.get("year")
company_name = metadata.get("company_name")
region = metadata.get("region")  # This is the problem!

print(f"ticker = {repr(ticker)}")
print(f"quarter = {repr(quarter)}")
print(f"year = {repr(year)}")
print(f"company_name = {repr(company_name)}")
print(f"region = {repr(region)}")

if region is None:
    print("\n❌ region is None!")
    print("   If code tries: region.upper()")
    try:
        result = region.upper()
    except AttributeError as e:
        print(f"   💥 CRASH: {e}")
else:
    print(f"\n✅ region has value: {repr(region)}")
    print(f"   Calling region.upper()...")
    try:
        result = region.upper()
        print(f"   ✅ Success: {repr(result)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Now test NEW code with defaults
print("\n--- NEW CODE (with safety defaults) ---")
region_safe = metadata.get("region") or "UNKNOWN"
company_name_safe = metadata.get("company_name") or ""

print(f"region_safe = {repr(region_safe)}")
print(f"company_name_safe = {repr(company_name_safe)}")

print(f"\nCalling region_safe.upper()...")
try:
    result = region_safe.upper()
    print(f"✅ Success: {repr(result)}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 5: Check required vs optional fields
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Required vs Optional Fields")
print("="*80)

required_fields = ['ticker', 'quarter', 'year']
optional_fields = ['company_name', 'date', 'region']

print("\nRequired fields (must not be None):")
required_ok = True
for field in required_fields:
    value = metadata.get(field)
    is_none = value is None
    symbol = "❌" if is_none else "✅"
    print(f"   {symbol} {field}: {repr(value)}")
    if is_none:
        required_ok = False

print("\nOptional fields (can be None, but need defaults):")
optional_none = []
for field in optional_fields:
    value = metadata.get(field)
    is_none = value is None
    symbol = "⚠️ " if is_none else "✅"
    print(f"   {symbol} {field}: {repr(value)}")
    if is_none:
        optional_none.append(field)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n📊 Results:")
print(f"   • All required fields present: {'✅ YES' if required_ok else '❌ NO'}")

if optional_none:
    print(f"   • Optional fields that are None: {', '.join(optional_none)}")
    print(f"     ⚠️  These need default values in structure_transcripts!")
else:
    print(f"   • All optional fields populated: ✅ YES")

print("\n🔍 Specific findings:")
if "region" not in metadata:
    print("   ❌ CRITICAL: 'region' field missing from return dict")
    print("      → Fix: Update extract_metadata_from_pdf() to include region")
elif metadata.get("region") is None:
    print("   ❌ CRITICAL: 'region' field is None")
    print("      → Fix: Set region = 'US' or 'EU' based on index")
else:
    print(f"   ✅ 'region' field present and populated: {repr(metadata['region'])}")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if "region" not in metadata or metadata.get("region") is None:
    print("\n🔧 ACTION REQUIRED:")
    print("\n   Update extract_metadata_from_pdf() in pdf_metadata.py:")
    print("""
    def extract_metadata_from_pdf(pdf_path: str, index: str):
        # ... existing code ...
        
        if index_upper == "SP500":
            date, quarter, year, ticker, company_name = _extract_company_ticker_date_us(...)
            region = "US"  # ← ADD THIS
        elif index_upper == "STOXX600":
            date, quarter, year, ticker, company_name = _extract_company_ticker_date_eu(...)
            region = "EU"  # ← ADD THIS
        
        return {
            "file": filename,
            "date": date,
            "quarter": quarter,
            "year": year,
            "ticker": ticker,
            "company_name": company_name,
            "region": region,  # ← ADD THIS
        }
    """)
else:
    print("\n✅ No action needed - region field is properly set!")

print("\n" + "="*80)