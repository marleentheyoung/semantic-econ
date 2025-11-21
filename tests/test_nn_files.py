#!/usr/bin/env python3
"""
Diagnostic test for NN files that are still failing.

Find which field is None and causing .upper() errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_measurement.data.pdf_metadata import extract_metadata_from_pdf

# Test one of the failing NN files
TEST_FILES = [
    "NN - Q1 2016 Earnings Call 26May2016 400 AM ET CORRECTED TRANSCRIPT - 26-May-16.pdf",
    "NN - Q2 2014 Earnings Call 6August2014 600 AM ET CORRECTED TRANSCRIPT - 06-Aug-14.pdf",
]

PDF_FOLDER = "/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/STOXX600/"

print("="*80)
print("DIAGNOSTIC: NN Files Still Failing")
print("="*80)

for test_filename in TEST_FILES:
    test_pdf_path = PDF_FOLDER + test_filename
    
    print(f"\n{'='*80}")
    print(f"Testing: {test_filename}")
    print(f"{'='*80}")
    
    # Check if file exists
    pdf_path = Path(test_pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {test_pdf_path}")
        continue
    
    print(f"✅ File exists ({pdf_path.stat().st_size:,} bytes)")
    
    # Extract metadata
    try:
        metadata = extract_metadata_from_pdf(test_pdf_path, index="STOXX600")
        
        print(f"\n✅ Metadata extraction succeeded")
        print(f"\nReturned fields:")
        
        none_fields = []
        for key, value in metadata.items():
            is_none = value is None
            symbol = "⚠️ " if is_none else "✅"
            print(f"   {symbol} {key}: {repr(value)}")
            if is_none:
                none_fields.append(key)
        
        if none_fields:
            print(f"\n❌ FOUND THE PROBLEM!")
            print(f"   Fields that are None: {', '.join(none_fields)}")
            print(f"\n   If structure_transcripts tries to call .upper() on these:")
            for field in none_fields:
                print(f"      {field}.upper() → 💥 CRASH!")
        else:
            print(f"\n✅ All fields populated - should not crash")
            
    except Exception as e:
        print(f"\n❌ Metadata extraction failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
If any fields are None:
→ These need to be handled in structure_transcripts.py

Currently we have defaults for:
  • company_name = meta.get('company_name') or ''
  • region = meta.get('region') or 'UNKNOWN'
  • date = meta.get('date') or ''

If OTHER fields (like ticker, quarter, year) are None:
→ The transcript should be SKIPPED (missing required data)
→ Check if structure_transcripts is properly checking these
""")