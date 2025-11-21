#!/usr/bin/env python3
"""
Compute statistics for both SP500 and STOXX600.
"""

import subprocess
import sys
from pathlib import Path

def run_stats(index: str):
    """Run statistics for one index."""
    print(f"\n{'='*70}")
    print(f"Processing {index}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [
            sys.executable,
            'notebooks/compute_transcript_statistics.py',
            '--index', index
        ],
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"⚠️ Failed to compute statistics for {index}")
        return False
    
    return True

def main():
    print("\nComputing transcript statistics for both indexes...")
    
    success_sp500 = run_stats('SP500')
    success_stoxx = run_stats('STOXX600')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"SP500: {'✓' if success_sp500 else '✗'}")
    print(f"STOXX600: {'✓' if success_stoxx else '✗'}")
    print(f"\nResults saved in: outputs/statistics/")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()