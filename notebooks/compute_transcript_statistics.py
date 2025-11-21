#!/usr/bin/env python3
"""
Memory-efficient transcript statistics computation.

Computes statistics for:
- Structured calls (number of firms, region, section lengths, speakers)
- Paragraph-level data from JSONL files

Processes data in chunks to avoid memory overflow with large datasets.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

from semantic_measurement.config.global_calibration import DATA_ROOT


# ============================================================================
# Structured Call Statistics (Memory-Efficient)
# ============================================================================

def stream_structured_calls(structured_root: Path, max_retries: int = 3):
    """
    Generator that yields call records one at a time.
    
    Args:
        structured_root: Path to structured_calls directory
        max_retries: Number of times to retry failed files (default: 3)
    
    Yields:
        dict: Individual call record
    """
    import time
    
    json_files = sorted(structured_root.glob('structured_calls_*.json'))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    failed_files = []
    
    for json_file in tqdm(json_files, desc="Loading structured call files", unit="file"):
        success = False
        
        for attempt in range(max_retries):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    calls = json.load(f)
                
                if not isinstance(calls, list):
                    print(f"\n⚠️ Skipping {json_file.name}: not a list")
                    break
                
                for call in calls:
                    yield call
                
                success = True
                break
                
            except OSError as e:
                # Network/disk I/O errors (errno 60, 5, etc.)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"\n⚠️ I/O error on {json_file.name} (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ Failed to load {json_file.name} after {max_retries} attempts: {e}")
                    failed_files.append(json_file.name)
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON in {json_file.name}: {e}")
                failed_files.append(json_file.name)
                break
                
            except Exception as e:
                print(f"\n❌ Unexpected error loading {json_file.name}: {e}")
                failed_files.append(json_file.name)
                break
    
    if failed_files:
        print(f"\n{'='*70}")
        print(f"WARNING: {len(failed_files)} files could not be loaded:")
        for fname in failed_files:
            print(f"  - {fname}")
        print(f"{'='*70}\n")


def compute_call_statistics(structured_root: Path, index: str):
    """
    Compute statistics on structured calls without loading all into memory.
    
    Returns:
        dict: Statistics summary
    """
    print(f"\n{'='*70}")
    print(f"STRUCTURED CALL STATISTICS: {index}")
    print(f"{'='*70}\n")
    
    # Accumulators
    n_calls = 0
    tickers = set()
    companies = set()
    year_counts = Counter()
    
    mgmt_lengths = []
    qa_lengths = []
    mgmt_para_counts = []
    qa_para_counts = []
    
    mgmt_speaker_counts = []
    qa_speaker_counts = []
    total_speaker_counts = []
    
    # Stream through calls
    for call in stream_structured_calls(structured_root):
        n_calls += 1
        
        # Basic counts
        tickers.add(call.get('ticker'))
        companies.add(call.get('company_name'))
        year_counts[call.get('year')] += 1
        
        # Section lengths
        mgmt_text = str(call.get('management_discussion_full', ''))
        qa_text = str(call.get('qa_section_full', ''))
        mgmt_lengths.append(len(mgmt_text))
        qa_lengths.append(len(qa_text))
        
        # Paragraph counts
        mgmt_paras = call.get('management_paragraphs', [])
        qa_paras = call.get('qa_paragraphs', [])
        mgmt_para_counts.append(len(mgmt_paras) if isinstance(mgmt_paras, list) else 0)
        qa_para_counts.append(len(qa_paras) if isinstance(qa_paras, list) else 0)
        
        # Speaker counts
        def count_unique_speakers(segments):
            if not isinstance(segments, list):
                return 0
            return len({seg.get('speaker') for seg in segments if seg.get('speaker')})
        
        mgmt_speakers = count_unique_speakers(call.get('speaker_segments_management', []))
        qa_speakers = count_unique_speakers(call.get('speaker_segments_qa', []))
        
        mgmt_speaker_counts.append(mgmt_speakers)
        qa_speaker_counts.append(qa_speakers)
        
        # Total unique speakers
        all_speakers = set()
        for seg_list in [call.get('speaker_segments_management', []), 
                         call.get('speaker_segments_qa', [])]:
            if isinstance(seg_list, list):
                all_speakers |= {seg.get('speaker') for seg in seg_list if seg.get('speaker')}
        total_speaker_counts.append(len(all_speakers))
    
    # Compute summary statistics
    stats = {
        'index': index,
        'n_calls': n_calls,
        'n_firms_ticker': len(tickers),
        'n_firms_company': len(companies),
        'year_distribution': dict(year_counts),
        
        'mgmt_length_chars': {
            'mean': pd.Series(mgmt_lengths).mean(),
            'std': pd.Series(mgmt_lengths).std(),
            'min': pd.Series(mgmt_lengths).min(),
            'median': pd.Series(mgmt_lengths).median(),
            'max': pd.Series(mgmt_lengths).max(),
        },
        
        'qa_length_chars': {
            'mean': pd.Series(qa_lengths).mean(),
            'std': pd.Series(qa_lengths).std(),
            'min': pd.Series(qa_lengths).min(),
            'median': pd.Series(qa_lengths).median(),
            'max': pd.Series(qa_lengths).max(),
        },
        
        'mgmt_paragraphs': {
            'mean': pd.Series(mgmt_para_counts).mean(),
            'std': pd.Series(mgmt_para_counts).std(),
            'min': pd.Series(mgmt_para_counts).min(),
            'median': pd.Series(mgmt_para_counts).median(),
            'max': pd.Series(mgmt_para_counts).max(),
        },
        
        'qa_paragraphs': {
            'mean': pd.Series(qa_para_counts).mean(),
            'std': pd.Series(qa_para_counts).std(),
            'min': pd.Series(qa_para_counts).min(),
            'median': pd.Series(qa_para_counts).median(),
            'max': pd.Series(qa_para_counts).max(),
        },
        
        'mgmt_speakers': {
            'mean': pd.Series(mgmt_speaker_counts).mean(),
            'std': pd.Series(mgmt_speaker_counts).std(),
            'min': pd.Series(mgmt_speaker_counts).min(),
            'median': pd.Series(mgmt_speaker_counts).median(),
            'max': pd.Series(mgmt_speaker_counts).max(),
        },
        
        'qa_speakers': {
            'mean': pd.Series(qa_speaker_counts).mean(),
            'std': pd.Series(qa_speaker_counts).std(),
            'min': pd.Series(qa_speaker_counts).min(),
            'median': pd.Series(qa_speaker_counts).median(),
            'max': pd.Series(qa_speaker_counts).max(),
        },
        
        'total_speakers': {
            'mean': pd.Series(total_speaker_counts).mean(),
            'std': pd.Series(total_speaker_counts).std(),
            'min': pd.Series(total_speaker_counts).min(),
            'median': pd.Series(total_speaker_counts).median(),
            'max': pd.Series(total_speaker_counts).max(),
        },
    }
    
    # Print results
    print(f"Total calls: {stats['n_calls']:,}")
    print(f"Unique firms (ticker): {stats['n_firms_ticker']}")
    print(f"Unique firms (company_name): {stats['n_firms_company']}")
    
    print(f"\nYear distribution:")
    for year in sorted(stats['year_distribution'].keys()):
        print(f"  {year}: {stats['year_distribution'][year]:,} calls")
    
    print(f"\nManagement section length (chars):")
    print(f"  Mean: {stats['mgmt_length_chars']['mean']:,.0f}")
    print(f"  Median: {stats['mgmt_length_chars']['median']:,.0f}")
    print(f"  Std: {stats['mgmt_length_chars']['std']:,.0f}")
    
    print(f"\nQ&A section length (chars):")
    print(f"  Mean: {stats['qa_length_chars']['mean']:,.0f}")
    print(f"  Median: {stats['qa_length_chars']['median']:,.0f}")
    print(f"  Std: {stats['qa_length_chars']['std']:,.0f}")
    
    print(f"\nManagement paragraphs per call:")
    print(f"  Mean: {stats['mgmt_paragraphs']['mean']:.1f}")
    print(f"  Median: {stats['mgmt_paragraphs']['median']:.0f}")
    
    print(f"\nQ&A paragraphs per call:")
    print(f"  Mean: {stats['qa_paragraphs']['mean']:.1f}")
    print(f"  Median: {stats['qa_paragraphs']['median']:.0f}")
    
    print(f"\nSpeakers per call:")
    print(f"  Management - Mean: {stats['mgmt_speakers']['mean']:.1f}")
    print(f"  Q&A - Mean: {stats['qa_speakers']['mean']:.1f}")
    print(f"  Total - Mean: {stats['total_speakers']['mean']:.1f}")
    
    return stats


# ============================================================================
# Paragraph-Level Statistics (Streaming)
# ============================================================================

def stream_paragraphs(paragraphs_file: Path, chunk_size: int = 50_000):
    """
    Stream paragraphs from JSONL file in chunks.
    
    Yields:
        dict: Individual paragraph record
    """
    with open(paragraphs_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_paragraph_statistics(paragraphs_file: Path, index: str):
    """
    Compute paragraph-level statistics using streaming.
    
    Returns:
        dict: Statistics summary
    """
    print(f"\n{'='*70}")
    print(f"PARAGRAPH-LEVEL STATISTICS: {index}")
    print(f"{'='*70}\n")
    
    # Accumulators
    n_paragraphs = 0
    section_counts = Counter()
    year_section_counts = defaultdict(Counter)
    
    mgmt_lengths = []
    qa_lengths = []
    mgmt_token_counts = []
    qa_token_counts = []
    
    # Stream through paragraphs with indefinite progress
    print(f"Processing paragraphs from: {paragraphs_file.name}")
    print("(This may take several minutes for large files...)\n")
    
    update_interval = 10000  # Print progress every 10k paragraphs
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with open(paragraphs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        para = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    n_paragraphs += 1
                    
                    # Print progress periodically
                    if n_paragraphs % update_interval == 0:
                        print(f"  Processed {n_paragraphs:,} paragraphs...")
                    
                    section = para.get('section')
                    year = para.get('year')
                    text = str(para.get('text', ''))
                    
                    section_counts[section] += 1
                    year_section_counts[year][section] += 1
                    
                    # Length statistics
                    text_len = len(text)
                    token_count = len(text.split())
                    
                    if section == 'management':
                        mgmt_lengths.append(text_len)
                        mgmt_token_counts.append(token_count)
                    elif section in ['qa', 'qa_pair']:
                        qa_lengths.append(text_len)
                        qa_token_counts.append(token_count)
            
            # Success - break retry loop
            break
            
        except OSError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\n⚠️ I/O Error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"   Retrying in {wait_time}s...")
                import time
                time.sleep(wait_time)
            else:
                print(f"\n❌ Failed to read paragraphs file after {max_retries} attempts: {e}")
                print(f"   This may indicate network storage issues or disk problems.")
                raise
    
    print(f"\n✓ Finished processing {n_paragraphs:,} paragraphs")
    
    # Compute summary
    stats = {
        'index': index,
        'n_paragraphs': n_paragraphs,
        'section_distribution': dict(section_counts),
        'year_section_distribution': {
            year: dict(counts) for year, counts in year_section_counts.items()
        },
        
        'mgmt_length_chars': {
            'mean': pd.Series(mgmt_lengths).mean(),
            'std': pd.Series(mgmt_lengths).std(),
            'min': pd.Series(mgmt_lengths).min(),
            'median': pd.Series(mgmt_lengths).median(),
            'max': pd.Series(mgmt_lengths).max(),
        } if mgmt_lengths else {},
        
        'qa_length_chars': {
            'mean': pd.Series(qa_lengths).mean(),
            'std': pd.Series(qa_lengths).std(),
            'min': pd.Series(qa_lengths).min(),
            'median': pd.Series(qa_lengths).median(),
            'max': pd.Series(qa_lengths).max(),
        } if qa_lengths else {},
        
        'mgmt_tokens': {
            'mean': pd.Series(mgmt_token_counts).mean(),
            'std': pd.Series(mgmt_token_counts).std(),
            'min': pd.Series(mgmt_token_counts).min(),
            'median': pd.Series(mgmt_token_counts).median(),
            'max': pd.Series(mgmt_token_counts).max(),
        } if mgmt_token_counts else {},
        
        'qa_tokens': {
            'mean': pd.Series(qa_token_counts).mean(),
            'std': pd.Series(qa_token_counts).std(),
            'min': pd.Series(qa_token_counts).min(),
            'median': pd.Series(qa_token_counts).median(),
            'max': pd.Series(qa_token_counts).max(),
        } if qa_token_counts else {},
    }
    
    # Print results
    print(f"\nTotal paragraphs: {stats['n_paragraphs']:,}")
    
    print(f"\nSection distribution:")
    for section, count in stats['section_distribution'].items():
        pct = 100 * count / stats['n_paragraphs']
        print(f"  {section}: {count:,} ({pct:.1f}%)")
    
    if stats['mgmt_length_chars']:
        print(f"\nManagement paragraph length (chars):")
        print(f"  Mean: {stats['mgmt_length_chars']['mean']:.0f}")
        print(f"  Median: {stats['mgmt_length_chars']['median']:.0f}")
        print(f"  Std: {stats['mgmt_length_chars']['std']:.0f}")
    
    if stats['qa_length_chars']:
        print(f"\nQ&A paragraph length (chars):")
        print(f"  Mean: {stats['qa_length_chars']['mean']:.0f}")
        print(f"  Median: {stats['qa_length_chars']['median']:.0f}")
        print(f"  Std: {stats['qa_length_chars']['std']:.0f}")
    
    if stats['mgmt_tokens']:
        print(f"\nManagement paragraph tokens:")
        print(f"  Mean: {stats['mgmt_tokens']['mean']:.1f}")
        print(f"  Median: {stats['mgmt_tokens']['median']:.0f}")
    
    if stats['qa_tokens']:
        print(f"\nQ&A paragraph tokens:")
        print(f"  Mean: {stats['qa_tokens']['mean']:.1f}")
        print(f"  Median: {stats['qa_tokens']['median']:.0f}")
    
    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute transcript statistics (memory-efficient)"
    )
    
    parser.add_argument(
        '--index',
        required=True,
        choices=['SP500', 'STOXX600'],
        help="Market index to analyze"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/statistics'),
        help="Directory to save output JSON files"
    )
    
    parser.add_argument(
        '--skip-calls',
        action='store_true',
        help="Skip structured call statistics"
    )
    
    parser.add_argument(
        '--skip-paragraphs',
        action='store_true',
        help="Skip paragraph statistics"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    structured_root = DATA_ROOT / 'intermediaries' / 'structured_calls' / args.index
    paragraphs_file = DATA_ROOT / 'intermediaries' / 'paragraphs' / f'{args.index}_paragraphs.jsonl'
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    # Structured call statistics
    if not args.skip_calls:
        if not structured_root.exists():
            print(f"⚠️ Structured calls directory not found: {structured_root}")
        else:
            call_stats = compute_call_statistics(structured_root, args.index)
            all_stats['calls'] = call_stats
    
    # Paragraph statistics
    if not args.skip_paragraphs:
        if not paragraphs_file.exists():
            print(f"⚠️ Paragraphs file not found: {paragraphs_file}")
        else:
            para_stats = compute_paragraph_statistics(paragraphs_file, args.index)
            all_stats['paragraphs'] = para_stats
    
    # Save to JSON
    output_file = args.output / f'{args.index}_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✓ Statistics saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()