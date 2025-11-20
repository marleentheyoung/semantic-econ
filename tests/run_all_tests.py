#!/bin/bash

# tests/run_all_tests.sh
# Convenience script to run all test suites

echo "========================================================================"
echo "RUNNING ALL TEST SUITES"
echo "========================================================================"

echo ""
echo "1. Backward Compatibility Tests"
echo "------------------------------------------------------------------------"
python -m pytest tests/test_backward_compatibility.py -v

echo ""
echo "2. Streaming API Correctness Tests"
echo "------------------------------------------------------------------------"
python -m pytest tests/test_streaming.py -v

echo ""
echo "3. Memory Efficiency Benchmarks"
echo "------------------------------------------------------------------------"
python tests/test_memory_efficiency.py

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"