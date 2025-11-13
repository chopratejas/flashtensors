#!/bin/bash
# Quick Start Script for FlashTensors Warm Swap Benchmarks

echo ""
echo "================================================================================"
echo "  🚀 FlashTensors Warm Swap Benchmark Suite"
echo "================================================================================"
echo ""
echo "This benchmark suite demonstrates FlashTensors' ultra-fast model swapping"
echo "compared to standard vLLM with SafeTensors."
echo ""
echo "Available benchmarks:"
echo ""
echo "  1. 🔵 Baseline (vLLM + SafeTensors)"
echo "     Standard approach - loads weights from disk every swap"
echo "     Command: python benchmark_baseline_warm_swap.py"
echo ""
echo "  2. ⚡ FlashTensors"
echo "     Optimized approach - loads weights from fast storage"
echo "     Command: python benchmark_flashtensors_warm_swap.py"
echo ""
echo "  3. 📊 Automated Comparison"
echo "     Runs both and shows speedup analysis"
echo "     Command: python benchmark_compare_warm_swap.py"
echo ""
echo "================================================================================"
echo ""

# Check if user wants to run comparison
read -p "Run automated comparison now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting automated comparison..."
    echo ""
    python benchmark_compare_warm_swap.py
else
    echo ""
    echo "To run benchmarks manually:"
    echo ""
    echo "  Baseline:     python benchmark_baseline_warm_swap.py"
    echo "  FlashTensors: python benchmark_flashtensors_warm_swap.py"
    echo "  Comparison:   python benchmark_compare_warm_swap.py"
    echo ""
    echo "See BENCHMARK_WARM_SWAP.md for detailed documentation."
    echo ""
fi
