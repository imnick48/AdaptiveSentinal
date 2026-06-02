#!/bin/bash
echo "Running Adaptive Sentinel"
python main.py --mode adaptive --threshold 0.4 --output results/ --bootstrap 
echo "Done. Check results/ directory."
