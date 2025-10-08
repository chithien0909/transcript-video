#!/usr/bin/env python3
"""
Performance monitoring script for the transcription tool.
Shows real-time system resource usage and optimization status.
"""

import os
import sys
import time
import psutil
import platform
from datetime import datetime

def get_system_info():
    """Get comprehensive system information."""
    info = {
        'cpu_count': os.cpu_count(),
        'physical_cores': psutil.cpu_count(logical=False),
        'memory_total': psutil.virtual_memory().total / (1024**3),
        'memory_available': psutil.virtual_memory().available / (1024**3),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_freq': psutil.cpu_freq(),
        'platform': platform.system(),
        'architecture': platform.machine(),
        'python_version': platform.python_version()
    }
    return info

def monitor_resources(duration=60, interval=2):
    """Monitor system resources for specified duration."""
    print(f"=== PERFORMANCE MONITOR ===")
    print(f"Monitoring for {duration} seconds (interval: {interval}s)")
    print(f"Press Ctrl+C to stop early")
    print()
    
    start_time = time.time()
    max_cpu = 0
    max_memory = 0
    
    try:
        while time.time() - start_time < duration:
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Update maximums
            max_cpu = max(max_cpu, cpu_percent)
            max_memory = max(max_memory, memory.percent)
            
            # Display current status
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\r[{timestamp}] CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | Available: {memory.available/(1024**3):4.1f}GB", end="", flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped by user")
    
    print(f"\n\n=== MONITORING SUMMARY ===")
    print(f"Duration: {time.time() - start_time:.1f}s")
    print(f"Peak CPU usage: {max_cpu:.1f}%")
    print(f"Peak RAM usage: {max_memory:.1f}%")
    
    return max_cpu, max_memory

def check_optimizations():
    """Check if performance optimizations are active."""
    print("=== OPTIMIZATION STATUS ===")
    
    # Check environment variables
    env_vars = {
        'OMP_NUM_THREADS': 'OpenMP threading',
        'MKL_NUM_THREADS': 'Intel MKL threading', 
        'OPENBLAS_NUM_THREADS': 'OpenBLAS threading',
        'NUMEXPR_NUM_THREADS': 'NumExpr threading'
    }
    
    for var, description in env_vars.items():
        value = os.environ.get(var, 'Not set')
        status = "[OK]" if value != 'Not set' else "[MISSING]"
        print(f"{status} {description}: {value}")
    
    # Check system info
    info = get_system_info()
    print(f"\n=== SYSTEM INFORMATION ===")
    print(f"CPU: {info['cpu_count']} logical cores, {info['physical_cores']} physical cores")
    print(f"Memory: {info['memory_total']:.1f}GB total, {info['memory_available']:.1f}GB available")
    print(f"Platform: {info['platform']} {info['architecture']}")
    print(f"Python: {info['python_version']}")
    
    if info['cpu_freq']:
        print(f"CPU Frequency: {info['cpu_freq'].current:.0f} MHz")
    
    # Calculate optimal settings
    optimal_workers = min(10, info['cpu_count'] - 4) if info['cpu_count'] >= 16 else min(6, info['cpu_count'] - 2)
    optimal_chunk = min(30, max(10, int(info['memory_available'] / optimal_workers)))
    
    print(f"\n=== RECOMMENDED SETTINGS ===")
    print(f"Optimal workers: {optimal_workers}")
    print(f"Optimal chunk size: {optimal_chunk}s")
    print(f"Max memory per worker: 2.0GB")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            monitor_resources(duration)
        elif sys.argv[1] == "check":
            check_optimizations()
        else:
            print("Usage: python performance_monitor.py [monitor|check] [duration]")
    else:
        check_optimizations()
        print(f"\nTo monitor resources: python performance_monitor.py monitor [duration]")
        print(f"To check optimizations: python performance_monitor.py check")

if __name__ == "__main__":
    main()
