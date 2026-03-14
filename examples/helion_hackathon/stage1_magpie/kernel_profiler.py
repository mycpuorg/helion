"""
Kernel Profiler for Stage 1: Magpie as the Kernel Selection Oracle.

Profiles transformer workloads to identify hot kernels and their
contribution to total execution time.
"""
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict


@dataclass
class ProfileResult:
    """Result of profiling a single kernel."""
    kernel_name: str
    total_time_ms: float
    call_count: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    config_space_size: int
    wall_time_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "total_time_ms": self.total_time_ms,
            "call_count": self.call_count,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "config_space_size": self.config_space_size,
            "wall_time_fraction": self.wall_time_fraction,
        }


class KernelProfiler:
    """
    Profiles GPU kernels in a workload to determine execution time distribution.

    This is the entry point for Stage 1 - it mimics Magpie's discover_kernels
    functionality by profiling actual kernel execution and ranking by
    wall-time contribution.
    """

    def __init__(self, warmup_iterations: int = 10, profile_iterations: int = 100):
        """
        Initialize the profiler.

        Args:
            warmup_iterations: Number of warmup runs before profiling
            profile_iterations: Number of iterations to profile
        """
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.kernel_times: Dict[str, List[float]] = defaultdict(list)
        self.kernel_configs: Dict[str, int] = {}
        self._profiling = False

    def register_kernel(self, name: str, config_space_size: int):
        """Register a kernel with its configuration space size."""
        self.kernel_configs[name] = config_space_size

    def record_kernel_time(self, name: str, time_ms: float):
        """Record execution time for a kernel invocation."""
        if self._profiling:
            self.kernel_times[name].append(time_ms)

    def profile_workload(
        self,
        workload_fn: Callable[[], Any],
        kernel_timers: Optional[Dict[str, Callable[[], float]]] = None,
    ) -> List[ProfileResult]:
        """
        Profile a workload function to collect kernel timing data.

        Args:
            workload_fn: Function that executes the workload once
            kernel_timers: Optional dict of kernel_name -> timing function

        Returns:
            List of ProfileResult sorted by wall-time fraction
        """
        import torch

        # Warmup
        for _ in range(self.warmup_iterations):
            workload_fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Reset timing data
        self.kernel_times.clear()

        # Profile
        self._profiling = True
        total_wall_time_ms = 0.0

        for _ in range(self.profile_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            workload_fn()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000
            total_wall_time_ms += elapsed_ms

        self._profiling = False

        # Compute results
        results = []
        total_kernel_time = sum(sum(times) for times in self.kernel_times.values())

        for name, times in self.kernel_times.items():
            if not times:
                continue

            total_time = sum(times)
            config_size = self.kernel_configs.get(name, 1)

            result = ProfileResult(
                kernel_name=name,
                total_time_ms=total_time,
                call_count=len(times),
                avg_time_ms=total_time / len(times),
                min_time_ms=min(times),
                max_time_ms=max(times),
                config_space_size=config_size,
                wall_time_fraction=total_time / total_kernel_time if total_kernel_time > 0 else 0,
            )
            results.append(result)

        # Sort by wall-time fraction descending
        results.sort(key=lambda r: r.wall_time_fraction, reverse=True)
        return results

    def profile_with_cuda_events(
        self,
        kernels: Dict[str, Callable[[], None]],
        input_shapes: Dict[str, tuple],
    ) -> List[ProfileResult]:
        """
        Profile kernels using CUDA events for accurate GPU timing.

        Args:
            kernels: Dict of kernel_name -> kernel_function
            input_shapes: Dict of kernel_name -> input_shape tuple

        Returns:
            List of ProfileResult sorted by wall-time fraction
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for profiling")

        results = []
        total_time = 0.0

        for name, kernel_fn in kernels.items():
            # Warmup
            for _ in range(self.warmup_iterations):
                kernel_fn()
            torch.cuda.synchronize()

            # Profile with CUDA events
            times = []
            for _ in range(self.profile_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                kernel_fn()
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                times.append(elapsed_ms)

            total = sum(times)
            total_time += total
            config_size = self.kernel_configs.get(name, 1)

            results.append(ProfileResult(
                kernel_name=name,
                total_time_ms=total,
                call_count=len(times),
                avg_time_ms=total / len(times),
                min_time_ms=min(times),
                max_time_ms=max(times),
                config_space_size=config_size,
            ))

        # Compute wall-time fractions
        for result in results:
            result.wall_time_fraction = result.total_time_ms / total_time if total_time > 0 else 0

        results.sort(key=lambda r: r.wall_time_fraction, reverse=True)
        return results

    @staticmethod
    def format_profile_table(results: List[ProfileResult]) -> str:
        """Format profiling results as a readable table."""
        lines = []
        lines.append("=" * 100)
        lines.append(f"{'Kernel Name':<25} {'Wall-time %':>12} {'Config Space':>15} {'Avg (ms)':>12} {'Calls':>8}")
        lines.append("=" * 100)

        for r in results:
            space_str = f"{r.config_space_size:.2e}" if r.config_space_size > 1000 else str(r.config_space_size)
            lines.append(
                f"{r.kernel_name:<25} {r.wall_time_fraction*100:>11.2f}% {space_str:>15} "
                f"{r.avg_time_ms:>12.4f} {r.call_count:>8}"
            )

        lines.append("=" * 100)
        return "\n".join(lines)


def discover_kernels(
    workload_fn: Callable[[], Any],
    kernel_configs: Dict[str, int],
    warmup: int = 10,
    iterations: int = 100,
) -> List[ProfileResult]:
    """
    Convenience function to discover and profile kernels in a workload.

    This is the Magpie-style entry point for kernel discovery.

    Args:
        workload_fn: Function that executes the workload
        kernel_configs: Dict of kernel_name -> config_space_size
        warmup: Warmup iterations
        iterations: Profiling iterations

    Returns:
        Sorted list of ProfileResult
    """
    profiler = KernelProfiler(warmup_iterations=warmup, profile_iterations=iterations)

    for name, config_size in kernel_configs.items():
        profiler.register_kernel(name, config_size)

    return profiler.profile_workload(workload_fn)
