"""Base class for Ray-based parallel operations."""

from abc import ABCMeta
from typing import Callable

import pandas as pd
from loguru import logger
from tqdm import tqdm

from .base_op import BaseOp
from ..base_dict import BaseDict

_RAY_IMPORT_ERROR = None

try:
    import ray
except ImportError as _e:
    _RAY_IMPORT_ERROR = _e
    ray = None


class BaseRayOp(BaseOp, metaclass=ABCMeta):
    """Base class for Ray-based parallel operations."""

    def __init__(self, **kwargs):
        if _RAY_IMPORT_ERROR:
            raise ImportError("Ray requires extra dependencies. Install with `pip install ray`")

        super().__init__(**kwargs)
        self._ray_task_list: list = []

    def submit_and_join_parallel_op(self, op: BaseOp, **kwargs) -> list:
        """Submit a BaseOp to be executed in parallel via Ray."""
        return self.submit_and_join_ray_task(fn=op.call, task_desc=op.name, context=self.context, **kwargs)

    def submit_and_join_ray_task(self, fn: Callable, parallel_key: str = "", task_desc: str = "", **kwargs) -> list:
        """Divide data into chunks and execute them across Ray workers."""
        max_workers = self.service_context.ray_max_workers
        self._ray_task_list.clear()

        # Automatically detect the key containing the list to parallelize
        if not parallel_key:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    parallel_key = key
                    break

        if not parallel_key:
            raise ValueError("No list found in kwargs to parallelize over.")

        parallel_list = kwargs.pop(parallel_key)
        logger.info(f"Parallelizing '{parallel_key}' across {max_workers} workers")

        # Put large shared objects into the Ray Object Store once
        optimized_kwargs = {
            k: (ray.put(v) if isinstance(v, (pd.DataFrame, pd.Series, dict, list, BaseDict)) else v)
            for k, v in kwargs.items()
        }

        # Submit sliced chunks to reduce internode data transfer
        remote_task_loop = ray.remote(self._ray_task_loop)
        for i in range(max_workers):
            chunk = parallel_list[i::max_workers]
            if not chunk:
                continue

            task = remote_task_loop.remote(
                fn,
                parallel_key,
                chunk,
                i,
                **optimized_kwargs,
            )
            self._ray_task_list.append(task)
            logger.info(f"Submitted task {i + 1}/{max_workers} for {task_desc}")

        return self.join_ray_task(task_desc=task_desc)

    @staticmethod
    def _ray_task_loop(internal_fn: Callable, parallel_key: str, chunk: list, actor_index: int, **kwargs) -> list:
        """Execute the function over a specific chunk of data on a worker."""
        results = []
        for value in chunk:
            current_kwargs = {**kwargs, "actor_index": actor_index, parallel_key: value}
            t_result = internal_fn(**current_kwargs)

            if t_result is not None:
                if isinstance(t_result, list):
                    results.extend(t_result)
                else:
                    results.append(t_result)
        return results

    def submit_ray_task(self, fn, *args, **kwargs):
        """Submit a single Ray task to the task list for later execution."""
        if not ray.is_initialized():
            ray.init(num_cpus=self.service_context.ray_max_workers, ignore_reinit_error=True)

        remote_fn = ray.remote(fn)
        task = remote_fn.remote(*args, **kwargs)
        self._ray_task_list.append(task)
        return self

    def join_ray_task(self, task_desc: str | None = None) -> list:
        """Collect results from Ray workers using a progress bar."""
        results = []
        unfinished = list(self._ray_task_list)

        with tqdm(total=len(unfinished), desc=task_desc or f"{self.name}_ray") as pbar:
            while unfinished:
                ready, unfinished = ray.wait(unfinished, num_returns=1)
                for obj_ref in ready:
                    try:
                        t_result = ray.get(obj_ref)
                        if isinstance(t_result, list):
                            results.extend(t_result)
                        elif t_result is not None:
                            results.append(t_result)
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                    pbar.update(1)

        self._ray_task_list.clear()
        return results
