from abc import ABC

import pandas as pd
from loguru import logger
from tqdm import tqdm

from .base_op import BaseOp
from ..context import BaseContext, C


class BaseRayOp(BaseOp, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ray_task_list = []

    def submit_and_join_parallel_op(self, op: BaseOp, **kwargs):
        # Simplify logic, directly pass to core method to handle key detection
        return self.submit_and_join_ray_task(
            fn=op.call,
            task_desc=f"{op.name}.parallel",
            context=self.context,
            **kwargs,
        )

    def submit_and_join_ray_task(
            self,
            fn,
            parallel_key: str = "",
            task_desc: str = "",
            **kwargs,
    ):
        import ray

        max_workers = C.service_config.ray_max_workers
        self._ray_task_list.clear()

        # 1. Automatically detect parallel key
        if not parallel_key:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    parallel_key = key
                    break

        if not parallel_key:
            raise ValueError("No list found in kwargs to parallelize over.")

        logger.info(f"Using parallel_key='{parallel_key}' with {max_workers} workers")

        # 2. Extract parallel list
        parallel_list = kwargs.pop(parallel_key)
        assert isinstance(parallel_list, list)

        # 3. Preprocess large data objects (Object Store)
        # Note: When scheduling with Ray, passing ObjectRef is much faster than passing large objects themselves
        optimized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (pd.DataFrame, pd.Series, dict, list, BaseContext)):
                optimized_kwargs[key] = ray.put(value)
            else:
                optimized_kwargs[key] = value

        # 4. Submit tasks
        for i in range(max_workers):
            # Pass slicing logic to Ray remote function
            self.submit_ray_task(
                self.ray_task_loop,
                parallel_key=parallel_key,
                parallel_list=parallel_list,
                actor_index=i,
                max_workers=max_workers,
                internal_fn=fn,
                **optimized_kwargs,
            )
            logger.debug(f"Submitted ray task {i}/{max_workers} for {task_desc}")

        return self.join_ray_task(task_desc=task_desc)

    @staticmethod
    def ray_task_loop(
            parallel_key: str,
            parallel_list: list,
            actor_index: int,
            max_workers: int,
            internal_fn,
            **kwargs,
    ):
        """Worker internal loop to execute slice tasks"""
        result = []
        # Use slicing to distribute tasks: [start:stop:step]
        worker_tasks = parallel_list[actor_index::max_workers]

        for parallel_value in worker_tasks:
            # Use copy to avoid polluting kwargs
            current_kwargs = kwargs.copy()
            current_kwargs.update({"actor_index": actor_index, parallel_key: parallel_value})

            t_result = internal_fn(**current_kwargs)

            if t_result is not None:
                if isinstance(t_result, list):
                    result.extend(t_result)
                else:
                    result.append(t_result)
        return result

    def submit_ray_task(self, fn, *args, **kwargs):
        if not ray.is_initialized():
            ray.init(
                num_cpus=C.service_config.ray_max_workers,
                ignore_reinit_error=True
            )

        remote_fn = ray.remote(fn)
        task = remote_fn.remote(*args, **kwargs)
        self._ray_task_list.append(task)
        return self

    def join_ray_task(self, task_desc: str = None) -> list:
        """Use ray.wait to non-blockingly collect results, ensuring tqdm progress bar accurately reflects task completion"""
        results = []
        unfinished = list(self._ray_task_list)
        total_tasks = len(unfinished)

        with tqdm(total=total_tasks, desc=task_desc or f"{self.name}_ray") as pbar:
            while unfinished:
                # wait returns completed and uncompleted task lists
                ready, unfinished = ray.wait(unfinished, num_returns=1, timeout=None)
                for obj_ref in ready:
                    try:
                        t_result = ray.get(obj_ref)
                        if t_result:
                            if isinstance(t_result, list):
                                results.extend(t_result)
                            else:
                                results.append(t_result)
                    except Exception as e:
                        logger.error(f"Task failed with error: {e}")
                    pbar.update(1)

        self._ray_task_list.clear()
        return results
