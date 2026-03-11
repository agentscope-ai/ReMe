#!/usr/bin/env python3
"""
Map Management Tool - Pre-generate and manage test maps
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from loguru import logger


class MapManager:
    """Map Manager - pre-generating, storing and loading test maps"""

    def __init__(self, data_dir: str = "./map/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_test_maps(
        self,
        num_maps: int,
        map_size: int = 4,
        base_seed: int = 10000,
    ) -> str:
        """
        Generate test map collection and save

        Args:
            num_maps: Number of maps to generate
            map_size: Map size
            base_seed: Base random seed

        Returns:
            Path of saved file
        """
        logger.info(f"🗺️ Generating {num_maps} test maps (size={map_size})")

        maps_data = []
        for i in range(num_maps):
            seed = base_seed + i
            np.random.seed(seed)
            map_desc = generate_random_map(size=map_size)

            maps_data.append(
                {
                    "map_id": i,
                    "seed": seed,
                    "map_size": map_size,
                    "map_desc": map_desc,  # Convert to list for JSON serialization
                },
            )

        # Save to file
        filename = f"test_maps_{num_maps}_{map_size}x{map_size}.jsonl"
        filepath = self.data_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for map_data in maps_data:
                f.write(json.dumps(map_data, ensure_ascii=False) + "\n")

        logger.info(f"✅ Test maps saved to {filepath}")
        return str(filepath)

    def load_test_maps(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test maps

        Args:
            filepath: Map file path

        Returns:
            Map data list
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Map file not found: {filepath}")

        maps_data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    map_data = json.loads(line)
                    # Convert list back to numpy array
                    maps_data.append(map_data)

        logger.info(f"📖 Loaded {len(maps_data)} test maps from {filepath}")
        return maps_data

    def get_map_by_index(self, maps_data: List[Dict], index: int) -> Optional[list]:
        """Get map by index"""
        if 0 <= index < len(maps_data):
            return maps_data[index]["map_desc"]
        return None

    def get_or_create_test_maps(self, num_maps: int, map_size: int = 4) -> List[Dict[str, Any]]:
        """
        Get or create test maps
        If file exists and has sufficient quantity, load directly; otherwise regenerate
        """
        filename = f"test_maps_{num_maps}_{map_size}x{map_size}.jsonl"
        filepath = self.data_dir / filename

        if filepath.exists():
            try:
                maps_data = self.load_test_maps(str(filepath))
                if len(maps_data) >= num_maps:
                    logger.info(f"✅ Using existing test maps: {filepath}")
                    return maps_data[:num_maps]  # Return required number of maps
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing maps: {e}, regenerating...")

        # File doesn't exist or insufficient quantity, regenerate
        self.generate_test_maps(num_maps, map_size)
        return self.load_test_maps(str(filepath))


if __name__ == "__main__":
    # Usage example
    manager = MapManager()

    # Generate 100 4x4 test maps
    manager.generate_test_maps(num_maps=100, map_size=4)

    # Load and view the first map
    maps = manager.load_test_maps("./map/test_maps_100_4x4.jsonl")
    print(f"First map:\n{maps[0]['map_desc']}")
