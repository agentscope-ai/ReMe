"""
Self-contained script for CacheHandler's comprehensive test suite.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from reme_ai.core.utils.cache_handler import CacheHandler


def run_tests():
    """Execute comprehensive tests for CacheHandler."""
    test_dir = Path("test_cache_system")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    handler = CacheHandler(cache_dir=test_dir)
    logger.info("Starting CacheHandler tests...")

    # 1. Test Data Types
    logger.info("Testing data types support...")

    # DataFrame
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert handler.save("df_test", df)
    assert isinstance(handler.load("df_test"), pd.DataFrame)
    assert handler.load("df_test").shape == (2, 2)

    # Dict & List
    d = {"key": "value", "nested": [1, 2]}
    l_value = [1, "string", {"a": 1}]
    assert handler.save("dict_test", d)
    assert handler.save("list_test", l_value)
    assert handler.load("dict_test")["key"] == "value"
    assert handler.load("list_test")[1] == "string"

    # String
    s = "Hello World"
    assert handler.save("str_test", s)
    assert handler.load("str_test") == "Hello World"

    # 2. Test Expiration
    logger.info("Testing expiration logic...")
    # Save with 1 second expiry (approx 0.00027 hours)
    handler.save("exp_test", {"data": 1}, expire_hours=0.00001)
    assert handler.exists("exp_test") is True

    # Manually modify metadata to force expiration for instant test
    handler.metadata["exp_test"]["expire_at"] = (datetime.now() - timedelta(seconds=1)).isoformat()
    assert handler.exists("exp_test") is False
    assert handler.load("exp_test") is None
    assert "exp_test" not in handler.metadata  # Auto-cleaned

    # 3. Test Existence and Deletion
    logger.info("Testing delete and exists...")
    handler.save("del_test", "delete me")
    assert handler.exists("del_test") is True
    handler.delete("del_test")
    assert handler.exists("del_test") is False
    assert not (test_dir / "del_test.txt").exists()

    # 4. Test Persistence (Reload handler)
    logger.info("Testing persistence...")
    handler.save("persist_test", [1, 2, 3])
    new_handler = CacheHandler(cache_dir=test_dir)
    assert new_handler.exists("persist_test") is True
    assert new_handler.load("persist_test") == [1, 2, 3]

    # 5. Test Statistics and Clear
    logger.info("Testing stats and clear...")
    stats = handler.get_stats()
    assert stats["count"] > 0
    handler.clear_all()
    assert handler.get_stats()["count"] == 0
    assert len(list(test_dir.glob("*"))) == 1  # Only metadata.json remains

    # 6. Test Error Handling
    logger.info("Testing error handling...")
    assert handler.load("non_existent_key") is None
    # Test unsupported type
    assert handler.save("invalid", {1, 2}) is False

    logger.success("All tests passed successfully!")

    # Cleanup after tests
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    run_tests()
