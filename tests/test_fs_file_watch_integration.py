"""Integration test for ReMeFb file watching with memory_search and memory_get.

This test demonstrates the complete workflow:
1. Create markdown files with personal information in test_reme folder
2. Initialize ReMeFb with file watching enabled
3. Start file watching to automatically index files into the database
4. Use memory_search and memory_get to retrieve the indexed content
5. Modify the markdown files
6. Verify that modified content is properly indexed and retrievable

This validates the full pipeline:
  - File creation → File watcher → Database indexing
  - Search and retrieval functionality
  - File modification → Re-indexing → Updated search results
"""

import asyncio
import json
import shutil
from pathlib import Path

from reme import ReMeFb


# ==================== Test Configuration ====================


class TestConfig:
    """Test configuration settings."""

    WORKING_DIR = "test_reme"
    MEMORY_SUBDIR = "memory"


# ==================== Helper Functions ====================


def create_test_markdown_files(base_dir: str):
    """Create test markdown files with personal information.

    Args:
        base_dir: Base directory to create test files in
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    memory_path = base_path / TestConfig.MEMORY_SUBDIR
    memory_path.mkdir(parents=True, exist_ok=True)

    # Create personal profile markdown
    profile_file = memory_path / "profile.md"
    profile_content = """# Personal Profile

## Basic Information
My name is Zhang Wei (张伟). I am a 32-year-old software engineer living in Beijing, China.
I work at ByteDance as a senior backend engineer.

## Professional Skills
- Programming Languages: Python, Go, Java
- Specialization: Distributed systems and microservices architecture
- Experience: 8 years in software development

## Education
- Master's degree in Computer Science from Tsinghua University (2014)
- Focus on machine learning and data mining
"""
    profile_file.write_text(profile_content, encoding="utf-8")
    print(f"✓ Created: {profile_file}")

    # Create hobbies and interests markdown
    hobbies_file = memory_path / "hobbies.md"
    hobbies_content = """# Hobbies and Interests

## Technical Interests
I am passionate about cloud computing and containerization technologies.
Recently, I've been exploring Kubernetes and service mesh architectures.

## Personal Hobbies
- Reading: Love science fiction novels, especially works by Liu Cixin
- Sports: Play basketball every weekend with friends
- Travel: Visited 15 provinces in China, planning to visit Japan next year

## Learning Goals
- Deep dive into distributed tracing systems
- Learn more about database internals
- Improve English communication skills
"""
    hobbies_file.write_text(hobbies_content, encoding="utf-8")
    print(f"✓ Created: {hobbies_file}")

    # Create work projects markdown
    projects_file = memory_path / "projects.md"
    projects_content = """# Work Projects

## Current Projects

### Project Alpha (2024-present)
Building a high-performance message queue system to handle 1M+ QPS.
Using Go and Redis for the core infrastructure.

### Project Beta (2023-2024)
Developed a distributed configuration management system.
Integrated with Kubernetes for dynamic config updates.

## Past Experience
- Led the migration of monolithic services to microservices (2021-2023)
- Built automated deployment pipelines using Jenkins and GitLab CI (2020-2021)

## Technical Challenges Solved
- Resolved race conditions in concurrent data processing
- Optimized database queries reducing response time by 60%
"""
    projects_file.write_text(projects_content, encoding="utf-8")
    print(f"✓ Created: {projects_file}")

    return [profile_file, hobbies_file, projects_file]


def modify_test_markdown_files(base_dir: str):
    """Modify the test markdown files with updated information.

    Args:
        base_dir: Base directory containing test files
    """
    base_path = Path(base_dir)
    memory_path = base_path / TestConfig.MEMORY_SUBDIR

    # Modify profile - update job title and add new skill
    profile_file = memory_path / "profile.md"
    profile_content = """# Personal Profile

## Basic Information
My name is Zhang Wei (张伟). I am a 32-year-old software engineer living in Beijing, China.
I work at ByteDance as a **principal engineer** and tech lead.

## Professional Skills
- Programming Languages: Python, Go, Java, Rust
- Specialization: Distributed systems, microservices, and cloud-native architectures
- Experience: 8 years in software development
- **New**: Expert in observability and monitoring systems

## Education
- Master's degree in Computer Science from Tsinghua University (2014)
- Focus on machine learning and data mining
"""
    profile_file.write_text(profile_content, encoding="utf-8")
    print(f"✓ Modified: {profile_file}")

    # Modify hobbies - add new hobby
    hobbies_file = memory_path / "hobbies.md"
    hobbies_content = """# Hobbies and Interests

## Technical Interests
I am passionate about cloud computing and containerization technologies.
Recently, I've been exploring Kubernetes, service mesh, and eBPF technologies.

## Personal Hobbies
- Reading: Love science fiction novels, especially works by Liu Cixin
- Sports: Play basketball every weekend with friends
- Travel: Visited 15 provinces in China, planning to visit Japan next year
- **New**: Photography - Recently bought a Sony A7 III camera

## Learning Goals
- Deep dive into distributed tracing and eBPF
- Learn more about database internals and query optimization
- Improve English communication skills
- **New**: Master advanced photography techniques
"""
    hobbies_file.write_text(hobbies_content, encoding="utf-8")
    print(f"✓ Modified: {hobbies_file}")

    # Modify projects - add new project
    projects_file = memory_path / "projects.md"
    projects_content = """# Work Projects

## Current Projects

### Project Gamma (2024-present) **NEW**
Leading the development of an observability platform using OpenTelemetry.
Integrating metrics, traces, and logs into a unified dashboard.

### Project Alpha (2024-present)
Building a high-performance message queue system to handle 1M+ QPS.
Using Go and Redis for the core infrastructure.
**Update**: Successfully deployed to production, handling 2M+ QPS now.

### Project Beta (2023-2024)
Developed a distributed configuration management system.
Integrated with Kubernetes for dynamic config updates.

## Past Experience
- Led the migration of monolithic services to microservices (2021-2023)
- Built automated deployment pipelines using Jenkins and GitLab CI (2020-2021)

## Technical Challenges Solved
- Resolved race conditions in concurrent data processing
- Optimized database queries reducing response time by 60%
- **New**: Implemented distributed tracing reducing MTTR by 40%
"""
    projects_file.write_text(projects_content, encoding="utf-8")
    print(f"✓ Modified: {projects_file}")


def print_separator(title: str):
    """Print a formatted separator line."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_search_results(results: list[dict], query: str, context: str):
    """Pretty print search results.

    Args:
        results: List of search results
        query: The search query
        context: Context description (e.g., "BEFORE MODIFICATION")
    """
    print(f"\n{'-' * 80}")
    print(f"Search Results - {context}")
    print(f"Query: '{query}'")
    print(f"Found: {len(results)} results")
    print(f"{'-' * 80}")

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Path: {result.get('path', 'N/A')}")
        print(f"    Lines: {result.get('start_line', '?')}-{result.get('end_line', '?')}")
        print(f"    Score: {result.get('score', 0):.4f}")
        snippet = result.get("snippet", result.get("text", ""))
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"    Snippet: {snippet}")

    print(f"{'-' * 80}\n")


def print_get_result(content: str, path: str, context: str):
    """Pretty print memory_get result.

    Args:
        content: Content retrieved from memory_get
        path: File path
        context: Context description
    """
    print(f"\n{'-' * 80}")
    print(f"Memory Get Result - {context}")
    print(f"Path: {path}")
    print(f"Content length: {len(content)} chars, {len(content.split(chr(10)))} lines")
    print(f"{'-' * 80}")
    print(content[:500] + ("..." if len(content) > 500 else ""))
    print(f"{'-' * 80}\n")


# ==================== Test Functions ====================


async def test_file_watch_integration():
    """Complete integration test for file watching with search and get.

    This test validates:
    1. File creation and automatic indexing via file watcher
    2. Search functionality returns correct results
    3. Get functionality retrieves correct content
    4. File modification triggers re-indexing
    5. Updated content is properly searchable and retrievable
    """
    print_separator("FILE WATCH INTEGRATION TEST - START")

    # Clean up any existing test directory
    test_dir = Path(TestConfig.WORKING_DIR)
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✓ Cleaned up existing test directory: {test_dir}")

    # ==================== STEP 1: Create Test Files ====================
    print_separator("STEP 1: Creating Test Files")

    test_files = create_test_markdown_files(TestConfig.WORKING_DIR)
    print(f"\n✓ Created {len(test_files)} markdown files in {TestConfig.WORKING_DIR}")

    # ==================== STEP 2: Initialize ReMeFb ====================
    print_separator("STEP 2: Initializing ReMeFb with File Watching")

    reme_fs = ReMeFb(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_file_store_config={
            "backend": "sqlite",
            "store_name": "test_integration",
            "embedding_model": "default",
            "fts_enabled": True,
        },
        default_file_watcher_config={
            "backend": "full",
            "watch_paths": [TestConfig.WORKING_DIR, f"{TestConfig.WORKING_DIR}/memory"],
            "suffix_filters": [".md"],
            "recursive": False,
            "scan_on_start": True,
        },
    )

    print("✓ ReMeFb instance created")
    print(f"  Working directory: {TestConfig.WORKING_DIR}")
    print(f"  Watch paths: {TestConfig.WORKING_DIR}, {TestConfig.WORKING_DIR}/memory")
    print("  File filters: .md files")

    # ==================== STEP 3: Start File Watching ====================
    print_separator("STEP 3: Starting File Watcher")

    await reme_fs.start()
    print("✓ File watcher started")
    print("  Files will be automatically indexed into the database")

    # Give file watcher time to process files
    print("\nWaiting 3 seconds for file watcher to index files...")
    await asyncio.sleep(3)
    print("✓ File watcher should have processed all files")

    # ==================== STEP 4: Search Initial Content ====================
    print_separator("STEP 4: Searching Initial Content")

    queries_initial = [
        "What programming languages does Zhang Wei know?",
        "What are Zhang Wei's hobbies?",
        "What projects is Zhang Wei working on?",
    ]

    results_before = {}

    for query in queries_initial:
        print(f"\n📍 Searching: '{query}'")
        result_json = await reme_fs.memory_search(
            query=query,
            max_results=3,
            min_score=0.0,
        )
        results = json.loads(result_json)
        results_before[query] = results
        print_search_results(results, query, "BEFORE MODIFICATION")

        assert len(results) > 0, f"Should find results for query: {query}"
        print(f"✓ Found {len(results)} results")

    # ==================== STEP 5: Get Specific Content ====================
    print_separator("STEP 5: Getting Specific Content with memory_get")

    # Try to get content from profile.md
    profile_path = f"{TestConfig.MEMORY_SUBDIR}/profile.md"
    print(f"\n📍 Getting content from: {profile_path}")

    profile_content_before = await reme_fs.memory_get(
        path=profile_path,
        offset=1,
        limit=10,
    )
    print_get_result(profile_content_before, profile_path, "BEFORE MODIFICATION")

    assert "Zhang Wei" in profile_content_before, "Should contain Zhang Wei"
    assert "software engineer" in profile_content_before, "Should contain job title"
    print("✓ Content retrieved successfully")

    # Get full hobbies.md content
    hobbies_path = f"{TestConfig.MEMORY_SUBDIR}/hobbies.md"
    print(f"\n📍 Getting full content from: {hobbies_path}")

    hobbies_content_before = await reme_fs.memory_get(path=hobbies_path)
    print_get_result(hobbies_content_before, hobbies_path, "BEFORE MODIFICATION")

    assert "basketball" in hobbies_content_before, "Should contain hobbies"
    print("✓ Full content retrieved successfully")

    # ==================== STEP 6: Modify Files ====================
    print_separator("STEP 6: Modifying Test Files")

    print("Modifying markdown files with updated information...")
    modify_test_markdown_files(TestConfig.WORKING_DIR)

    # Give file watcher time to detect and re-index changes
    print("\nWaiting 3 seconds for file watcher to detect and re-index changes...")
    await asyncio.sleep(3)
    print("✓ File watcher should have re-indexed modified files")

    # ==================== STEP 7: Search Modified Content ====================
    print_separator("STEP 7: Searching Modified Content")

    queries_modified = [
        "What is Zhang Wei's current job title?",
        "Does Zhang Wei have any new hobbies?",
        "What new projects is Zhang Wei working on?",
        "What expertise does Zhang Wei have in observability?",
    ]

    results_after = {}

    for query in queries_modified:
        print(f"\n📍 Searching: '{query}'")
        result_json = await reme_fs.memory_search(
            query=query,
            max_results=3,
            min_score=0.0,
        )
        results = json.loads(result_json)
        results_after[query] = results
        print_search_results(results, query, "AFTER MODIFICATION")

        assert len(results) > 0, f"Should find results for query: {query}"
        print(f"✓ Found {len(results)} results")

    # ==================== STEP 8: Get Modified Content ====================
    print_separator("STEP 8: Getting Modified Content")

    # Get updated profile content
    print(f"\n📍 Getting updated content from: {profile_path}")
    profile_content_after = await reme_fs.memory_get(
        path=profile_path,
        offset=1,
        limit=10,
    )
    print_get_result(profile_content_after, profile_path, "AFTER MODIFICATION")

    assert "principal engineer" in profile_content_after, "Should contain updated job title"
    assert "Rust" in profile_content_after, "Should contain new programming language"
    print("✓ Updated profile content retrieved successfully")

    # Get updated hobbies content
    print(f"\n📍 Getting updated content from: {hobbies_path}")
    hobbies_content_after = await reme_fs.memory_get(path=hobbies_path)
    print_get_result(hobbies_content_after, hobbies_path, "AFTER MODIFICATION")

    assert "Photography" in hobbies_content_after, "Should contain new hobby"
    assert "Sony A7 III" in hobbies_content_after, "Should contain camera info"
    print("✓ Updated hobbies content retrieved successfully")

    # Get updated projects content
    projects_path = f"{TestConfig.MEMORY_SUBDIR}/projects.md"
    print(f"\n📍 Getting updated content from: {projects_path}")
    projects_content_after = await reme_fs.memory_get(path=projects_path)
    print_get_result(projects_content_after, projects_path, "AFTER MODIFICATION")

    assert "Project Gamma" in projects_content_after, "Should contain new project"
    assert "OpenTelemetry" in projects_content_after, "Should contain new technology"
    print("✓ Updated projects content retrieved successfully")

    # ==================== STEP 9: Verify Changes ====================
    print_separator("STEP 9: Verifying Content Changes")

    print("\n📍 Comparing BEFORE vs AFTER content:")

    # Verify profile changes
    print("\n1. Profile.md changes:")
    print(f"   Before: Contains 'software engineer' = {('software engineer' in profile_content_before.lower())}")
    print(f"   After: Contains 'principal engineer' = {('principal engineer' in profile_content_after.lower())}")
    print(f"   After: Contains 'Rust' = {('rust' in profile_content_after.lower())}")

    # Verify hobbies changes
    print("\n2. Hobbies.md changes:")
    print(f"   Before: Contains 'Photography' = {('photography' in hobbies_content_before.lower())}")
    print(f"   After: Contains 'Photography' = {('photography' in hobbies_content_after.lower())}")
    print(f"   After: Contains 'Sony A7 III' = {('sony' in hobbies_content_after.lower())}")

    # Verify projects changes
    print("\n3. Projects.md changes:")
    print(f"   After: Contains 'Project Gamma' = {('Project Gamma' in projects_content_after)}")
    print(f"   After: Contains 'OpenTelemetry' = {('OpenTelemetry' in projects_content_after)}")

    print("\n✓ All content changes verified successfully")

    # ==================== STEP 10: Cleanup ====================
    print_separator("STEP 10: Cleanup")

    await reme_fs.close()
    print("✓ ReMeFb closed")

    # Clean up test directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✓ Removed test directory: {test_dir}")
    else:
        print(f"⚠️  Test directory does not exist: {test_dir}")

    print("\n✓ All test data cleaned up")

    print_separator("FILE WATCH INTEGRATION TEST - COMPLETED SUCCESSFULLY")


# ==================== Main Entry Point ====================


async def main():
    """Run the file watch integration test."""
    print("\n" + "=" * 80)
    print("  ReMeFb File Watch Integration Test")
    print("=" * 80)
    print("\nThis test validates the complete file watching workflow:")
    print("  1. Create markdown files with personal information")
    print("  2. Initialize ReMeFb and start file watching")
    print("  3. Verify automatic indexing into database")
    print("  4. Search and retrieve initial content")
    print("  5. Modify files and verify re-indexing")
    print("  6. Search and retrieve modified content")
    print("  7. Compare before/after results")
    print("=" * 80)

    try:
        await test_file_watch_integration()

        print("\n" + "=" * 80)
        print("  ✓ All tests passed successfully!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"  ✗ Test failed with error: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
