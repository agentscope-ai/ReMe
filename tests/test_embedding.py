"""
Async unit tests for Embedding classes (OpenAIEmbeddingModel) covering:
- Async single text embedding
- Async batch text embeddings
- Async large batch with automatic batching
- Async VectorNode embedding (single and batch)
- Error handling and retries

Usage:
    python test_embedding.py --openai      # Test OpenAIEmbeddingModel only
    python test_embedding.py --all         # Test all embedding models
"""

# flake8: noqa: E402
# pylint: disable=C0413

import asyncio
import argparse
from typing import Type, List

from reme_ai.core.utils import load_env

load_env()

from reme_ai.core.embedding import OpenAIEmbeddingModel, BaseEmbeddingModel
from reme_ai.core.schema import VectorNode


def get_embedding_model(model_class: Type[BaseEmbeddingModel]) -> BaseEmbeddingModel:
    """Create and return an embedding model instance."""
    return model_class(
        model_name="text-embedding-v4",
        dimensions=1024,
        max_retries=2,
        raise_exception=True,
    )


def get_test_texts() -> List[str]:
    """Create test texts for embedding."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Solar energy is a renewable source of power.",
        "The capital of France is Paris.",
    ]


def get_large_batch_texts() -> List[str]:
    """Create a large batch of test texts to test automatic batching."""
    texts = []
    topics = [
        "Climate change and global warming",
        "Artificial intelligence and machine learning",
        "Renewable energy sources",
        "Space exploration and astronomy",
        "Medical research and healthcare",
        "Financial markets and economics",
        "Education and learning systems",
        "Transportation and urban planning",
    ]

    for i, topic in enumerate(topics):
        for j in range(3):
            texts.append(f"Text {i*3+j+1}: This is a sample text about {topic}.")

    return texts  # 24 texts total


def get_test_nodes() -> List[VectorNode]:
    """Create test VectorNodes for embedding."""
    texts = get_test_texts()
    return [
        VectorNode(
            content=text,
            metadata={"index": str(i), "category": "test"},
        )
        for i, text in enumerate(texts)
    ]


async def test_async_single_embedding(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous single text embedding."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Single Text Embedding")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    test_text = "Hello, this is a test sentence for embedding."

    print(f"Input text: {test_text}")

    embedding = await model.get_embedding(test_text)

    assert embedding is not None, f"{model_name}: Embedding is None"
    assert isinstance(embedding, list), f"{model_name}: Embedding is not a list"
    assert len(embedding) > 0, f"{model_name}: Empty embedding"
    assert len(embedding) == model.dimensions, f"{model_name}: Embedding dimension mismatch"
    assert all(isinstance(x, float) for x in embedding), f"{model_name}: Not all elements are floats"

    print("\n✓ Embedding generated successfully")
    print(f"  - Dimension: {len(embedding)}")
    print(f"  - First 5 values: {embedding[:5]}")
    print(f"  - Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")

    await model.close()
    print(f"✓ PASSED: {model_name} async single embedding")


async def test_async_batch_embeddings(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous batch text embeddings."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Batch Text Embeddings")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    test_texts = get_test_texts()

    print(f"Input: {len(test_texts)} texts")
    for i, text in enumerate(test_texts[:3], 1):
        print(f"  {i}. {text[:50]}...")

    embeddings = await model.get_embeddings(test_texts)

    assert embeddings is not None, f"{model_name}: Embeddings is None"
    assert isinstance(embeddings, list), f"{model_name}: Embeddings is not a list"
    assert len(embeddings) == len(test_texts), f"{model_name}: Embeddings count mismatch"

    for i, emb in enumerate(embeddings):
        assert isinstance(emb, list), f"{model_name}: Embedding {i} is not a list"
        assert len(emb) == model.dimensions, f"{model_name}: Embedding {i} dimension mismatch"
        assert all(isinstance(x, float) for x in emb), f"{model_name}: Embedding {i} has non-float values"

    print("\n✓ Batch embeddings generated successfully")
    print(f"  - Count: {len(embeddings)}")
    print(f"  - Dimension: {len(embeddings[0])}")
    print(f"  - First embedding preview: {embeddings[0][:3]}...")

    await model.close()
    print(f"✓ PASSED: {model_name} async batch embeddings")


async def test_async_large_batch_embeddings(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous large batch embeddings with automatic batching."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Large Batch with Auto-Batching")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    test_texts = get_large_batch_texts()

    print(f"Input: {len(test_texts)} texts")
    print(f"Max batch size: {model.max_batch_size}")
    print(f"Expected batches: {(len(test_texts) + model.max_batch_size - 1) // model.max_batch_size}")

    embeddings = await model.get_embeddings(test_texts)

    assert embeddings is not None, f"{model_name}: Embeddings is None"
    assert isinstance(embeddings, list), f"{model_name}: Embeddings is not a list"
    assert len(embeddings) == len(test_texts), f"{model_name}: Embeddings count mismatch"

    # Check all embeddings are valid
    for i, emb in enumerate(embeddings):
        assert isinstance(emb, list), f"{model_name}: Embedding {i} is not a list"
        assert len(emb) == model.dimensions, f"{model_name}: Embedding {i} dimension mismatch"

    print("\n✓ Large batch embeddings generated successfully")
    print(f"  - Total texts: {len(test_texts)}")
    print(f"  - Total embeddings: {len(embeddings)}")
    print(f"  - Dimension: {len(embeddings[0])}")

    await model.close()
    print(f"✓ PASSED: {model_name} async large batch embeddings")


async def test_async_single_node_embedding(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous single VectorNode embedding."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Single VectorNode Embedding")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    node = VectorNode(
        content="This is a test node for embedding.",
        metadata={"test": "true"},
    )

    print(f"Input node content: {node.content}")
    print(f"Initial vector: {node.vector}")

    result_node = await model.get_node_embedding(node)

    assert result_node is not None, f"{model_name}: Result node is None"
    assert result_node.vector is not None, f"{model_name}: Node vector is None"
    assert isinstance(result_node.vector, list), f"{model_name}: Vector is not a list"
    assert len(result_node.vector) == model.dimensions, f"{model_name}: Vector dimension mismatch"

    print("\n✓ Node embedding generated successfully")
    print(f"  - Vector dimension: {len(result_node.vector)}")
    print(f"  - First 5 values: {result_node.vector[:5]}")
    print(f"  - Metadata preserved: {result_node.metadata}")

    await model.close()
    print(f"✓ PASSED: {model_name} async single node embedding")


async def test_async_batch_node_embeddings(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous batch VectorNode embeddings."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Batch VectorNode Embeddings")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    nodes = get_test_nodes()

    print(f"Input: {len(nodes)} nodes")
    for i, node in enumerate(nodes[:3], 1):
        print(f"  {i}. {node.content[:50]}...")

    result_nodes = await model.get_node_embeddings(nodes)

    assert result_nodes is not None, f"{model_name}: Result nodes is None"
    assert isinstance(result_nodes, list), f"{model_name}: Result is not a list"
    assert len(result_nodes) == len(nodes), f"{model_name}: Nodes count mismatch"

    for i, node in enumerate(result_nodes):
        assert node.vector is not None, f"{model_name}: Node {i} vector is None"
        assert isinstance(node.vector, list), f"{model_name}: Node {i} vector is not a list"
        assert len(node.vector) == model.dimensions, f"{model_name}: Node {i} dimension mismatch"
        assert node.metadata is not None, f"{model_name}: Node {i} metadata is None"

    print("\n✓ Batch node embeddings generated successfully")
    print(f"  - Count: {len(result_nodes)}")
    print(f"  - All vectors populated: {all(n.vector is not None for n in result_nodes)}")
    print(f"  - All metadata preserved: {all(n.metadata is not None for n in result_nodes)}")

    await model.close()
    print(f"✓ PASSED: {model_name} async batch node embeddings")


async def test_async_large_batch_node_embeddings(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Test asynchronous large batch VectorNode embeddings with automatic batching."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: Async Large Batch Node Embeddings")
    print(f"{'='*60}")

    model = get_embedding_model(model_class)
    texts = get_large_batch_texts()
    nodes = [VectorNode(content=text, metadata={"index": str(i)}) for i, text in enumerate(texts)]

    print(f"Input: {len(nodes)} nodes")
    print(f"Max batch size: {model.max_batch_size}")
    print(f"Expected batches: {(len(nodes) + model.max_batch_size - 1) // model.max_batch_size}")

    result_nodes = await model.get_node_embeddings(nodes)

    assert result_nodes is not None, f"{model_name}: Result nodes is None"
    assert len(result_nodes) == len(nodes), f"{model_name}: Nodes count mismatch"

    # Check all nodes have embeddings
    nodes_with_vectors = sum(1 for n in result_nodes if n.vector is not None)
    print("\n✓ Large batch node embeddings generated successfully")
    print(f"  - Total nodes: {len(result_nodes)}")
    print(f"  - Nodes with vectors: {nodes_with_vectors}")
    print(f"  - Success rate: {nodes_with_vectors/len(result_nodes)*100:.1f}%")

    assert nodes_with_vectors == len(nodes), f"{model_name}: Not all nodes have vectors"

    await model.close()
    print(f"✓ PASSED: {model_name} async large batch node embeddings")


async def run_all_tests_for_model(model_class: Type[BaseEmbeddingModel], model_name: str):
    """Run all tests for a specific embedding model class."""
    print(f"\n\n{'#'*60}")
    print(f"# Running all tests for: {model_name}")
    print(f"{'#'*60}")

    await test_async_single_embedding(model_class, model_name)
    await test_async_batch_embeddings(model_class, model_name)
    await test_async_large_batch_embeddings(model_class, model_name)
    await test_async_single_node_embedding(model_class, model_name)
    await test_async_batch_node_embeddings(model_class, model_name)
    await test_async_large_batch_node_embeddings(model_class, model_name)

    print(f"\n{'='*60}")
    print(f"✓ All tests passed for {model_name}!")
    print(f"{'='*60}")


async def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(
        description="Run async embedding model tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_embedding.py --openai      # Test OpenAIEmbeddingModel only
  python test_embedding.py --all         # Test all embedding models
        """,
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Test OpenAIEmbeddingModel",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all available embedding models",
    )

    args = parser.parse_args()

    # Determine which models to test
    models_to_test = []

    if args.openai:
        models_to_test.append((OpenAIEmbeddingModel, "OpenAIEmbeddingModel"))
    elif args.all:
        models_to_test.append((OpenAIEmbeddingModel, "OpenAIEmbeddingModel"))
    else:
        # Default to all models if no argument provided
        models_to_test = [(OpenAIEmbeddingModel, "OpenAIEmbeddingModel")]
        print("No model specified, defaulting to all models")
        print("Use --openai to test OpenAI specifically\n")

    # Run tests for each model
    for model_class, model_name in models_to_test:
        try:
            await run_all_tests_for_model(model_class, model_name)
        except Exception as e:
            print(f"\n✗ FAILED: {model_name} tests failed with error:")
            print(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#'*60}")
    print("# TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"✓ All tests passed for {len(models_to_test)} embedding model(s):")
    for _, model_name in models_to_test:
        print(f"  - {model_name}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
