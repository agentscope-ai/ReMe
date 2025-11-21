"""
Test script for ContextCompressOp.

This script demonstrates how to use the context compression operation to reduce
token usage in conversation histories using language models.
"""

import asyncio

from loguru import logger

from reme_ai.context.offload.context_compress_op import ContextCompressOp
from reme_ai.main import ReMeApp


async def main():
    """Main function to test ContextCompressOp."""

    async with ReMeApp():
        logger.info("=" * 80)
        logger.info("Testing ContextCompressOp - LLM-based Context Compression")
        logger.info("=" * 80)

        # Create a mock conversation with multiple messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specialized in software development.",
            },
            {
                "role": "user",
                "content": "I need help building a REST API in Python. I want to use FastAPI.",
            },
            {
                "role": "assistant",
                "content": "Great choice! FastAPI is an excellent framework for building REST APIs. "
                "It's fast, modern, and has automatic API documentation. To get started, you'll need "
                "to install FastAPI and uvicorn. Would you like me to guide you through setting up "
                "your first endpoint?",
            },
            {
                "role": "user",
                "content": "Yes please. I want to create a user management API with CRUD operations.",
            },
            {
                "role": "assistant",
                "content": "Perfect! For a user management API, I recommend this structure:\n"
                "1. Define a User model using Pydantic\n"
                "2. Create POST /users endpoint for creating users\n"
                "3. Create GET /users and GET /users/{id} for reading\n"
                "4. Create PUT /users/{id} for updates\n"
                "5. Create DELETE /users/{id} for deletion\n"
                "We'll also need a database. Would you prefer SQLite, PostgreSQL, or MongoDB?",
            },
            {
                "role": "user",
                "content": "Let's use PostgreSQL. Also, I need JWT authentication.",
            },
            {
                "role": "assistant",
                "content": "Excellent. PostgreSQL is a robust choice. For JWT authentication, we'll use "
                "python-jose library. Here's what we'll implement:\n"
                "1. User registration endpoint\n"
                "2. Login endpoint that returns JWT token\n"
                "3. Protected endpoints that require valid JWT\n"
                "4. Password hashing using bcrypt\n"
                "Let me show you the code for the User model first.",
            },
            {
                "role": "user",
                "content": "Before we proceed, I also need rate limiting and input validation.",
            },
            {
                "role": "assistant",
                "content": "Good thinking! For rate limiting, we can use slowapi library which integrates "
                "well with FastAPI. For input validation, Pydantic (which FastAPI uses) handles most of it, "
                "but we can add custom validators. I'll also add request validation middleware. "
                "Let's start implementing all of this step by step.",
            },
            {
                "role": "user",
                "content": "I need to build a distributed task queue system in Python that can handle millions of tasks"
                " per day. It needs to be horizontally scalable and fault-tolerant.",
            },
            {
                "role": "assistant",
                "content": "That's a challenging but exciting project! For a high-throughput distributed task queue, "
                "I recommend a architecture with:\n\n1. **Message Broker**: Redis or RabbitMQ for task "
                "distribution\n2. **Task Workers**: Multiple worker processes across multiple machines\n3."
                " **Result Backend**: Redis or PostgreSQL for storing task results\n4. **Monitoring**:"
                " Prometheus + Grafana for metrics\n5. **API Layer**: FastAPI for task submission and "
                "status queries\n\nFor the core library, we can build on top of Celery or create a custo"
                "m solution. Would you like me to design the system architecture first, or do you have pr"
                "eferences for specific technologies?",
            },
            {
                "role": "user",
                "content": "I want to build this from scratch without Celery. I need custom features like task"
                " priorities, retry policies with exponential backoff, and task dependencies. Also,"
                " I need it to support both synchronous and asynchronous task execution patterns.",
            },
            {
                "role": "assistant",
                "content": "Excellent! Building from scratch gives us full control. Let me design a comprehensive "
                "system architecture:\n\n**Core Components:**\n1. **Task Router**: Distributes tasks based"
                " on priority queues\n2. **Worker Manager**: Handles worker lifecycle and load balancing\n3."
                " **Retry Engine**: Implements exponential backoff with jitter\n4. **Dependency Graph**:"
                " Manages task dependencies using topological sorting\n5. **State Manager**: Tracks task "
                "states (pending, running, completed, failed)\n6. **Metrics Collector**: Real-time "
                "performance metrics\n\n**Data Structures:**\n- Priority queues using Redis Sorted Sets\n-"
                " Task metadata in PostgreSQL with JSONB columns\n- Distributed locks with Redis Redlock"
                " algorithm\n\n**Features:**\n- At-least-once delivery guarantee\n- Dead letter queue for "
                "failed tasks\n- Rate limiting per task type\n- Circuit breaker pattern for failing "
                "services\n\nWould you like me to start with the core task model and priority queue "
                "implementation?",
            },
            {
                "role": "user",
                "content": "That sounds good. But I also need WebSocket support for real-time task status updates, a "
                "web dashboard for monitoring, and the ability to dynamically scale workers based on queue"
                " depth. Plus, I need task deduplication and exactly-once processing semantics.",
            },
            {
                "role": "assistant",
                "content": "Perfect! Let me enhance the architecture with these advanced requirements:\n\n**Real-time "
                "Features:**\n1. **WebSocket Manager**: Using Python's websocket-server with Redis pub/sub "
                "for multi-instance communication\n2. **Event Stream**: Task state changes published to Re"
                "dis streams\n3. **Client SDK**: JavaScript/TypeScript library for real-time subscriptio"
                "ns\n\n**Auto-scaling Architecture:**\n1. **Metrics Aggregator**: Collects queue depth, "
                "processing latency, error rates\n2. **Scaler Service**: Kubernetes HPA integration or cu"
                "stom Docker Swarm scaling\n3. **Predictive Scaling**: ML model to predict load based on hi"
                "storical patterns\n\n**Exactly-once Processing:**\n1. **Idempotency Keys**: Each task has "
                "UUID-based idempotency key\n2. **Deduplication Store**: Redis with TTL for processed task "
                "IDs\n3. **Transactional Outbox**: PostgreSQL outbox pattern for reliable event publishing"
                "\n4. **Two-phase Commit**: For distributed task processing\n\n**Dashboard Features:**\n- "
                "Real-time queue metrics with D3.js visualizations\n- Task timeline view with dependency g"
                "raphs\n- Worker health monitoring with automatic restart\n- Alert system for SLA violati"
                "ons\n\n**Additional Components:**\n- Task scheduler for delayed tasks using Redis keys w"
                "ith TTL\n- Batch processing support for bulk operations\n- Multi-tenant isolation with na"
                "mespace support\n\nThis is getting quite comprehensive! Should I start with the core tas"
                "k model and database schema, or would you prefer to begin with the WebSocket real-time up"
                "date system?",
            },
            {
                "role": "user",
                "content": "This is getting complex. I also need multi-region support with eventual consistency, the "
                "ability to pause/resume task processing by task type, and A/B testing capabilities for "
                "different worker implementations. Also, I need comprehensive audit logging and GDPR comp"
                "liance features.",
            },
            {
                "role": "assistant",
                "content": "Absolutely! This is evolving into an enterprise-grade system. Let me architect the compl"
                "ete solution:\n\n**Multi-region Architecture:**\n1. **Global Load Balancer**: GeoDNS with"
                " health checks\n2. **Cross-region Replication**: Redis Cluster with active-active setup "
                "using CRDTs\n3. **Conflict Resolution**: Vector clocks for task ordering, last-writer-win"
                "s for metadata\n4. **Region-aware Routing**: Route tasks to workers in same region when p"
                "ossible\n5. **Failover Mechanism**: Automatic region failover with 30-second RTO\n\n**Adv"
                "anced Control Features:**\n1. **Task Type Governance**: \n   - Pause/resume via Redis fe"
                "ature flags with immediate propagation\n   - Rate limits per task type with burst capaci"
                "ty\n   - Resource quotas (CPU/memory) per task category\n2. **A/B Testing Framework**:\n"
                "   - Task routing based on consistent hashing of task ID\n   - Variant assignment with s"
                "tickiness\n   - Statistical significance tracking for performance metrics\n   - Automati"
                "c winner selection based on success rate and latency\n\n**Compliance & Audit:**\n1. **A"
                "udit Trail**:\n   - Immutable task history in PostgreSQL with row-level security\n   -"
                " Change data capture (CDC) using Debezium\n   - Cryptographic signing of audit logs\n "
                "  - 7-year retention policy with automated archival to S3\n2. **GDPR Compliance**:\n  "
                " - Right to be forgotten: Cascade delete with verification\n   - Data portability: JSO"
                "N export of all user tasks\n   - Consent management: Task processing consent tracking"
                "\n   - Data anonymization: PII encryption with rotating keys\n\n**Enhanced Monitoring:*"
                "*\n1. **Distributed Tracing**: OpenTelemetry integration with Jaeger\n2. **Performance "
                "Profiling**: Py-spy continuous profiling\n3. **Cost Analytics**: Per-task-type resource"
                " consumption tracking\n4. **SLA Monitoring**: 99.9% task processing latency targets\n\n"
                "**Security Features:**\n1. **Zero-trust Architecture**: mTLS between all services\n2. **"
                "Secrets Management**: HashiCorp Vault integration\n3. **Network Segmentation**: VPC is"
                "olation with private subnets\n4. **Encryption**: AES-256 at rest, TLS 1.3 in transit\n"
                "\n**Operational Excellence:**\n1. **Blue-green Deployments**: Zero-downtime updates\n2."
                " **Canary Releases**: Gradual rollout of worker changes\n3. **Chaos Engineering**: Rand"
                "om worker termination testing\n4. **Capacity Planning**: Automated infrastructure provi"
                "sioning\n\nThis is becoming a mission-critical system! Would you like me to start with "
                "the database schema and core task lifecycle management, or should we begin with the mul"
                "ti-region replication strategy? I can also create a detailed implementation roadmap wi"
                "th phases if you prefer.",
            },
            {
                "role": "user",
                "content": "How do I reverse a string in Python?",
            },
            {
                "role": "assistant",
                "content": "You can reverse a string in Python using slicing: `reversed_string = original_string[::"
                "-1]`. The `[::-1]` slice notation means start at the end of the string and end at posit"
                "ion 0, moving with the step -1.",
            },
            {
                "role": "user",
                "content": "What's the difference between a list and a tuple?",
            },
            {
                "role": "assistant",
                "content": "Lists are mutable (you can change their content after creation) and use square bracket"
                "s `[]`. Tuples are immutable and use parentheses `()`. Lists are better for collectio"
                "ns that need to change, while tuples are faster and can be used as dictionary keys.",
            },
            {
                "role": "user",
                "content": "How do I handle exceptions in Python?",
            },
            {
                "role": "assistant",
                "content": "Use try-except blocks: `try:` followed by your risky code, then `except SpecificError"
                ":` to catch particular exceptions. Always catch specific exceptions rather than the g"
                "eneric `Exception` when possible. You can also use `finally:` for cleanup code that ru"
                "ns regardless.",
            },
        ]

        # Test 1: Messages below threshold (should not compress)
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: Messages below threshold (should skip compression)")
        logger.info("=" * 60)

        compress_op1 = ContextCompressOp()

        await compress_op1.async_call(
            messages=messages,
            max_total_tokens=50000,  # High threshold, won't trigger
            keep_recent_count=2,
        )

        result_messages1 = compress_op1.context.response.answer
        logger.info(f"✓ Result: {len(result_messages1)} messages (unchanged)")

        # Test 2: Messages above threshold (should compress)
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: Messages above threshold (should compress)")
        logger.info("=" * 60)

        compress_op2 = ContextCompressOp()

        await compress_op2.async_call(
            messages=messages,
            max_total_tokens=2000,  # Low threshold, will trigger
            keep_recent_count=2,
            compress_system_message=False,
        )

        result_messages2 = compress_op2.context.response.answer
        logger.info(f"✓ Result: {len(result_messages2)} messages (compressed)")

        # Display compression results
        logger.info("\n" + "=" * 60)
        logger.info("Compression Result Details:")
        logger.info("=" * 60)
        logger.info(f"Original messages: {len(messages)}")
        logger.info(f"Compressed messages: {len(result_messages2)}")

        # Test 3: Messages above threshold (should compress)
        logger.info("\n" + "=!" * 30)
        logger.info("Test 3: Messages above micro threshold (should compress)")
        logger.info("=!" * 30)

        compress_op2 = ContextCompressOp()

        await compress_op2.async_call(
            messages=messages,
            max_total_tokens=2000,  # Low threshold, will trigger
            keep_recent_count=2,
            compress_system_message=False,  # Don't compress system messages
            group_token_threshold=1500,
        )

        result_messages2 = compress_op2.context.response.answer
        logger.info(f"✓ Result: {len(result_messages2)} messages (compressed)")

        # Display compression results
        logger.info("\n" + "=" * 60)
        logger.info("Compression Result Details:")
        logger.info("=" * 60)
        logger.info(f"Original messages: {len(messages)}")
        logger.info(f"Compressed messages: {len(result_messages2)}")


if __name__ == "__main__":
    asyncio.run(main())
