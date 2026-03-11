import asyncio
import json

import aiohttp

# API base URL
base_url = "http://0.0.0.0:8002"


async def main():
    # Create a unique workspace ID
    workspace_id = "personal_memory_demo"

    async with aiohttp.ClientSession() as session:
        # Step 1: Clear existing memories in the workspace
        print("Clearing existing memories...")
        async with session.post(
            f"{base_url}/vector_store",
            json={
                "action": "delete",
                "workspace_id": workspace_id,
            },
            headers={"Content-Type": "application/json"},
        ) as response:
            result = await response.json()
            print(json.dumps(result, ensure_ascii=False))

        # Step 2: Create a conversation with rich personal information
        print("\nCreating conversation with personal information...")
        messages = [
            {
                "role": "user",
                "content": "My name is John Smith, I'm 28 years old, and I work at a tech company in San Francisco",
            },
            {"role": "assistant", "content": "Nice to meet you, John!"},
            {
                "role": "user",
                "content": "I'm a software engineer, mainly doing backend development using Python and Go",
            },
            {"role": "assistant", "content": "I see, you're a backend engineer working with Python and Go."},
            {
                "role": "user",
                "content": "I enjoy playing basketball and watching sci-fi movies. I recently watched Dune Part 2",
            },
            {
                "role": "assistant",
                "content": "Basketball and sci-fi movies are great hobbies! Dune Part 2 was indeed amazing.",
            },
            {"role": "user", "content": "I have a cat named Shadow who is 3 years old"},
            {"role": "assistant", "content": "Shadow sounds adorable! 3-year-old cats are quite playful."},
            {"role": "user", "content": "I'm planning a trip to Japan next month, mainly to Tokyo and Kyoto"},
            {
                "role": "assistant",
                "content": "Your Japan trip sounds exciting! Tokyo and Kyoto are both wonderful destinations with their own unique charm.",
            },
            {"role": "user", "content": "I'm really interested in Japanese cuisine, especially sushi and ramen"},
            {
                "role": "assistant",
                "content": "Japanese cuisine is delicious! Sushi and ramen are very popular choices.",
            },
        ]

        # Step 3: Summarize personal memories from the conversation
        print("\nSummarizing personal memories...")
        async with session.post(
            f"{base_url}/summary_personal_memory",
            json={
                "trajectories": [
                    {"messages": messages, "score": 1.0},
                ],
                "workspace_id": workspace_id,
            },
            headers={"Content-Type": "application/json"},
        ) as response:
            result = await response.json()
            result = json.dumps(result, ensure_ascii=False, indent=2)
            print(result)

            with open("personal_memory.jsonl", "w") as f:
                f.write(result)

        # Wait for the memories to be processed and stored
        print("\nWaiting for memories to be processed...")
        await asyncio.sleep(2)

        # Step 4: Retrieve personal memories with different queries
        queries = [
            "What's my name and age?",
            "What do I do for work?",
            "What are my hobbies?",
            "Do I have any pets?",
            "What are my travel plans?",
            "What foods do I like?",
        ]

        print("\nRetrieving personal memories...")
        for query in queries:
            print(f"\nQuery: {query}")
            async with session.post(
                f"{base_url}/retrieve_personal_memory",
                json={
                    "query": query,
                    "workspace_id": workspace_id,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                result = await response.json()
                print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
