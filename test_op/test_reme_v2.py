import asyncio


async def main():
    from reme_ai import ReMeApp
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the meaning of life?"
        },
        {
            "role": "assistant",
            "content": "The meaning of life is to live life to the fullest and enjoy every moment of it."
        },
    ]
    async with ReMeApp() as app:
        from reme_ai.core.meta_mem_summarizer_op import BaseMemSummarizerOp

        meta_mem_summarizer = BaseMemSummarizerOp()

        await meta_mem_summarizer.async_call(messages=messages)


if __name__ == "__main__":
    asyncio.run(main())
