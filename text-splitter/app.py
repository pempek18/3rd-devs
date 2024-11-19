import os
import json
from TextService import TextSplitter

splitter = TextSplitter()

async def process_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    docs = await splitter.split(text, 1000)
    json_file_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}.json"
    )
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2)

    chunk_sizes = [doc['metadata']['tokens'] for doc in docs]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
    min_chunk_size = min(chunk_sizes)
    max_chunk_size = max(chunk_sizes)
    sorted_sizes = sorted(chunk_sizes)
    median_chunk_size = sorted_sizes[len(sorted_sizes) // 2]

    return {
        'file': os.path.basename(file_path),
        'avgChunkSize': f"{avg_chunk_size:.2f}",
        'medianChunkSize': median_chunk_size,
        'minChunkSize': min_chunk_size,
        'maxChunkSize': max_chunk_size,
        'totalChunks': len(chunk_sizes)
    }

async def main():
    # Get all markdown files in the current directory
    directory_path = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(directory_path)
    reports = []

    for file in files:
        if file.endswith('.md'):
            report = await process_file(os.path.join(directory_path, file))
            reports.append(report)

    # Print table using pandas for better formatting
    import pandas as pd
    df = pd.DataFrame(reports)
    print(df.to_string(index=False))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())