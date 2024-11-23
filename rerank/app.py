from typing import List, Dict, Any
from OpenAIService import OpenAIService
from TextService import TextSplitter
from VectorService import VectorService

data = [
    {"author": "Jim Collins", "text": "Good to Great: \"Good is the enemy of great. To go from good to great requires transcending the curse of competence.\""},
    {"author": "Jim Collins", "text": "Built to Last: \"Clock building, not time telling. Focus on building an organization that can prosper far beyond the presence of any single leader and through multiple product life cycles.\""},
    {"author": "Jim Collins", "text": "Great by Choice: \"20 Mile March. Achieve consistent performance markers, in good times and bad, as a way to build resilience and maintain steady growth.\""},
    {"author": "Jim Collins", "text": "How the Mighty Fall: \"Five stages of decline: hubris born of success, undisciplined pursuit of more, denial of risk and peril, grasping for salvation, and capitulation to irrelevance or death.\""},
    {"author": "Jim Collins", "text": "Beyond Entrepreneurship 2.0: \"The flywheel effect. Success comes from consistently pushing in a single direction, gaining momentum over time.\""},
    {"author": "Jim Collins", "text": "Turning the Flywheel: \"Disciplined people, thought, and action. Great organizations are built on a foundation of disciplined individuals who engage in disciplined thought and take disciplined action.\""},
    {"author": "Jim Collins", "text": "Built to Last: \"Preserve the core, stimulate progress. Enduring great companies maintain their core values and purpose while their business strategies and operating practices endlessly adapt to a changing world.\""},
    {"author": "Jim Collins", "text": "Good to Great: \"First who, then what. Get the right people on the bus, the wrong people off the bus, and the right people in the right seats before you figure out where to drive it.\""},
    {"author": "Simon Sinek", "text": "Start with Why: \"People don't buy what you do; they buy why you do it. And what you do simply proves what you believe.\""},
    {"author": "Simon Sinek", "text": "Leaders Eat Last: \"The true price of leadership is the willingness to place the needs of others above your own. Great leaders truly care about those they are privileged to lead and understand that the true cost of the leadership privilege comes at the expense of self-interest.\""},
    {"author": "Simon Sinek", "text": "The Infinite Game: \"In the Infinite Game, the true value of an organization cannot be measured by the success it has achieved based on a set of arbitrary metrics over arbitrary time frames. The true value of an organization is measured by the desire others have to contribute to that organization's ability to keep succeeding, not just during the time they are there, but well beyond their own tenure.\""}
]

query = "What does Sinek and Collins said about working with people?"
COLLECTION_NAME = "aidevs"

async def initialize_data(openai: OpenAIService, 
                        vector_service: VectorService, 
                        text_splitter: TextSplitter) -> None:
    points = []
    for item in data:
        doc = await text_splitter.document(item["text"], "gpt-4o", {"author": item["author"]})
        points.append(doc)

    await vector_service.initialize_collection_with_data(COLLECTION_NAME, points)

async def main():
    openai = OpenAIService()
    vector_service = VectorService(openai)
    text_splitter = TextSplitter()

    await initialize_data(openai, vector_service, text_splitter)

    # Determine authors
    determine_author = await openai.completion(
        messages=[
            {"role": "system", "content": """You are a helpful assistant that determines the author(s) of a given text. 
                                        Pick between Jim Collins and Simon Sinek. If both are relevant, list them comma-separated. Write back with the name(s) and nothing else."""},
            {"role": "user", "content": query}
        ]
    )

    authors = determine_author.choices[0].message.content.split(',') if determine_author.choices[0].message.content else []
    authors = [a.strip() for a in authors]

    # Create filter based on authors
    filter_dict = {
        "should": [
            {
                "key": "author",
                "match": {"value": author}
            } for author in authors
        ]
    } if authors else None

    # Perform search
    search_results = await vector_service.perform_search(COLLECTION_NAME, query, filter_dict, 15)

    # Check relevance
    relevance_checks = []
    for result in search_results:
        relevance_check = await openai.completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines if a given text is relevant to a query. Respond with 1 if relevant, 0 if not relevant."},
                {"role": "user", "content": f"Query: {query}\nText: {result.payload.get('text', '')}"}
            ]
        )
        is_relevant = relevance_check.choices[0].message.content == "1"
        result_dict = dict(result)
        result_dict["is_relevant"] = is_relevant
        relevance_checks.append(result_dict)

    relevant_results = [result for result in relevance_checks if result["is_relevant"]]

    # Print results
    print(f"Query: {query}")
    print(f"Author(s): {', '.join(authors)}")
    
    # Create table-like output
    print("\n{:<20} {:<50} {:<10}".format("Author", "Text", "Score"))
    print("-" * 80)
    for result in relevant_results:
        author = result.get("payload", {}).get("author", "")
        text = result.get("payload", {}).get("text", "")
        text = f"{text[:45]}..." if text else ""
        score = result.get("score", 0)
        print("{:<20} {:<50} {:<10.4f}".format(author, text, score))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 