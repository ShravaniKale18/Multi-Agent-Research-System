from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # ============================================================
    # STEP 1 - SEARCH
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""
                Find recent, reliable and detailed information
                about: {topic}
                """
            )
        ]
    })

    state["search_results"] = (
        search_result["messages"][-1].content
    )

    print("\nSearch Results:\n")
    print(state["search_results"])


    # ============================================================
    # STEP 2 - READER
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 2 - Reader agent is working...")
    print("=" * 50)

    reader_agent = build_reader_agent()

    reader_prompt = f"""
    Based on the following search results about "{topic}":

    Pick the most relevant URL and scrape it for deeper content.

    Search Results:
    {state["search_results"][:800]}
    """

    reader_result = reader_agent.invoke({
        "messages": [
            ("user", reader_prompt)
        ]
    })

    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )

    print("\nScraped Content:\n")
    print(state["scraped_content"])


    # ============================================================
    # STEP 3 - WRITER
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = f"""
    SEARCH RESULTS:
    {state["search_results"]}

    DETAILED SCRAPED CONTENT:
    {state["scraped_content"]}
    """

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report:\n")
    print(state["report"])


    # ============================================================
    # STEP 4 - CRITIC
    # ============================================================

    print("\n" + "=" * 50)
    print("STEP 4 - Reviewing the report...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCritic Report:\n")
    print(state["feedback"])


    return state


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    topic = input("\nEnter a Research Topic: ")

    run_research_pipeline(topic)