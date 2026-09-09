from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}

    # -----------------------
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("\n"+" ="*50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [("user", f"find recent, reliable and detailed information about: {topic}")]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("Search Result = ", state["search_results"])

    # ---------------

    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("\n"+" ="*50)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages" : [("user",
            f"Based on following search results about '{topic}",
            f"pick the most relevant URL and scrap it for deeper content. \n\n",
            f"Search results:\n{state["search_results"][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\n Scraped Content \n", state['scraped_content'])

    # --------------

    