import importlib

try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Streamlit is required. Install it with: python -m pip install streamlit"
    ) from exc

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 AI Research Assistant")
st.markdown(
    "Search the web → Read relevant sources → "
    "Generate a report → Critically review it."
)


# ============================================================
# RESEARCH PIPELINE
# ============================================================

def run_research_pipeline(topic: str):

    state = {}

    # --------------------------------------------------------
    # STEP 1 - SEARCH
    # --------------------------------------------------------

    with st.status("🔎 Searching the web...", expanded=True) as status:

        search_agent = build_search_agent()

        search_result = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
                    Find recent, reliable and detailed information
                    about: {topic}

                    Return useful sources with titles, URLs,
                    and relevant information.
                    """
                )
            ]
        })

        state["search_results"] = (
            search_result["messages"][-1].content
        )

        st.write("Search completed.")

        status.update(
            label="✅ Web search completed",
            state="complete"
        )


    # --------------------------------------------------------
    # STEP 2 - READER
    # --------------------------------------------------------

    with st.status("📖 Reading the best source...", expanded=True) as status:

        reader_agent = build_reader_agent()

        reader_prompt = f"""
        Based on the following search results about "{topic}":

        Pick the most relevant and reliable URL.

        Use the scrape tool to read that URL and return
        detailed information useful for writing a research report.

        Search Results:
        {state["search_results"][:4000]}
        """

        reader_result = reader_agent.invoke({
            "messages": [
                ("user", reader_prompt)
            ]
        })

        state["scraped_content"] = (
            reader_result["messages"][-1].content
        )

        st.write("Source reading completed.")

        status.update(
            label="✅ Source reading completed",
            state="complete"
        )


    # --------------------------------------------------------
    # STEP 3 - WRITER
    # --------------------------------------------------------

    with st.status("✍️ Writing research report...", expanded=True) as status:

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

        status.update(
            label="✅ Research report generated",
            state="complete"
        )


    # --------------------------------------------------------
    # STEP 4 - CRITIC
    # --------------------------------------------------------

    with st.status("🧐 Reviewing the report...", expanded=True) as status:

        state["feedback"] = critic_chain.invoke({
            "report": state["report"]
        })

        status.update(
            label="✅ Report reviewed",
            state="complete"
        )


    return state


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    st.info(
        "Enter a topic and the AI will search for information, "
        "read a relevant source, write a report, and review it."
    )


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(
    "Research Topic",
    placeholder="Example: Transformers in Generative AI"
)


# ============================================================
# RUN BUTTON
# ============================================================

if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        try:

            result = run_research_pipeline(topic)

            st.session_state["research_result"] = result

        except Exception as e:

            st.error("Something went wrong.")

            st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "research_result" in st.session_state:

    result = st.session_state["research_result"]

    st.divider()

    # --------------------------------------------------------
    # SEARCH RESULTS
    # --------------------------------------------------------

    with st.expander("🔎 Search Results", expanded=False):

        st.markdown(
            result["search_results"]
        )


    # --------------------------------------------------------
    # SCRAPED CONTENT
    # --------------------------------------------------------

    with st.expander("📖 Detailed Source Content", expanded=False):

        st.markdown(
            result["scraped_content"]
        )


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    st.header("📑 Research Report")

    st.markdown(
        result["report"]
    )


    # --------------------------------------------------------
    # CRITIC FEEDBACK
    # --------------------------------------------------------

    st.header("🧐 Critic Review")

    st.markdown(
        result["feedback"]
    )


    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------

    st.download_button(
        label="📥 Download Report",
        data=result["report"],
        file_name=f"{topic.replace(' ', '_')}_research_report.txt",
        mime="text/plain",
        use_container_width=True
    )
