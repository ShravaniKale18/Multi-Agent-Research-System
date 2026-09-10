from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LLM

# llm = ChatOpenAI(
#     model="openai/gpt-oss-20b:deepinfra",
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.getenv("HF_TOKEN"),
#     temperature=0
# )

llm = ChatOpenAI(
    model="Qwen/Qwen3.5-9B:deepinfra",
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    temperature=0
)


# ============================================================
# SEARCH AGENT

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# ============================================================
# READER AGENT

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# ============================================================
# WRITER CHAIN

parser = StrOutputParser()

write_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.

        Your job is to produce a factual, evidence-based research report.

        IMPORTANT RULES:
        1. Use ONLY information contained in the research provided.
        2. Do NOT invent facts, statistics, dates, company investments, valuations, events, or sources.
        3. Clearly distinguish between:
        - verified facts
        - forecasts or predictions
        - opinions
        - uncertain/unverified claims
        4. Never present a future prediction as an established historical fact.
        5. If a claim cannot be verified from the supplied research, say that it is unverified or omit it.
        6. Preserve important URLs from the research.
        7. Do not fabricate citations or URLs.
        8. Prefer specific evidence over generic statements.
        9. Write in a professional research-report style.
        """
            ),
            (
                "human",
                """Write a detailed research report on the following topic.

        Topic:
        {topic}

        Research Gathered:
        {research}

        Use this structure:

        # Research Report: [Topic]

        ## 1. Introduction
        Briefly explain the topic and the purpose/scope of the research.

        ## 2. Key Findings

        ### 2.1 [Finding]
        Explain the finding using evidence from the supplied research.

        ### 2.2 [Finding]
        Explain the finding using evidence from the supplied research.

        ### 2.3 [Finding]
        Explain the finding using evidence from the supplied research.

        Add additional findings only when supported by the research.

        ## 3. Evidence and Verification
        Identify important claims and indicate whether they are:
        - well-supported by the supplied sources
        - forecasts/predictions
        - uncertain or insufficiently supported

        Do not treat predictions as facts.

        ## 4. Conclusion
        Summarize the strongest evidence-based conclusions.

        ## 5. Sources
        List only URLs that actually appear in the supplied research.

        IMPORTANT:
        Do not create information that is not present in the research.
        Do not invent dates, statistics, financial figures, company announcements, or URLs.
        If the research contains conflicting information, explicitly mention the conflict.
        """
    ),
])


writer_chain = write_prompt | llm | parser


# CRITIC CHAIN

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict research fact-checker and research-quality critic.

        Your primary goal is to detect:
        - fabricated facts
        - unsupported statistics
        - hallucinated events
        - incorrect dates
        - unsupported financial figures
        - future predictions presented as facts
        - unreliable or insufficient sources
        - contradictions between claims and sources

        Be strict but constructive.

        A report with good writing but serious factual problems must receive a low score.
        """
            ),
            (
                "human",
                """Critically evaluate the research report below.

        Report:
        {report}

        Evaluate it according to these criteria:

        1. Factual accuracy
        2. Evidence supporting major claims
        3. Source credibility
        4. Correct treatment of forecasts vs historical facts
        5. Accuracy of dates, statistics, financial figures, and events
        6. Completeness
        7. Structure and clarity
        8. Professional research quality

        Scoring guide:

        9-10 = Excellent research quality; claims are well-supported and factual.
        7-8 = Good report with minor weaknesses.
        5-6 = Mixed quality; several claims need verification.
        3-4 = Significant factual/evidence problems.
        1-2 = Major hallucinations, fabricated claims, or unreliable evidence.

        IMPORTANT:
        A polished writing style must NOT compensate for factual inaccuracies.

        Respond in exactly this format:

        Score: X/10

        Strengths:
        - ...
        - ...
        - ...

        Factual Issues:
        - ...
        - ...
        - ...

        Source/Evidence Issues:
        - ...
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...
        - ...

        One line verdict:
        ...
        """
    ),
])


critic_chain = critic_prompt | llm | parser