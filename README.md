# Multi-Agent Research System

An interactive AI-powered research assistant built with **Python**, **Streamlit**, and multi-agent workflows. This system leverages specialized agents to automate information gathering, synthesis, and analysis to streamline research tasks.

🚀 **Live Demo:** [Multi-Agent Research System Streamlit App](https://multi-agent-research-system-ejzkqjr7uwexjtrbxchmme.streamlit.app/)

---

## 📸 Features

* **Multi-Agent Collaboration**: Orchestrates multiple specialized agents (`agents.py`) working in sequence/parallel to tackle research queries.
* **Custom Tools Integration**: Extends agent capabilities using targeted tools defined in `tools.py`.
* **Interactive UI**: Clean, easy-to-use user interface powered by Streamlit (`app.py`).
* **DevContainer Ready**: Pre-configured environment settings for quick containerized setup and development.

---

## 🛠️ Tech Stack

* **Language**: Python
* **UI Framework**: [Streamlit](https://streamlit.io/)
* **Architecture**: Multi-Agent Orchestration (`pipeline.py`, `agents.py`, `tools.py`)

---

## 📁 Repository Structure

```text
├── .devcontainer/     # DevContainer configuration files
├── agents.py          # Agent definitions, prompts, and roles
├── app.py             # Main Streamlit application entry point
├── pipeline.py        # Pipeline execution logic for coordinating agents
├── requirements.txt   # Python dependencies
├── tools.py           # Custom tools and function utilities for agents
└── README.md          # Project documentation
