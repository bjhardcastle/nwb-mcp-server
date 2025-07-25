[MCP Tool Output Standard for Tabular Data](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/930)

# Intro
- brief timeline of Copilot tools in VSCode (auto-complete, small-snippets, large-scale edits, agent mode)
- discussed with Corbett in 2023 the idea of natural-language analysis (Minority Report style -
  image) - all the tools to that are available right now 
- agent mode allows them to do proper work: generate a plan, write code, run it, check the output,
  iterate

# Levels of analysis of idea
## idea
    - input hypothesis in natural language
    - get a report with summary, figures, tables, statistics, etc.
## tools
    - interact with LLM chatbot
    - LLM uses MCP to access data and formulate a plan
        - data is a directory of NWB files
        - MCP provides access to NWB files as SQL tables
        - LLM can query the data using SQL strings and gets results back as JSON strings
    - writes queries, runs them, checks the output
    - iterates until satisfied with the output

# Observations 
- no metadata - just good naming of tables, columns etc.
    - providing metadata likely to be beneficial
- a sensible API for NWB files was necessary to make this work
- lazy-loading of NWB files is a must (pynwb is too slow, uses too much memory)
- o4-mini is not good at interpreting available MCP servers/tools
- instructions to FastMCP class are not followed in Copilot Chat

# Vision 
- user can throw out a hypothesis in natural language (in teams, say)
- multiple agents can run, in parallel, 24/7, they're fast, expert users of all the tools and
  techniques, don't get tired
- agents can generate their own hypotheses based on previous findings

# Questions for discussion (or to investigate and pre-emptively answer)
- tweaking the instructions/prompt to get good results
- how trustworthy is the output?
  - run multiple times, or with different models and compare results?
  - manually check the output? (somewhat defeats the point)
  - write tests that demonstrate correctness of small building blocks
  - does the agent "aim to please" and cherry-pick, misrepresent, fabricate data?
- how much would it cost if using an API?
- can we run it in codeocean as a streamlit app or similar (instead of in VScode)?
- benchmark against human experts

# Prompts:
- "summarize performance in these sessions"
- "test the hypothesis that performance gets worse with age. consider the effect of performance
  individuals over sessions (age is negligible from first session to last)"
