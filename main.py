#!/usr/bin/env python
"""
Command-line LLM-based bot that answers questions about scientific papers.
"""
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Frame, TextArea

import llm

cached = None

llm_path = TextArea(
    prompt="LLM path: ",
    text="./models/mistral-7b-instruct-v0.2.Q4_0.gguf",
    multiline=False,
    focus_on_click=True,
)
emb_model_name = TextArea(
    prompt="Embedding model name: ",
    text="all-MiniLM-L6-v2",
    multiline=False,
    focus_on_click=True,
    name="2",
)
paper_url = TextArea(
    prompt="Paper URL: ",
    multiline=False,
    text="",
    focus_on_click=True,
)
query = TextArea(
    multiline=False,
    text="",
    focus_on_click=True,
)
answer = TextArea(
    multiline=True, read_only=True, text="", focus_on_click=True, scrollbar=True
)

root_container = HSplit(
    [
        llm_path,
        emb_model_name,
        paper_url,
        Frame(query, title="Your question"),
        Frame(answer, title="Answer"),
    ]
)
layout = Layout(container=root_container)
layout.focus(query)


# Key bindings.
kb = KeyBindings()


@kb.add("c-c")
def exit(event):
    "Quit when control-c is pressed."
    event.app.exit()


@kb.add("enter")
async def run(event):
    "Run inference when Enter is pressed."
    if not query.text.strip():
        return
    answer.text = "Loading the model..."
    event.app._redraw()
    global cached
    chain, cached = llm.get_chain(
        paper_url=paper_url.text,
        llm_path=llm_path.text,
        emb_model_name=emb_model_name.text,
        n_chunks=5,
        cached=cached,
    )
    answer.text = "Thinking..."
    event.app._redraw()
    cur_answer = ""
    for chunk in chain.stream({"query": query.text.strip()}):
        cur_answer += chunk
        answer.text = cur_answer
        event.app._redraw()


# Build a main application object.
application = Application(
    layout=layout,
    key_bindings=kb,
    full_screen=True,
    mouse_support=True,
)


def main():
    application.run()


if __name__ == "__main__":
    main()
