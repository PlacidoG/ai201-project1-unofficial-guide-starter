"""Gradio query interface (Milestone 5).

Run it:

    python app.py

then open http://localhost:7860
"""

import gradio as gr

from query import ask, format_examples   # retrieve -> Groq Llama 3.3 -> answer + examples + sources


def handle_query(question):
    result = ask(question)
    examples = format_examples(result["examples"])   # empty for simple questions
    sources = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], examples, sources


with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about UHD Computer Science professors — answers are grounded in student "
        "reviews (RateMyProfessors + Reddit) retrieved from the vector store."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which professor is most lenient with late work?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    examples = gr.Textbox(
        label="Supporting reviews",
        lines=10,
        placeholder="Example reviews backing the answer (shown only for in-depth questions).",
    )
    sources = gr.Textbox(label="Retrieved from", lines=4)

    outputs = [answer, examples, sources]
    btn.click(handle_query, inputs=inp, outputs=outputs)
    inp.submit(handle_query, inputs=inp, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
