# week1_serve/05_ui.py
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

def respond(message, history, system_prompt, temperature):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:                        # gradio messages format
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature,
        max_tokens=800, stream=True,
    )
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            yield partial

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    additional_inputs=[
        gr.Textbox("You are a helpful, concise assistant.", label="System prompt"),
        gr.Slider(0.0, 1.5, value=0.7, label="Temperature"),
    ],
    title="Local SmolLM2-135M",
    description="Served by vLLM. 135M parameters, under 1 GB of VRAM.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
