import gradio as gr
import time
from groq import Groq
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wav
import os
from gtts import gTTS

api_key = "<your-api-key>"

gclient = Groq(
    api_key=api_key,
)

models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {
        "name": "Mixtral-8x7b-Instruct-v0.1",
        "tokens": 32768,
        "developer": "Mistral",
    },
}


def stream_to_wav(stream, sample_rate, output_filename="audio.wav"):
    # Ensure the stream is in the correct data type
    stream = (stream * 32767).astype(np.int16)

    # Write the WAV file
    wav.write(output_filename, sample_rate, stream)

    print(f"WAV file saved as {output_filename}")


def text_to_speech(translated_text):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text)
    my_obj.save(file_name)
    return file_name


def chat(message, history):
    # Prepare the conversation history for the API call
    conversation = "\n".join([f"Human: {h[0]}\nSara: {h[1]}" for h in history])
    conversation += f"\nHuman: {message}\nSara:"
    if len(conversation) > 2500:
        conversation = f"\nHuman: {message}\nSara:"

    chat_completion = gclient.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are Sara, a speaking Voice Assistant System. Answer in less than 50 words. Do not add quotes",
            },
            {"role": "user", "content": conversation},
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content


def transcribe(new_chunk, history):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    stream_to_wav(y, sample_rate=sr)
    wav_filename = "audio.wav"

    with open("audio.wav", "rb") as file:
        transcription = gclient.audio.transcriptions.create(
            file=(wav_filename, file),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            language="en",
            temperature=0.0,
        )

    os.remove(wav_filename)

    text = transcription.text
    out = chat(text, history)

    # Update history
    history.append((text, out))

    audio_file = text_to_speech(out)

    print(f"Human: {text}")
    print(f"Sara: {out}")

    return gr.Audio(audio_file), history


# Interface for Simple apps
"""demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type="numpy", label="Speak"),
        gr.State([]),  # This will store the conversation history
    ],
    outputs=[
        gr.Audio(label="Sara's response"),
        gr.State(),  # This will receive and store the updated history
    ],
    title="Sara - Voice Assistant",
    description="Speak to Sara, your AI voice assistant",
    allow_flagging="never",
    clear_btn=None ,
)"""


css = """
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
}
.custom-audio {
    border: 2px solid #4a90e2;
    border-radius: 10px;
    padding: 10px;
    background-color: rgba(74, 144, 226, 0.1);
    box-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
    margin-bottom: 20px;
}
.gr-button {
    border: none;
    color: #fff;
    background: #4a90e2;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-right: 10px;
}
.gr-button:hover {
    background: #357ab7;
}
.gr-row {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
.gr-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=css,
) as demo:
    gr.Markdown(
        """
        # 🌟 Sara - AI Voice Assistant 🌟
        Welcome to the future of voice interaction!
        """
    )

    with gr.Column(elem_id="audio-container"):
        audio_io = gr.Audio(
            sources="microphone",
            type="numpy",
            label="🎙️ Speak your command & hear Sara's response",
            elem_classes="custom-audio",
        )

    with gr.Row():
        next_btn = gr.Button("🚀 Next Command", variant="primary")
        clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

    history = gr.State([])

    audio_io.stop_recording(
        transcribe, inputs=[audio_io, history], outputs=[audio_io, history]
    )

    def start_new_recording():
        # This function will be called when the button is clicked
        return None, None  # Reset the audio input and output

    def clear_history():
        return [], None, None

    next_btn.click(
        fn=start_new_recording,
        inputs=None,
        outputs=[audio_io, audio_io],
        js="() => {document.querySelector('#component-3 button').click(); return [null, null];}",
    )

    clear_btn.click(
        fn=clear_history, inputs=None, outputs=[history, audio_io, audio_io]
    )

    gr.Markdown(
        """
        ### Experience the power of AI-driven conversation!
        Speak clearly and listen to Sara's intelligent responses.
        """
    )

demo.launch()
