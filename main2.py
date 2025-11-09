import asyncio
from agents.voice import VoicePipeline, AudioInput
import streamlit as st
from agents import (
    Agent,
    OutputGuardrailTripwireTriggered,
    Runner,
    SQLiteSession,
    InputGuardrailTripwireTriggered,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile
from models import UserAccountContext
from my_agents.triage_agent import triage_agent
import numpy as np
import wave, io
import sounddevice as sd

from workflow import CustomWorkflow

user_account_ctx = UserAccountContext(customer_id=1, name="teddy", tier="basic")

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        session_id="chat-history",
        db_path="customer-support-memory.db",
    )
session: SQLiteSession = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent


def convert_audio(audio_input: UploadedFile):
    audio_data = audio_input.getvalue()

    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        audio_frames = wav_file.readframes(-1)

    return np.frombuffer(audio_frames, dtype=np.int16)


async def run_agent(audio_input: UploadedFile):
    with st.chat_message("ai"):
        status_container = st.status("⏳ Processing void message...")
        try:

            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)

            workflow = CustomWorkflow(context=user_account_ctx)

            pipeline = VoicePipeline(workflow=workflow)
            status_container.update(label="Running workflow", state="running")

            result = await pipeline.run(audio_input=audio)

            player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
            
            player.start()
            
            status_container.update(state="complete")
            
            async for event in result.stream():
                
                if event.type == 'voice_stream_event_audio':
                    player.write(event.data)
                
                

        except InputGuardrailTripwireTriggered:
            st.write("그 질문은 제가 도와드릴 수 없어요.")

        except OutputGuardrailTripwireTriggered:
            st.write("그 질문에 답변할 수 없습니다.")


audio_input = st.audio_input(
    "Record your message",
)

if audio_input:

    with st.chat_message("human"):
        st.audio(audio_input)
    asyncio.run(run_agent(audio_input))

with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
