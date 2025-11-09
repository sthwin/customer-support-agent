from agents import Runner
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
import streamlit as st


class CustomWorkflow(VoiceWorkflowBase):

    def __init__(self, context):

        self.context = context

    # transcription은 파이프라인에서 전달받게됨.
    async def run(self, transcription: str):

        result = Runner.run_streamed(
            st.session_state["agent"],
            transcription,
            session=st.session_state["session"],
            context=self.context,
        )

        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        st.session_state["agent"] = result.last_agent
