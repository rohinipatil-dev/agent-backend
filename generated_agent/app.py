import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from gensim.summarization import summarize
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_transcript = " ".join([x['text'] for x in transcript])
    return full_transcript

def summarize_text(text):
    return summarize(text)

def chat_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    return response.choices[0].text.strip()

def main():
    st.title("YouTube Video Summarizer")
    video_url = st.text_input('Enter a YouTube Video URL:')
    if video_url:
        video_id = video_url.split('watch?v=')[-1]
        transcript = fetch_transcript(video_id)
        summary = summarize_text(transcript)
        st.write(summary)

        st.title("Chat with AI")
        prompt = st.text_input('Enter a message to chat with AI:')
        if prompt:
            response = chat_with_gpt3(prompt)
            st.write(response)

if __name__ == "__main__":
    main()