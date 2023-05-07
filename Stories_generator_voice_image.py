import os

import replicate
import streamlit as st
from elevenlabs import generate, voices
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
eleven_api_key = st.secrets["eleven_api_key"]
os.environ["REPLICATE_API_TOKEN"] = st.secrets["replicate_api_token"] 

voces = voices()

llm = OpenAI(temperature=0.9)

def generate_story(text):
    """Genera una historia en idioma español utilizando la libreria de langchain y el modelo GPT-3 de OpenAI."""

    prompt = PromptTemplate(
        input_variables=["text"],
        template=""" 
         Eres un narrador divertido y experimentado. Genera una historia de máximo 30 palabras para mí acerca de {text}.
                 """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(text=text)


def generate_audio(text, voice):
    """Convierte la historia generada a audio utilizando el API de Eleven Labs."""
    audio = generate(text=text, voice=voice, model="eleven_multilingual_v1", api_key=eleven_api_key)
    return audio


def generate_images(story_text):
    """Genera imágenes utilizando el texto de la historia utilizando el API de Replicate."""
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": story_text}
    )
    return output


def app():
    st.title("Generador automático de historias con voz e imágenes")

    with st.form(key='my_form'):
        text = st.text_input(
            "Introduce una frase para generar una historia",
            max_chars=None,
            type="default",
            placeholder="Introduce una frase para generar una historia",
        )
        options = [voces[0].name, voces[1].name, voces[2].name, voces[3].name, voces[4].name, voces[5].name, voces[6].name, voces[7].name, voces[8].name]
        voice = st.selectbox("Selecciona una voz", options)

        if st.form_submit_button("Enviar"):
            with st.spinner('Generando la historia...'):
                story_text = generate_story(text)
                audio = generate_audio(story_text, voice)

            st.audio(audio, format='audio/mp3')
            images = generate_images(story_text)
            for item in images:
                st.image(item)

    if not text or not voice:
        st.info("Por favor, introduce una palabra y selecciona una voz")


if __name__ == '__main__':
    app()
