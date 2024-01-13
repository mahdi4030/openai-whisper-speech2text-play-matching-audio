from sentence_transformers import SentenceTransformer, util
import openai
import asyncio
import whisper
import pydub
from pydub import playback
import speech_recognition as sr
import json
import os
import torch

# Initialize the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()

audio_path = 'audio/'

def play_audio(file):
  sound = pydub.AudioSegment.from_file(audio_path + file, format="mp3")
  playback.play(sound)

async def main():
  embedder = SentenceTransformer('all-MiniLM-L6-v2')

  try:
    with open("./database.json", 'r') as f:
      contents = json.load(f)
      corpus = list(contents.keys())
      corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
  except Exception as e:
    print(e)

  while True:
    with sr.Microphone() as source:
      recognizer.adjust_for_ambient_noise(source)
      print(f"----- Waiting for new question -----\n")
      audio = recognizer.listen(source)
      try:
        with open("audio.wav", "wb") as f:
          f.write(audio.get_wav_data())
        # Use the preloaded tiny_model
        model = whisper.load_model("tiny.en")
        result = model.transcribe("audio.wav", fp16=False)
        question = result["text"]
        print("You asked : ", question, "\n")

        if question == "I will quit":
          break

        queries = [question]
        top_k = min(1, len(corpus))
        for query in queries:
          query_embedding = embedder.encode(query, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        top_score = -1
        top_score_question = ''
        for score, idx in zip(top_results[0], top_results[1]):
          if top_score < score:
            top_score_question = corpus[idx]
        
        if top_score < 0.6:
          play_audio("score is too low.mp3")
          print(corpus[idx], "(Score: {:.4f})".format(score), " -> audio : score is too low.mp3", "\n")
        else:
          play_audio(contents[top_score_question])
          print(corpus[idx], "(Score: {:.4f})".format(score), " -> audio : ", contents[top_score_question], "\n")
        print("\n\n")
      except Exception as e:
        print("Error : {0}".format(e))
        continue

if __name__ == "__main__":
  asyncio.run(main())
