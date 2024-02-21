
from audio_recognition import AudioRecognition
from query_llm  import QueryLLM
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--audio_file_path", type=str, default="src/audio/sample.mp3")
argparse.add_argument("--model", type=str, default="llama2")

if __name__ == "__main__":

    args = argparse.parse_args()
    audio_file_path = args.audio_file_path

    # Audio transcription class
    audio_recognition = AudioRecognition()
    # LLM class
    query_llm = QueryLLM()
    query = audio_recognition.forward(audio_file_path)
    response = query_llm.query(query['text'])
    print(response)
