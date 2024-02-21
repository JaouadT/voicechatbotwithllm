import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

class AudioRecognition(object):
    def __init__(self, model_id="openai/whisper-large-v3"):
        
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'mps'
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def forward(self, audio_file_path):
        
        audio = self.read_audio(audio_file_path)
        transcriber = self.initialize_transcriber()
        transcriber_pipeline = self.initialize_processor(transcriber)

        text = transcriber_pipeline(audio)

        return text

    def read_audio(self, audio_file_path):
        audio = sf.read(audio_file_path)
        return audio[0]
    
    def initialize_transcriber(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        return model
    
    def initialize_processor(self, model):
        processor = AutoProcessor.from_pretrained(self.model_id)
        transcribe_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        return transcribe_pipeline
    
# if __name__ == "__main__":
#     audio_recognition = AudioRecognition()
#     text = audio_recognition.forward("src/audio/sample.mp3")
#     print(text)



