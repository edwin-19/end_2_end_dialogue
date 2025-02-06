import typer
import transformers
import soundfile as sf
import torch
import numpy as np
from typing import Dict, Any
from pathlib import Path
import os
from tqdm import tqdm
import re
import logging
import soundfile as sf
import librosa
from demo import load_tts_model
import time

logging.basicConfig(level=logging.INFO)

app = typer.Typer()

def preprocess(inputs: Dict[str, Any], device, dtype, processor):
    turns: list = inputs.get("turns", [])

    audio = inputs.get("audio", None)
    # Convert to float32 if needed.
    if isinstance(audio, np.ndarray):
        if audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        elif audio.dtype == np.int16:
            audio = audio.astype(np.float32) / np.float32(32768.0)
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / np.float32(2147483648.0)

    if audio is not None and (len(turns) == 0 or turns[-1]["role"] != "user"):
        prompt = inputs.get("prompt", "<|audio|>")
        if "<|audio|>" not in prompt:
            print(
                "Prompt does not contain '<|audio|>', appending '<|audio|>' to the end of the prompt."
            )

            prompt += " <|audio|>"
        turns.append({"role": "user", "content": prompt})

    text = processor.tokenizer.apply_chat_template(
        turns, add_generation_prompt=True, tokenize=False
    )

    if "sampling_rate" not in inputs and audio is not None:
        print(
            "No sampling rate provided, using default of 16kHz. We highly recommend providing the correct sampling rate."
        )

    output = processor(
        text=text,
        audio=audio,
        sampling_rate=inputs.get("sampling_rate", 16000),
    )
    if "audio_values" in output:
        output["audio_values"] = output["audio_values"].to(device, dtype)
    return output.to(device, dtype)

@torch.inference_mode()
@app.command()
def main(
    # speech_llm_path:str=typer.Option("models/ultravox-gemma-2-2b-jpn-it"),
    # tts_path:str=typer.Option("models/Kokoro-82M"),
    # voice_path:str=typer.Option('./models/Kokoro-82M/voices/jf_alpha.pt'),
    # data_path:Path=typer.Option("question/"),
    # outdir:str=typer.Option("output"), max_tokens:int=typer.Option(300)
    
    speech_llm_path:str=typer.Option("neody/ultravox-gemma-2-2b-jpn-it"),
    tts_path:str=typer.Option(""),
    voice_path:str=typer.Option('jf_alpha'),
    data_path:Path=typer.Option("question/"),
    outdir:str=typer.Option("output"), max_tokens:int=typer.Option(300)
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    model = transformers.AutoModel.from_pretrained(
        speech_llm_path, trust_remote_code=True
    )
    model.to("cuda", dtype=torch.bfloat16)
    processor = transformers.AutoProcessor.from_pretrained(
        speech_llm_path, trust_remote_code=True
    )
    tts_pipeline = load_tts_model(tts_path)
    
    all_wavs = sorted(list(data_path.glob("*")))
    
    total_time = []
    for index, wav_path in enumerate(tqdm(all_wavs)):
        y, sr = librosa.load(wav_path, sr=16000)
        turns = []
        start_time = time.time()
        inputs = preprocess(
            {"audio": y, "turns": turns, "sampling_rate": sr}, "cuda", torch.bfloat16,
            processor
        )
        logits = model.generate(**inputs,  max_new_tokens=max_tokens,)
        pred_text = processor.tokenizer.decode(logits.squeeze(), skip_special_tokens=True,)
        
        pred_text = pred_text.replace('model', '')
        pred_text = pred_text.replace('user', '')
        pred_text = re.sub(r'\n+', '\n', pred_text).strip()
        
        logging.info('Data: {}'.format(pred_text))
        
        generator = tts_pipeline(
            pred_text, voice=voice_path, # <= change voice here
            speed=1, split_pattern=r'\n+'
        )
        
        audios = []
        for i, (gs, ps, audio) in enumerate(generator):
            audios.append(audio)
        
        merged_audio = np.concatenate(audios, axis=0)
        sf.write(os.path.join(outdir, '{}.wav'.format(index)), merged_audio, 24000)
        total_time.append(time.time() - start_time)
        
    logging.info("Total Time: {:.2f}, Average Time: {:.2f}".format(sum(total_time), np.mean(total_time)))

if __name__ == '__main__':
    app()