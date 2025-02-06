import typer
from datasets import load_dataset
import soundfile as sf
from transformers import pipeline
import torch
from kokoro import KPipeline, KModel
import json
import os
import numpy as np
import logging
import random
from tqdm import tqdm

random.seed(42)
logging.basicConfig(level=logging.INFO)

app = typer.Typer(pretty_exceptions_enable=False)

def load_tts_model(tts_path):
    if tts_path != "":
        model_path = str(os.path.join(tts_path, 'kokoro-v1_0.pth'))
        config = os.path.join(tts_path, 'config.json')
        with open(config, 'r', encoding='utf-8') as r:
            config = json.load(r)
    else:
        model_path = None
        config = None
        
    tts_model = KModel(config, model_path)
    tts_model.eval()
    tts_pipeline = KPipeline(lang_code='j', model=tts_model, device='cuda:0') # <= make sure lang_code matches voice
        
    return tts_pipeline

@torch.inference_mode()
@app.command()
def main(
    # Local Env
    # data_path:str=typer.Option("dataset/qa-dev-speechq"),
    # speech_llm:str=typer.Option("models/ultravox-v0_4_1-llama-3_1-8b"),
    # translate_llm:str=typer.Option("/data5/edwin/sample_tools/visual_llm/models/gemma-2-2b-jpn-it"),
    # tts_path:str=typer.Option("models/Kokoro-82M"),
    # voice_path:str=typer.Option('./models/Kokoro-82M/voices/jf_alpha.pt'),
    # batch_size:int=typer.Option(3),
    # outdir:str=typer.Option("output")
    
    data_path:str=typer.Option("amuvarma/qa-dev-speechq"),
    speech_llm:str=typer.Option("fixie-ai/ultravox-v0_4_1-llama-3_1-8b"),
    translate_llm:str=typer.Option("google/gemma-2-2b-jpn-it"),
    tts_path:str=typer.Option(""),
    voice_path:str=typer.Option('jf_alpha'),
    batch_size:int=typer.Option(3),
    outdir:str=typer.Option("output")
):
    dataset = load_dataset(data_path)['train']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    tts_pipeline = load_tts_model(tts_path)
    trans_jp_pipe = pipeline(
        "text-generation",
        model=translate_llm,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",  # replace with "mps" to run on a Mac device
    )
    pipe = pipeline(model=speech_llm, trust_remote_code=True, device=0)
    
    sel_batch = np.random.randint(0, len(dataset) - 1, batch_size)
    for index in tqdm(sel_batch):
        data = dataset[int(index)]
        logging.info("Question: {}".format(data['question']))

        audio = data['audio']['array']
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
        ]
        output = pipe({'audio': audio, 'turns': turns, 'sampling_rate': 16000}, max_new_tokens=200)
        logging.info('Ultravox Output: {}'.format(output))
        
        translation_input_text = f"Translate the following sentence from English to Japanese:\n\n{output}, do not add any explnation just the translation"
        messages = [
            {"role": "user", "content": translation_input_text},
        ]
        outputs = trans_jp_pipe(messages, return_full_text=False, max_new_tokens=1024)
        translated_response = outputs[0]["generated_text"].strip()
        logging.info('Jp Translated: {}'.format(translated_response))
        
        generator = tts_pipeline(
            translated_response, voice=voice_path, # <= change voice here
            speed=1, split_pattern=r'\n+'
        )

        audios = []
        for i, (gs, ps, audio) in enumerate(generator):
            audios.append(audio)
        
        merged_audio = np.concatenate(audios, axis=0)
        sf.write(os.path.join(outdir, '{}.wav'.format(index)), merged_audio, 24000)

if __name__ == "__main__":
    app()