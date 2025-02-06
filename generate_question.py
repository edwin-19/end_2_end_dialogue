import typer
from transformers import pipeline
from datasets import load_dataset
from demo import load_tts_model
import numpy as np
import os
import torch
from tqdm import tqdm
import logging
import soundfile as sf

logging.basicConfig(level=logging.INFO)

app = typer.Typer()

@app.command()
def main(
    # Local Env
    # data_path:str=typer.Option("dataset/qa-dev-speechq"),
    # translate_llm:str=typer.Option("/data5/edwin/sample_tools/visual_llm/models/gemma-2-2b-jpn-it"),
    # tts_path:str=typer.Option("models/Kokoro-82M"),
    # voice_path:str=typer.Option('./models/Kokoro-82M/voices/jm_kumo.pt'),
    
    data_path:str=typer.Option("amuvarma/qa-dev-speechq"),
    translate_llm:str=typer.Option("google/gemma-2-2b-jpn-it"),
    tts_path:str=typer.Option(""),
    voice_path:str=typer.Option('jm_kumo'),
    outdir:str=typer.Option("question"), batch_size:int=typer.Option(6)
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
    sel_batch = np.random.randint(0, len(dataset) - 1, batch_size)
    for index, sel_index in enumerate(tqdm(sel_batch)):
        data = dataset[int(sel_index)]
        logging.info("Question: {}".format(data['question']))
        
        translation_input_text = f"Translate the following sentence from English to Japanese:\n\n{data['question']}, do not add any explnation just the translation"
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
        for i, (_, _, audio) in enumerate(generator):
            audios.append(audio)
        
        merged_audio = np.concatenate(audios, axis=0)
        sf.write(os.path.join(outdir, '{}.wav'.format(index)), merged_audio, 24000)

if __name__ == "__main__":
    app()