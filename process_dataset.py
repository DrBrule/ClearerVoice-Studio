from clearvoice import ClearVoice
import librosa, soundfile
from scipy.io.wavfile import write as wavwrite
import numpy as np
import pandas as pd
import time, os
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset



# helper functions

def int_to_str(number):
    out = str(number)
    while len(out) < 4:
        out = '0'+out
    return out

base_file = 'samples/paula_comb.wav'

# will pass array of files next

enhance      = True
num_speakers = 1
speaker_name = 'paula'
speech_enhancement_model = 'MossFormer2_SE_48K' # 'MossFormerGAN_SE_16K'
language    = 'en'



cutting_model = 'pyannote' # 'silero'
output_dir   = 'samples/processed_'+ speaker_name + '/'



os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

info        = soundfile.info(base_file)
wav, sr     = librosa.load(base_file, sr=info.samplerate)

print('detected SR : ' + str(sr))

# now SEPARATE audio chunks 
ramp_len    = 1000
#diarization is VERY tight...
in_ramp     = np.linspace(0,1,ramp_len)
out_ramp    = np.linspace(1,0,ramp_len)

def ramps(wav, in_ramp, out_ramp):
    wav[0:ramp_len] = wav[0:ramp_len] * in_ramp
    wav[-ramp_len:] = wav[-ramp_len:] * out_ramp
    return wav


if sr == 24000:
    wav_24k = wav.astype(np.float32)

wav_16k     = librosa.resample(wav, orig_sr=sr, target_sr=16000)
wavwrite('samples/input_16k.wav', 16000, wav_16k)


if enhance == True:
    # most all of these require 16K... so we lose everything above 8K
    # do it at 48K then downsample AT THE END!
    speech_enhancement = ClearVoice(
                task='speech_enhancement', 
                model_names=['MossFormer2_SE_48K']
                )
    
    wav_enhanced    = speech_enhancement(input_path=base_file, online_write=False)
    wav_enhanced    = wav_enhanced.astype(np.float32)
    wavwrite('samples/enhanced.wav', 48000, wav_enhanced)    
    wav_24k     = librosa.resample(wav_enhanced, 
                        orig_sr=48000, 
                        target_sr=24000).astype(np.float32)
    wav_16k     = librosa.resample(wav_enhanced, 
                        orig_sr=48000, 
                        target_sr=16000).astype(np.float32)
    
    





# OR if multiples, get speaker ID...assuming there are 1-6 speakers in here..
from pyannote.audio import Pipeline
diarization_pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

diarization_pipeline.to(torch.device("cuda"))
# run the pipeline on an audio file
if enhance == True:
    waveform, sample_rate = torchaudio.load('samples/enhanced.wav')
else:
    waveform, sample_rate = torchaudio.load('samples/input_16k.wav')

diarization = diarization_pipeline(
    {"waveform": waveform, 
    "sample_rate": sample_rate},
    min_speakers=1, 
    max_speakers=num_speakers)

diarization_data = []
# Extract speaker segments and append to list
for turn, _, speaker in diarization.itertracks(yield_label=True):
    diarization_data.append({
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end
    })

### load whisper 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

whisper_model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)



# we are always exporting at 24000 for training data.

## now main cutting and transcription loop

metadata        = []
i = 0
wav_chunk   = np.array([])
last_speaker = diarization_data[0]['speaker']

# let's let the speakers talk for up to 15s

for chunk in diarization_data:
    start_sample    = int( ( chunk['start'] * sr ) - 1000 )# pad a bit (100 samples)
    if start_sample < 1:
        start_sample = 1
    end_sample      = int( ( chunk['end']   * sr ) + 1000 )
    if end_sample > len(wav_24k):
        end_sample = len(wav_24k)
    
    speaker         = chunk['speaker']
    
    if (end_sample - start_sample) < (ramp_len * 4):
        # chunk too small ( 4000 samples???)
        continue
    else:
        new_chunk       = ramps( wav_24k[start_sample:end_sample] , in_ramp, out_ramp)
        if speaker == last_speaker:
            wav_chunk       = np.append(wav_chunk , new_chunk)
            # check length so far, if long enough, export and continue
            # randomize between 3 - 13 s
            if (len(wav_chunk) / sr) > ( 3 + np.random.randint(7)):
                i += 1
                wav_chunk       = wav_chunk.astype(np.float32)
                transcription   = whisper_pipe(wav_chunk)
                fn              = output_dir + 'chunk_' + int_to_str(i) + '.wav' 
                wavwrite(fn, 24000, wav_chunk)
                wav_chunk       = np.array([])
                transcription['filename'] = fn
                transcription['language'] = language
                transcription['speaker'] = speaker
                metadata.append(transcription)
        else:
            #speaker finished, write out chunk and begin new
            i += 1
            wav_chunk       = wav_chunk.astype(np.float32)
            transcription   = whisper_pipe(wav_chunk)
            fn              = output_dir + 'chunk_' + int_to_str(i) + '.wav' 
            wavwrite(fn, 24000, wav_chunk)
            wav_chunk       = new_chunk
            last_speaker    = speaker 
            transcription['filename'] = fn
            transcription['language'] = language
            transcription['speaker'] = speaker
            metadata.append(transcription)     



df = pd.DataFrame(metadata)
del df['chunks']


# dataframe and cleanup punctuation
from fastpunct import FastPunct
fastpunct = FastPunct()

def fixpunct(row):
    return fastpunct.punct(row.text)

df['text'] = df.apply(fixpunct, axis=1)

df = df[['filename','text','speaker','language']]
df.to_csv(output_dir +'metadata.csv', sep='|', index=False, header=False)


end_time = time.time()

total_len = (len(wav_24k) / sr ) / 60
print('TOTAL TIME TO PROCESS ' + str(total_len) + ' minute file::')
print( ( str((end_time - start_time) / 60)) )



