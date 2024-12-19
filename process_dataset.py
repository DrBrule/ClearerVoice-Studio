from clearvoice import ClearVoice
import librosa
import soundfile
from scipy.io.wavfile import write as wavwrite

speech_enhancement = ClearVoice(
            task='speech_enhancement', 
            model_names=['MossFormer2_SE_48K']
            )

speech_separation = ClearVoice(
            task='speech_separation', 
            model_names=['MossFormer2_SS_16K']
            )

# myClearVoice = ClearVoice(task='target_speaker_extraction', model_names=['AV_MossFormer2_TSE_16K'])
# target isn't as important


# input mp3 file 
# convert to wav
# upsample if needed ( -> 48K with librosa )
# speech enhancement
# do separation on downsampled copy
# then use sample markers (index x 3) to CUT 48K version

# process single wave file - will be 48K
base_file   = 'samples/75-1069.mp3'

info        = soundfile.info(base_file)
wav, sr     = librosa.load(base_file, sr=info.samplerate)

# for SR < 16K, doesn't matter which model we use
# BUT i dont want to downsample and upsample...so...?
wav = wav[0:30*sr] # cut to 30S, otherwise taking TOO LONG
# on m1 - .75x real time ( 30s audio = 45s proc time )

wav_48k     = librosa.resample(wav, orig_sr=sr, target_sr=48000)
wavwrite('samples/input_48k.wav', 48000, wav_48k)

enhanced_wav    = speech_enhancement(input_path='samples/input_48k.wav', online_write=False)
wavwrite('samples/enhanced.wav', 48000, enhanced_wav)



# works IF there are multiple speakers...
# generates as many wav files as speakers, 
# 16K is pretty crap

wav_16k     = librosa.resample(wav, orig_sr=sr, target_sr=16000)
wavwrite('samples/input_16k.wav', 16000, wav_16k)


separated_wav   = speech_separation(input_path='samples/input_16k.wav', online_write=False)
speech_separation.write(separated_wav, output_path='samples/output_MossFormer2_SS_16K.wav')

myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

