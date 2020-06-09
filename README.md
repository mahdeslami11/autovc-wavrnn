# AutoVC-WavRNN
>voice conversion system

Data Preprocess
-----------------
>To get the audio: 

>>1.Load and rescale the wav with the max absolute value.

>>2.Normalize the volume of wavs.(target dBFS is -30)

>>3.Skip utterances that are too short.(less than 1.5s)

>To get the mel-spectrogram:

>>4.Preemphasis.(filter coefficient is 0.97)

>>5.STFT.(n_fft=1024, hop_size=256,win_size=1024)

>>6.Build mel filter bank.

>>7.Inner product the result of STFT and mel filter bank then get 80 channel mel-spectrogram.

>>8.Transform amplitude to dB.(ref_level_db=16)

>>9.Normalize mel-spectrogram to [0,1].

>try: ```python synthesizer_preprocess_audio.py /data ``` to preprocess audio

>try: ```python synthesizer_preprocess_embeds.py /data/SV2TTS_autovc-ttsdb/synthesizer_vad -e pretrained.pt``` to preprocess speaker embedding

AutoVC Training step
-----------------
>the AutoVC training step is include in the ```autovc_train.py```, the model of AutoVC include in the ```model_vc.py```.

>try: ```python autovc_train.py autovc-vcc2020 /data -g```

WavRNN Training step
-----------------
>try: ```python vocoder_train.py my_vocoder /data/ -g```

Relevent Repositories
-----------------
>CorentinJ/Real-Time-Voice-Cloning: https://github.com/CorentinJ/Real-Time-Voice-Cloning

>auspicious3000/autovc: https://github.com/auspicious3000/autovc

Relevent Paper
-----------------
AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss https://arxiv.org/abs/1905.05879

