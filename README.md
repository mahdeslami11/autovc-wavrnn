# AutoVC-WavRNN
>voice conversion system

>This repository provides a PyTorch implementation of AutoVC-WavRNN 

Audio Demo
-----------------
>The audio demo for AUTOVC-WavRNN can be found in results.

![image](http://github.com/freenowill/AutoVC-WavRNN/raw/master/picture/orignal.png "Orignal spectrogram")

![image](http://github.com/freenowill/AutoVC-WavRNN/raw/master/picture/generate.png "Generate spectrogram")

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

Pretrained model
-----------------
>speaker encoder pretrained model: https://drive.google.com/file/d/1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc/view.

>the autoVC-WavRNN model which trained by mixed VCTK dataset and VCC2020 dataset is in the folder of ```/model```, please choose all 
rar files and decompression.

AutoVC Training step
-----------------
>the AutoVC training step is included in the ```autovc_train.py```, the model of AutoVC included in the ```model_vc.py```.

>try: ```python autovc_train.py autovc-vcc2020 /data -g```

WavRNN Training step
-----------------
>try: ```python vocoder_train.py my_vocoder /data/ -g```

Inference
-----------------
>try:```python convert.py```

Relevent Repositories
-----------------
>CorentinJ/Real-Time-Voice-Cloning: https://github.com/CorentinJ/Real-Time-Voice-Cloning

>auspicious3000/autovc: https://github.com/auspicious3000/autovc

Relevent Paper
-----------------
non-parallel many-to-many voice conversion, as well as zero-shot voice conversion, remain under- explored areas. Deep style transfer algorithms, such as generative adversarial networks (GAN) and conditional variational autoencoder (CVAE), are being applied as new solutions in this field. However, GAN training is sophisticated and diffi- cult, and there is no strong evidence that its gen- erated speech is of good perceptual quality.. On the other hand, CVAE training is simple but does not come with the distribution-matching property of a GAN. In this paper, we propose a new style transfer scheme that involves only an autoencoder with a carefully designed bottleneck. We formally show that this scheme can achieve distribution- matching style transfer by training only on a self- reconstruction loss. Based on this scheme, we proposed AUTOVC, which achieves state-of-the- art results in many-to many voice conversion with non-parallel data, and which is the first to perform zero-shot voice conversion.
First, most voice conversion systems assume the availability of parallel training data speech pairs where the two speakers utter the same sentences. Only a few can be trained on non-parallel data. Second, among the few existing algorithms that work on non-parallel data, even fewer can work for many-to-many conversion, i.e. converting from multiple source speakers to multiple target speakers. Last but not least, no voice conver- sion systems are able to perform zero-shot conversion, i.e. conversion to the voice of an unseen speaker by looking at only a few of his/her utterances
no change was made.
https://1drv.ms/v/s!AjJEvYoU3ZQOeaX5sBlNrmBUwco
https://1drv.ms/b/s!AjJEvYoU3ZQOei_ja4cqd5N-KnQ 
https://github.com/freenowill/AutoVC-WavRNN
Hadis Mahmoudi Bardzard, a master's student in medical engineering, majoring in bioelectricity. Project number 31 Titled zero-shot voice style conversion with only auto-encoder loss
https://1drv.ms/b/s!AjJEvYoU3ZQOe-zUlwbufIhIM9k
https://1drv.ms/v/s!AjJEvYoU3ZQOdeVebl3jjuVU9O4 https://1drv.ms/v/s!AjJEvYoU3ZQOdgIJcKN6c4z7lBE https://1drv.ms/v/s!AjJEvYoU3ZQOeaX5sBlNrmBUwco
