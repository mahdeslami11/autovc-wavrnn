from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from vocoder import inference_wavrnn as vocoder_wavrnn
from encoder import inference as encoder
from synthesizer import audio
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
from synthesizer.hparams import hparams
from vocoder.vocoder_dataset import pad_seq
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(1)  # 解决pytorch占用过多CPU

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="encoder/saved_models/english_pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="synthesizer/saved_models/logs-my_model/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-vx", "--voc_wavrnn_model_fpath", type=Path,
                        default="vocoder/saved_models/autovc_vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)


    vcc_path = "/data/VCTK/vcc2020_evaluation/"
    en_path = "/data/VCTK/vcc2020_training/"
    source = ["SEF1","SEF2","SEM1","SEM2"]
    target=["TEF1","TEF2","TEM1","TEM2","TFF1","TFM1","TGF1","TGM1",
                "TMF1","TMM1"]
    #target = ["TEF1","TEF2","TEM1","TEM2"]
    use_wavrnn = True

    if use_wavrnn==True:
        vocoder_wavrnn.load_model(args.voc_wavrnn_model_fpath)
    encoder.load_model(args.enc_model_fpath)
    model = torch.load("model_vcc_32_32_v4.pkl")
    for s in source:
        source_wav_name = os.listdir(vcc_path + s)
        for name in source_wav_name:
            source_wav_fpath = vcc_path+s+"/"+name
            source_wav = encoder.preprocess_wav(source_wav_fpath)
            e1 = encoder.embed_utterance(source_wav)
            e1 = e1[np.newaxis, :, np.newaxis]
            e1=torch.tensor(e1)
            for t in target:
                target_wav_name = os.listdir(en_path + t + "/wavs")
                embedding_tr = 0
                for i in range(10):
                    target_name = target_wav_name[i]
                    target_wav_fpath = en_path +t+"/wavs"+"/"+ target_name
                    target_wav = encoder.preprocess_wav(target_wav_fpath)
                    e2 = encoder.embed_utterance(target_wav)
                    embedding_tr = embedding_tr+ e2
                embedding_tr /=10

                print(embedding_tr.shape) 
                mel = audio.melspectrogram(source_wav, hparams)
                mel = pad_seq(mel.T).T
                mel = torch.from_numpy(mel[None, ...])

                embedding_tr =  embedding_tr[np.newaxis, :, np.newaxis]
                embedding_tr =torch.tensor(embedding_tr)
                mel,e1,embedding_tr = mel.cuda(),e1.cuda(),embedding_tr.cuda()
                #print("mel shape:",mel.shape)
                #print("e1 shape:",e1.shape)
                #print("e2 shape:",e2.shape)

                C,X_C,X_before,X_after,_ = model(mel, e1, embedding_tr)
                mel_out = torch.tensor(X_after).clone().detach().cpu().numpy()
                #print("mel_out shape:",mel_out.shape)
                if use_wavrnn:
                    wav = vocoder_wavrnn.infer_waveform(mel_out[0,0,:,:].T)
                else:
                    wav = audio.inv_mel_spectrogram(mel_out[0,0,:,:].T, hparams)
                wav = librosa.resample(wav,16000,24000)
                out_dir="/data/VCTK/out_v5/vcc2020-teams:00004/"
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                fname = t +"_"+ s +"_"+name[:-4]+".wav"
                out_dir_fpath = out_dir+"/"+fname
                librosa.output.write_wav(out_dir_fpath, wav.astype(np.float32),24000)
                print("write:{}".format(fname))














