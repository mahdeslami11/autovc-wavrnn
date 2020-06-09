from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
import vocoder.hparams as hp
import numpy as np
import torch
from math import ceil

class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path,embed_dir: Path):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, wav_dir))
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]

        #2019.11.26
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]

        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths, embed_fpaths))
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        mel_path, wav_path, embed_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32)
        
        # Load the wav
        wav = np.load(wav_path)
        #if hp.apply_preemphasis:
        #    wav = audio.pre_emphasis(wav)
        #wav = np.clip(wav, -1, 1)

        #2019.11.26
        embed=np.load(embed_path).astype(np.float32)
        # normalization(-4,4)
        #k = 8/(embed.max()-embed.min())
        #embed = -4.0+k*(embed-embed.min())      
     
        return mel.astype(np.float32),embed,wav

    def __len__(self):
        return len(self.samples_fpaths)
        
def pad_seq(x, base=48):
    len_out = int(base * ceil(x.shape[0]/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant',constant_values=(0.0,0.0))

def pad(mel,shape):
    len_pad = shape - mel.shape[0]
    assert len_pad >= 0
    return np.pad(mel, ((0,len_pad),(0,0)), 'constant',constant_values=(0.0,0.0))
        
def collate_vocoder(batch):
   # mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    
   # max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
   # max_offsets_em = [x[1].shape[0] - 2 - (mel_win + 2 * hp.voc_pad) for x in batch]
   # print("max_offsets_em:",max_offsets_em)
   # mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
   # embd_offsets=[np.random.randint(0, offset) for offset in max_offsets_em]    

   # sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]
    mels=[x[0].transpose() for x in batch]
    #print("mel_shape:",mels[0].shape) 
    #max_pad=max([x.shape[0] for x in mels]) 
    #print("max_pad",max_pad) 
    mels=[pad_seq(x) for x in mels]
    min_shape = min([mel.shape[0] for mel in mels])
    #mels=[ pad(mel,max_shape)  for mel in mels]
    mels = [ x[:min_shape,:]  for x in mels]
    #print("mel shape:",mels[0].shape)
    #print("mel_pad_shape",pad_seq(mels[0],max_pad).shape) 
   # mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
   #print("x[1]",batch[0][1])
    embds= [x[1][np.newaxis, :] for  x in batch]
   # labels = [x[2][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]
   
    mels = np.stack(mels).astype(np.float32)
  
    embds=np.stack(embds).astype(np.float32)
   
   # labels = np.stack(labels).astype(np.int64)
   # print("labels_shape:",labels.shape)
    mels = torch.tensor(mels)  # [batch_size,sequence_size,80]
    embds=torch.tensor(embds)  # [batch_size,1,256]
    mels=mels.permute(0,2,1)   # [batch_size,80,sequence_size]
    embds=embds.permute(0,2,1) # [batch_size,256,sequence_size]
    import random
    
    wav=batch[random.sample(range(0,len(batch)),1)[0]][2]
    
   # print("mels:",mels.shape)
   # print("embds:",embds.shape)
   # labels = torch.tensor(labels).long()

   # x = labels[:, :hp.voc_seq_len]
   # y = labels[:, 1:]

   # bits = 16 if hxis, :]p.voc_mode == 'MOL' else hp.bits

   # x = audio.label_2_float(x.float(), bits)

   # if hp.voc_mode == 'MOL' :
   #     y = audio.label_2_float(y.float(), bits)
    
    return  mels,embds,wav
