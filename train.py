from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.display import stream, simple_table
from vocoder.gen_wavernn import gen_testset
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
import torch.nn.functional as F
import vocoder.hparams as hp
import numpy as np
import time
from vocoder.model_vc import *
import torch.nn as nn
from vocoder.griffin_lin import *
import scipy 
import matplotlib.pyplot as plt
import torch

def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = torch.tensor(global_step + 1, dtype=torch.float32)
    return init_lr * warmup_steps**0.5 * torch.min(step * warmup_steps**-1.5, step**-0.5)


def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool,
          save_every: int, backup_every: int, force_restart: bool):
    # Check to make sure the hop length is correctly factorised
   # assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length
    
    # Instantiate the model
    print("Initializing the model...")
   # model = WaveRNN(
   #     rnn_dims=hp.voc_rnn_dims,
   #     fc_dims=hp.voc_fc_dims,
   #     bits=hp.bits,
   #     pad=hp.voc_pad,
   #     upsample_factors=hp.voc_upsample_factors,
   #     feat_dims=hp.num_mels,
   #     compute_dims=hp.voc_compute_dims,
   #     res_out_dims=hp.voc_res_out_dims,
   #     res_blocks=hp.voc_res_blocks,
   #     hop_length=hp.hop_length,
   #     sample_rate=hp.sample_rate,
   #     mode=hp.voc_mode
   # ).cuda()
    model= model_VC(32,256,512,32).cuda() 
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups: 
        p["lr"] = hp.voc_lr
   
    loss_recon = nn.MSELoss()
    loss_content=nn.L1Loss()
    # Load the weights
    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir.joinpath(run_id + ".pt")
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of AutoVC from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("AutoVC weights loaded from step %d" % model.step)
    
    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
        voc_dir.joinpath("synthesized.txt")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    #2019.11.26
    embed_dir=syn_dir.joinpath("embeds")

    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir,embed_dir)
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])
   
    for epoch in range(1, 350):
  
        model.train()
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_vocoder,
                                 batch_size=hp.voc_batch_size,
                                 num_workers=2,
                                 shuffle=True,
                                 pin_memory=True)
        start = time.time()
        running_loss = 0.

        for i, (m, e,_) in enumerate(data_loader, 1):
            #print("e:",e.shape)
            #print("m:",m.shape)
            model.train()
            m, e= m.cuda(), e.cuda()
            # Forward pass   
            C,X_C,X_before,X_after,_ = model(m, e,e)
        
            #c_org shape: torch.Size([100, 256, 1])
            #x shape: torch.Size([100, 80, 544])
            #c_org_expand shape torch.Size([100, 256, 544])
            #encoder_outputs shape: torch.Size([100, 544, 320])
            #C shape: torch.Size([100, 544, 64])
            #X shape: torch.Size([100, 1, 544, 80])
            X_after = X_after.squeeze(1).permute(0,2,1)
            X_before = X_before.squeeze(1).permute(0,2,1)

            #print("C shape:",C.shape)
            #if X_C:
            #    print("X_C shape:",X_C.shape)
            #print("X shape:",X.shape)    
            # Backward pass
            loss_rec_before = loss_recon(X_before,m)
            loss_rec_after = loss_recon(X_after, m)  
            loss_c=loss_content(C,X_C)     
            loss = loss_rec_before + loss_rec_after + loss_c
            #print("recon loss:",loss1)
            #print("content loss:",loss2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("loss:",loss.item())
            running_loss += loss.item()
            #print("running loss:",running_loss)
            speed = i / (time.time() - start)
            avg_loss = running_loss / i
            #print("avg_loss:",avg_loss)
            step = model.get_step()
            
            if hp.decay_learning_rate==True:
                p["lr"]=_learning_rate_decay(p["lr"], step)
            k = step // 1000
            if step%100==0 and step !=0:
                model.eval()
                plt.figure(1)
                C,X_C,X_before,X_after,_ = model(m, e,e)
                X_after = X_after.squeeze(1).permute(0,2,1)
                mel_out=torch.tensor(X_after).clone().detach().cpu().numpy()

                from synthesizer import audio
                from synthesizer.hparams import hparams
                wav = audio.inv_mel_spectrogram(mel_out[0,:,:], hparams)
                librosa.output.write_wav("out.wav", np.float32(wav),hparams.sample_rate)

                mel_out=mel_out[0,:,:].transpose(1,0)
                plt.imshow(mel_out.T, interpolation='nearest', aspect='auto')
                plt.title("Generate Spectrogram")
                save_path=model_dir
                p_path=save_path.joinpath("generate.png")
                plt.savefig(p_path)

                plt.figure(2)
                m_out=m.squeeze(1).permute(0,2,1)
                m_out=torch.tensor(m).clone().detach().cpu().numpy()
                m_out=m_out[0,:,:].transpose(1,0)
                plt.imshow(m_out.T, interpolation='nearest', aspect='auto')
                plt.title("Orignal Spectrogram")
                o_path=save_path.joinpath("orignal.png")
                plt.savefig(o_path)

          
            if backup_every != 0 and step % backup_every == 0 :
                model.checkpoint(model_dir, optimizer)
                
            if save_every != 0 and step % save_every == 0 :
                model.save(weights_fpath, optimizer)
                torch.save(model,"model_ttsdb_48_48.pkl")

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                f"steps/s | Step: {k}k | "
            stream(msg)



       # gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,hp.voc_target,model_dir)
        print("")
