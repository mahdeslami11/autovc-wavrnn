from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from pathlib import Path
import torch
from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from datetime import datetime
from time import perf_counter as timer
import matplotlib.pyplot as plt
import numpy as np
# import webbrowser
import visdom
import umap
import seaborn as sns
import pandas as pd
colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255

marker={"F":"o","M":"x"}

class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=True):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        print("Updating the visualizations every %d steps." % update_every)

        # If visdom is disabled TODO: use a better paradigm for that
        self.disabled = disabled
        if self.disabled:
            return

            # Set the environment name
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = "%s (%s)" % (env_name, now)

        # Connect to visdom and open the corresponding window in the browser
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)

    def draw_projections(self, embeds, utterances_per_speaker, step, name, out_fpath=None,
                         max_speakers=10):
        max_speakers = min(max_speakers, len(colormap))
        
        embeds = embeds[:max_speakers * utterances_per_speaker]
   
        sex=[s.split("/")[-1].split("_")[-1]for s in name]
        sex=sex[:max_speakers]
        res_sex=[]
        for s in sex:
            for i in range(10):
                res_sex.append(s)
       
        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
       
      
        colors = [colormap[i] for i in ground_truth]
        markers=[marker[s] for s in res_sex]
        
        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
       
        sns.scatterplot(projected[:,0],projected[:,1],hue=ground_truth,style=markers)
        #plt.scatter(projected[:, 0], projected[:, 1], c=colors,marker=markers)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection (step %d)" % step)
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()

    

datasets_root=Path("/data/real-time/test/SV2TTS/encoder/")
encoder_model_fpath=Path("/data/real-time/test/encoder_2_lstm_80_mel_channel.pt")
encoder_out_dir=Path("/data/test/")

dataset = SpeakerVerificationDataset(datasets_root)
loader = SpeakerVerificationDataLoader(
    dataset,
    speakers_per_batch,
    utterances_per_speaker,
    num_workers=2,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FIXME: currently, the gradient is None if loss_device is cuda
loss_device = torch.device("cpu")

# Load the model
model = SpeakerEncoder(device, loss_device)
state_fpath = encoder_model_fpath
checkpoint = torch.load(state_fpath)
init_step = checkpoint["step"]
model.load_state_dict(checkpoint["model_state"])
vis = Visualizations("encoder_model")

count=0
for step, speaker_batch in enumerate(loader, init_step):
    print(speaker_batch)
    projection_fpath = encoder_out_dir.joinpath("%s_umap_%06d.png" % ("encoder_model", step))
    inputs = torch.from_numpy(speaker_batch.data).to(device)
    name=speaker_batch.names
    embeds = model(inputs)
    embeds = embeds.detach().cpu().numpy()
    vis.draw_projections(embeds, utterances_per_speaker, step=step, out_fpath=projection_fpath,name=name)
    print("Draw!")
    if count>5:
        break
    count+=1
