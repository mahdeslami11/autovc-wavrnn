from vocoder.model_vc import *
from vocoder import hparams as hp
import torch
import matplotlib.pyplot as plt
from math import ceil
_model = None   # type: model_vc

def pad_seq(x, base=1):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant',constant_values=(0.0,0.0)), len_pad


def load_model(weights_fpath, verbose=True):
    global _model
    
    if verbose:
        print("Building auto-VC")
    _model = model_VC(32,256,512,1).cuda()
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def gen_waveform(mel,e1,e2,normalize=True):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")

    if normalize:
        mel = mel / hp.mel_max_abs_value

    m,_ = pad_seq(mel)
    print("pad seq mel shape:", m.shape)
    m = torch.from_numpy(m[None, ...])


    e1 = e1[np.newaxis, :, np.newaxis]
    e1=torch.tensor(e1)
    e2 = e2[np.newaxis, :, np.newaxis]
    e2=torch.tensor(e2)

    m, e1, e2 = m.cuda(), e1.cuda(),e2.cuda()

    print("mel shape", m.shape)
    print("embed shape", e1.shape)
    m=m.permute(0,2,1)
    C, _, _, X, _ = _model(m,e1,e2)
    # fig = plt.figure(2, figsize=(12, 6))
    # plt.imshow(X_C.detach().cpu().numpy(), interpolation='nearest', aspect='auto')
    # fig = plt.figure(3, figsize=(12, 6))
    # plt.imshow(C.detach().cpu().numpy(), interpolation='nearest', aspect='auto')
    # plt.subplot(1)
    # plt.plot(X)
    # plt.title("generate mel spectrogram")
    # plt.show()
    return C,X
