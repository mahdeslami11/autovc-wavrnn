from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *
from vocoder.griffin_lin import *
import scipy 

def gen_testset(model, test_set, samples, batched,target, save_path):
    k = model.get_step() // 1000

    for i, (m, e, w) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        batch_str = "gen_batched_target%d" % (target) if batched else \
            "gen_not_batched"

        target_path=save_path.joinpath("target_%dk_steps_%d_%s.wav"%(k,i,batch_str))
        
        w=w.numpy()[0]
        scipy.io.wavfile.write(target_path,22050,w) 
        gen_path=save_path.joinpath("gen_batch_%dk_steps_%d_%s.wav" % (k, i, batch_str))
       
        _,_,_,mel,_ = model(m, e, None)
       
        y=melspectrogram2wav(mel)
        #save_wav(y, save_str)
        scipy.io.wavfile.write(gen_path,22050,y)

