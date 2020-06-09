import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org): #mel embd
        c_org=c_org.expand(-1,-1,x.size(-1)) # torch.Size([100, 256, 800])
                                             # x:  torch.Size([100, 80, 800])  
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        #print("out_forward shape:",out_forward.shape)  # torch.Size([100, 576, 32])
        #print("out_backward shape:",out_backward.shape)
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        #self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        self.conv1 = nn.Sequential(
                ConvNorm(dim_neck*2+dim_emb,      
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
                                     
        convolutions = []
        for i in range(2):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,       #dim_pre
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(dim_pre, 1024, 3, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        #self.lstm1.flatten_parameters()
        #x, _ = self.lstm1(x)
        #x = x.transpose(1, 2)
        #print("x shape:",x.shape) # x shape: torch.Size([8, 800, 320])
        x = x.transpose(1,2)

        x = self.conv1(x)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
 
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
  
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class model_VC(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):  #32 256 512 32
        super(model_VC, self).__init__()

        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.step=nn.Parameter(torch.zeros(1).long(),requires_grad=False)
    def forward(self, x, c_org,c_trg):
        self.step+=1
        codes = self.encoder(x, c_org)
        #print("codes_len:",len(codes))
        #print("codes0_shape:",codes[0].shape) # torch.Size([100, 64])   
        #print("codes1_shape:",codes[1].shape)   
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(2) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        #print("code_exp shape:",code_exp.shape) #[100,640,64]
        #print("c_org shape:",c_org.shape)  # torch.Size([100, 256, 1])
        #print("x shape:",x.shape)          # torch.Size([100, 80, 640])          
        c_trg_expand=c_trg.expand(-1,-1, x.size(-1))
      
        #print("c_org_expand shape",c_org_expand.shape)          
        encoder_outputs = torch.cat((code_exp, c_trg_expand.permute(0,2,1)), dim=-1)
        #print("encoder_outputs shape:",encoder_outputs.shape)  # torch.Size([100, 640, 320])
    
        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        #print("mel_outputs_postnet shape:",mel_outputs_postnet.shape) #  torch.Size([100, 1, 736, 80])
        X=mel_outputs_postnet.detach()
        X=X.squeeze(1).permute(0,2,1)
        codes_X = self.encoder(X, c_org)
        tmp_X=[] 
        for code_X in codes_X:
            tmp_X.append(code_X.unsqueeze(1).expand(-1, int(X.size(2) / len(codes_X)), -1))
        code_X_exp = torch.cat(tmp_X, dim=1)
        return code_exp,code_X_exp,mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
       

    def get_step(self): 
        return self.step.data.item()  
    
    def checkpoint(self,model_dir,optimizer):
        k_steps=self.get_step()//1000
        self.save(model_dir.joinpath("checkpoint_%dk_steps.pt"%k_steps),optimizer)
    def save(self,path,optimizer):
        torch.save({"model_state":self.state_dict(),"optimizer_state":optimizer.state_dict()},path)
    def load(self,path,optimizer):
        checkpoint=torch.load(path)
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            self.load_state_dict(checkpoint)
