import torch, math, os, sys
from torch.nn import functional as F
from torch import nn
import torch.distributions as td
from torch.autograd import Variable
from transformers import *
from wsd_models.util import *











class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss





    def forward(self, input_ids, attn_mask, output_mask=None):

        #encode gloss text
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]


        if output_mask is None:
            #training model to put all sense information on CLS token 
            gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        else:
            #average representations over target word(s)
            example_arr = []        
            for i in range(gloss_output.size(0)): 
                example_arr.append(process_encoder_outputs2(gloss_output[i], output_mask[i], as_tensor=True))
            gloss_output = torch.cat(example_arr, dim=0)


        return gloss_output





class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context





    def forward(self, input_ids, attn_mask, output_mask, flag):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        example_arr = []        
        for i in range(context_output.size(0)): 
            if flag: example_arr.append(process_encoder_outputs2(context_output[i], output_mask[i], as_tensor=True))
            else: example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)


        return context_output








class VAE(torch.nn.Module):
    
    """docstring for VAE"""
    def __init__(self, input_dim=768, hidden_dim=768, z_dim=768):
        super(VAE, self).__init__()

        ### Latent
        self.mu1 = nn.Linear(input_dim, z_dim)
        self.mu2 = nn.Linear(input_dim, z_dim)

        self.lv1_1 = nn.Linear(input_dim, z_dim)
        self.lv1_2 = nn.Linear(input_dim, z_dim)

        self.lv2 = nn.Linear(input_dim, z_dim)


        ### Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )




    def encode(self, x):
        lv1_1 = F.normalize(self.lv1_1(x), p=2, dim=1)
        lv1_2 = F.normalize(self.lv1_2(x), p=2, dim=1)

        lv1 = (1/2) * (torch.bmm(lv1_1.transpose(0,1).unsqueeze(0), lv1_1.unsqueeze(0)) + torch.bmm(lv1_2.transpose(0,1).unsqueeze(0), lv1_2.unsqueeze(0)))
        lv2 = torch.diag_embed(self.lv2(x))

        return self.mu1(x), lv1, self.mu2(x), lv2



    def decode(self, z):
        return self.decoder(z)



    def reparameterization(self, mu1, lv1, mu2, lv2):
        mean   = (1/2) * (mu1 + mu2)

        #logvar = (1/2) * (torch.diag(lv1) + torch.diag(lv2))
        logvar = (1/2) * (torch.diagonal(lv1, dim1=1, dim2=2) + torch.diagonal(lv2, dim1=1, dim2=2))


        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std



    def forward(self, x):

        # Encoder
        mu1, lv1, mu2, lv2 = self.encode(x)
        sampled_z = self.reparameterization(mu1, lv1, mu2, lv2)

        # Decoder
        x_hat = self.decode(sampled_z)

        return x_hat, sampled_z, mu1, lv1, mu2, lv2




    def compute_loss(self, mu1, lv1, mu2, lv2, weight):
        mu       = (1/2) * (mu1 + mu2)
        logsigma = (1/2) * (torch.diagonal(lv1, dim1=1, dim2=2) + torch.diagonal(lv2, dim1=1, dim2=2))
        
        #kl = (-0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) + 1e-5) / x.shape[0]
        kl = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim=1), dim=0)
        
        return weight * kl










class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders


        self.context_vae = VAE()
        self.gloss_vae   = VAE()


        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder  = ContextEncoder(encoder_name, freeze_context)

        
        if self.tie_encoders:
            self.gloss_encoder  = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)

        else:
            self.gloss_encoder  = GlossEncoder(encoder_name, freeze_gloss)

        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim




    def context_forward(self, context_input, context_input_mask, context_example_mask, flag=False):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask, flag)

    def gloss_forward(self, gloss_input, gloss_mask, output_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask, output_mask)

    def context_vae_forward(self, x):
        return self.context_vae.forward(x)

    def context_vae_loss(self, mu1, lv1, mu2, lv2, weight=1):
        return self.context_vae.compute_loss(mu1, lv1, mu2, lv2, weight)

    def gloss_vae_forward(self, x):
        return self.gloss_vae.forward(x)

    def gloss_vae_loss(self, mu1, lv1, mu2, lv2, weight=1):
        return self.gloss_vae.compute_loss(mu1, lv1, mu2, lv2, weight)
        

#EOF
