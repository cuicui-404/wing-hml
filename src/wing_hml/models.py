"""Equations (1)-(5) of Iwata & Kumagai (2020) and their DS/NP controls."""
from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call


def mlp(inputs: int, outputs: int, hidden: int, dropout: float):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, outputs))


def masked_mean(x, mask, dim):
    # mask shape is broadcastable to x; dividing by the same broadcasted count
    # makes padding irrelevant even when different tasks have different schemas.
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum(dim=dim) / mask.expand_as(x).sum(dim=dim).clamp_min(1)


class SetModel(nn.Module):
    def inputs(self, sx, sy, qx, support_mask, feature_mask, response_mask):
        unbatched = sx.ndim == 2
        if unbatched:
            sx,sy,qx = sx[None],sy[None],qx[None]
        b,n,i = sx.shape
        if sy.ndim != 3 or sy.shape[:2] != (b,n) or qx.shape[0] != b or qx.shape[2] != i:
            raise ValueError("Incompatible support/query shapes")
        sm = support_mask if support_mask is not None else torch.ones(b,n,dtype=torch.bool,device=sx.device)
        fm = feature_mask if feature_mask is not None else torch.ones(b,i,dtype=torch.bool,device=sx.device)
        rm = response_mask if response_mask is not None else torch.ones(b,sy.shape[2],dtype=torch.bool,device=sx.device)
        return sx,sy,qx,sm,fm,rm,unbatched


class HeterogeneousMetaRegressor(SetModel):
    def __init__(self, hidden=32, dropout=0.1, relationship_update=True):
        super().__init__()
        h = hidden
        # Each paired attribute/response module uses the SAME module instance.
        self.initial_f = mlp(1,h,h,dropout)
        self.initial_g = mlp(h,h,h,dropout)
        self.sample_f = mlp(h+1,h,h,dropout)
        self.sample_g = mlp(h,h,h,dropout)
        self.relation_f = mlp(h+1,h,h,dropout)
        self.relation_g = mlp(h,h,h,dropout)
        self.query_f = mlp(h+1,h,h,dropout)
        self.query_g = mlp(h,h,h,dropout)
        self.output_f = mlp(2*h,1,h,dropout)
        self.relationship_update = relationship_update

    def forward(self,sx,sy,qx,support_mask=None,feature_mask=None,response_mask=None):
        sx,sy,qx,sm,fm,rm,single = self.inputs(sx,sy,qx,support_mask,feature_mask,response_mask)
        b,n,i = sx.shape
        j,q = sy.shape[2],qx.shape[1]
        v0 = self.initial_g(masked_mean(self.initial_f(sx[...,None]),sm[:,:,None,None],1))
        c0 = self.initial_g(masked_mean(self.initial_f(sy[...,None]),sm[:,:,None,None],1))
        if self.relationship_update:
            ux = self.sample_f(torch.cat([v0[:,None].expand(-1,n,-1,-1),sx[...,None]],-1))
            uy = self.sample_f(torch.cat([c0[:,None].expand(-1,n,-1,-1),sy[...,None]],-1))
            u = self.sample_g(masked_mean(ux,fm[:,None,:,None],2)+masked_mean(uy,rm[:,None,:,None],2))
            vx = self.relation_f(torch.cat([u[:,:,None].expand(-1,-1,i,-1),sx[...,None]],-1))
            cy = self.relation_f(torch.cat([u[:,:,None].expand(-1,-1,j,-1),sy[...,None]],-1))
            v = self.relation_g(masked_mean(vx,sm[:,:,None,None],1))
            c = self.relation_g(masked_mean(cy,sm[:,:,None,None],1))
        else:
            v,c = v0,c0
        zx = self.query_f(torch.cat([v[:,None].expand(-1,q,-1,-1),qx[...,None]],-1))
        z = self.query_g(masked_mean(zx,fm[:,None,:,None],2))
        pairs = torch.cat([c[:,None].expand(-1,q,-1,-1),z[:,:,None].expand(-1,-1,j,-1)],-1)
        pred = self.output_f(pairs).squeeze(-1)
        return pred[0] if single else pred


class DeepSetRegressor(SetModel):
    def __init__(self, hidden=32, dropout=0.1, **kwargs):
        super().__init__()
        self.f = mlp(1,hidden,hidden,dropout)
        self.g = mlp(hidden,1,hidden,dropout)

    def forward(self,sx,sy,qx,support_mask=None,feature_mask=None,response_mask=None):
        sx,sy,qx,sm,fm,rm,single = self.inputs(sx,sy,qx,support_mask,feature_mask,response_mask)
        if sy.shape[-1] != 1:
            raise ValueError("DS control supports scalar-response tasks only")
        z = masked_mean(self.f(qx[...,None]),fm[:,None,:,None],2)
        pred = self.g(z)
        return pred[0] if single else pred


class NeuralProcessRegressor(SetModel):
    """Deterministic Deep Sets conditional model from the paper supplement."""
    def __init__(self,hidden=32,dropout=0.1,**kwargs):
        super().__init__()
        self.attr_f = mlp(1,hidden,hidden,dropout)
        self.attr_g = mlp(hidden,hidden,hidden,dropout)
        self.response_f = mlp(1,hidden,hidden,dropout)
        self.sample_f = mlp(2*hidden,hidden,hidden,dropout)
        self.query_f = mlp(hidden+1,hidden,hidden,dropout)
        self.output_f = mlp(hidden,1,hidden,dropout)

    def forward(self,sx,sy,qx,support_mask=None,feature_mask=None,response_mask=None):
        sx,sy,qx,sm,fm,rm,single = self.inputs(sx,sy,qx,support_mask,feature_mask,response_mask)
        if sy.shape[-1] != 1:
            raise ValueError("NP control supports scalar-response tasks only")
        ax = self.attr_g(masked_mean(self.attr_f(sx[...,None]),fm[:,None,:,None],2))
        ay = self.response_f(sy)
        sample = self.sample_f(torch.cat([ax,ay],-1))
        r = masked_mean(sample,sm[:,:,None],1)
        z = self.query_f(torch.cat([r[:,None,None].expand(-1,qx.shape[1],qx.shape[2],-1),qx[...,None]],-1))
        pred = self.output_f(masked_mean(z,fm[:,None,:,None],2))
        return pred[0] if single else pred


def base_method(method):
    return {"ds_ft":"ds", "np_ft":"np", "hml_ft":"hml"}.get(method,method)


def build_model(method, cfg):
    name = base_method(method).removesuffix("_maml")
    cls = {"hml":HeterogeneousMetaRegressor,"ds":DeepSetRegressor,"np":NeuralProcessRegressor}.get(name)
    if cls is None:
        raise ValueError(f"Unknown neural method: {method}")
    return cls(**cfg)


def forward_batch(model,batch,params=None,query_support=False):
    args = (batch["sx"],batch["sy"],batch["sx"] if query_support else batch["qx"])
    kwargs = {k:batch[k] for k in ["support_mask","feature_mask","response_mask"]}
    return model(*args,**kwargs) if params is None else functional_call(model,params,args,kwargs)


def batch_loss(pred,batch,support=False):
    target = batch["sy"] if support else batch["qy"]
    rows = batch["support_mask"] if support else batch["query_mask"]
    mask = rows[:,:,None] & batch["response_mask"][:,None,:]
    per_task = ((pred-target).square()*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    return per_task.mean()


def adapt_parameters(model,batch,steps,lr,create_graph=False,detach=False):
    params = dict(model.named_parameters())
    if detach:
        params = {k:v.detach().clone().requires_grad_(True) for k,v in params.items()}
    for _ in range(int(steps)):
        loss = batch_loss(forward_batch(model,batch,params,query_support=True),batch,support=True)
        grads = torch.autograd.grad(loss,tuple(params.values()),create_graph=create_graph,allow_unused=True)
        params = {k:p if g is None else p-lr*g for (k,p),g in zip(params.items(),grads)}
    return params
