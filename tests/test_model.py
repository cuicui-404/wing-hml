import numpy as np
import pytest
import torch

from wing_hml.models import HeterogeneousMetaRegressor,build_model,adapt_parameters,forward_batch,batch_loss
from wing_hml.tasks import Episode,pack_episodes
from wing_hml.training import predict_batch


def ep(n=7,d=3,q=4,j=1):
    rng=np.random.default_rng(n+d+j)
    return Episode(rng.normal(size=(n,d)),rng.normal(size=(n,j)),rng.normal(size=(q,d)),
                   rng.normal(size=(q,j)),np.arange(n),np.arange(q),"test")


def test_hml_matches_paper_equations():
    torch.manual_seed(42)
    m=HeterogeneousMetaRegressor(8,0).eval()
    sx,sy,qx=torch.randn(7,3),torch.randn(7,2),torch.randn(4,3)
    v0=m.initial_g(m.initial_f(sx[...,None]).mean(0))
    c0=m.initial_g(m.initial_f(sy[...,None]).mean(0))
    u=m.sample_g(m.sample_f(torch.cat([v0[None].expand(7,-1,-1),sx[...,None]],-1)).mean(1)+
                 m.sample_f(torch.cat([c0[None].expand(7,-1,-1),sy[...,None]],-1)).mean(1))
    v=m.relation_g(m.relation_f(torch.cat([u[:,None].expand(-1,3,-1),sx[...,None]],-1)).mean(0))
    c=m.relation_g(m.relation_f(torch.cat([u[:,None].expand(-1,2,-1),sy[...,None]],-1)).mean(0))
    z=m.query_g(m.query_f(torch.cat([v[None].expand(4,-1,-1),qx[...,None]],-1)).mean(1))
    manual=m.output_f(torch.cat([c[None].expand(4,-1,-1),z[:,None].expand(-1,2,-1)],-1)).squeeze(-1)
    torch.testing.assert_close(m(sx,sy,qx),manual)


@pytest.mark.parametrize("name",["hml","ds","np"])
def test_padding_invariance_and_query_label_is_not_input(name):
    torch.manual_seed(4)
    m=build_model(name,{"hidden":8,"dropout":0.,"relationship_update":True}).eval()
    a,b=ep(),ep(11,5,9)
    packed=pack_episodes([a,b],"cpu")
    result=forward_batch(m,packed)
    for k,e in enumerate([a,b]):
        individual=forward_batch(m,pack_episodes([e],"cpu"))
        torch.testing.assert_close(result[k,:len(e.qx)],individual[0],rtol=1e-5,atol=1e-6)
    packed["qy"]+=1000
    torch.testing.assert_close(forward_batch(m,packed),result)


def test_permutations_and_arbitrary_response_sizes():
    torch.manual_seed(5)
    m=HeterogeneousMetaRegressor(8,0).eval()
    sx,sy,qx=torch.randn(9,5),torch.randn(9,3),torch.randn(4,5)
    result=m(sx,sy,qx)
    rows=torch.randperm(9); cols=torch.randperm(5); responses=torch.randperm(3)
    shuffled=m(sx[rows][:,cols],sy[rows][:,responses],qx[:,cols])
    torch.testing.assert_close(result[:,responses],shuffled,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize("name",["ds_maml","np_maml"])
def test_second_order_meta_gradient_and_frozen_target_weights(name):
    torch.manual_seed(11)
    m=build_model(name,{"hidden":8,"dropout":0.,"relationship_update":True})
    batch=pack_episodes([ep()],"cpu")
    before={k:v.detach().clone() for k,v in m.named_parameters()}
    adapted=adapt_parameters(m,batch,2,0.01,create_graph=True)
    loss=batch_loss(forward_batch(m,batch,adapted),batch)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    pred=predict_batch(m,batch,name,2,0.01)
    assert torch.isfinite(pred).all()
    for k,v in m.named_parameters(): torch.testing.assert_close(v,before[k])
    optimizer=torch.optim.Adam(m.parameters(),lr=0.001)
    optimizer.step()
    assert any(not torch.equal(p,before[k]) for k,p in m.named_parameters())
