from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from .legacy_estimators import (MARSRegressor,RBFBasisRegressor,TensorProductSplineTransformer,
                                LocalMultilinearRegressor,XGBoostRegressorLite)

HISTORICAL = ["gpr","rsm","svr","rbf","random_forest","polynomial_ridge","polynomial_elasticnet",
              "mars","kernel_ridge","xgboost_lite","gam","tensor_spline","local_multilinear"]
NEURAL = ["hml","ds","ds_ft","ds_maml","np","np_ft","np_maml","hml_ft"]
SIMPLE = ["mean","linear_ridge","lasso","bayesian_ridge","linear_kernel_ridge","target_mlp","fixed_rbf_gp"]
ALL_MODELS = NEURAL+HISTORICAL+SIMPLE+["tabpfn"]


def candidates(name,small=False):
    if name=="gpr": return [{"kernel":"rbf"}] if small else [{"kernel":k} for k in ["rbf","matern15","matern25"]]
    if name=="svr": return [{"C":10.0,"epsilon":0.05}] if small else [{"C":c,"epsilon":e} for c in [10.,100.] for e in [0.01,0.05]]
    if name=="rbf": return [{"alpha":0.001}] if small else [{"alpha":a} for a in [1e-6,1e-3]]
    if name=="random_forest": return [{"min_samples_leaf":1}] if small else [{"min_samples_leaf":n} for n in [1,3,5,8]]
    if name in {"linear_ridge","lasso","linear_kernel_ridge"}:
        return [{"alpha":1.}] if small else [{"alpha":a} for a in [1e-3,1e-2,0.1,1.,10.]]
    if name=="target_mlp":
        return [{"lr":0.001,"epochs":200}] if small else [{"lr":lr,"epochs":n} for lr in [1e-4,1e-3] for n in [50,200,500]]
    if name in {"ds_ft","np_ft","ds_maml","np_maml","hml_ft"}:
        return [{"inner_lr":0.001}] if small else [{"inner_lr":lr} for lr in [1e-4,1e-3]]
    return [{}]


def build_estimator(name,params,n_features,seed):
    p = dict(params)
    if name in {"gpr","fixed_rbf_gp"}:
        kind = p.get("kernel","rbf")
        k = RBF(np.ones(n_features),length_scale_bounds="fixed") if kind=="rbf" else Matern(np.ones(n_features),length_scale_bounds="fixed",nu=1.5 if kind=="matern15" else 2.5)
        k = ConstantKernel(1.0,constant_value_bounds="fixed")*k+WhiteKernel(0.05**2,noise_level_bounds="fixed")
        return GaussianProcessRegressor(kernel=k,alpha=1e-8,optimizer=None,normalize_y=False,random_state=seed)
    if name=="rsm": return make_pipeline(PolynomialFeatures(2,include_bias=False),LinearRegression())
    if name=="svr": return SVR(kernel="rbf",gamma="scale",**p)
    if name=="rbf": return RBFBasisRegressor(n_centers=None,**p)
    if name=="random_forest": return RandomForestRegressor(n_estimators=100,max_depth=None,max_features=1.,random_state=seed,n_jobs=1,**p)
    if name=="polynomial_ridge": return make_pipeline(PolynomialFeatures(3,include_bias=False),StandardScaler(),Ridge(alpha=2.0))
    if name=="polynomial_elasticnet": return make_pipeline(PolynomialFeatures(3,include_bias=False),StandardScaler(),ElasticNet(alpha=0.03,l1_ratio=0.35,max_iter=50000,random_state=seed))
    if name=="mars": return MARSRegressor(max_terms=14,max_degree=2,alpha=1e-5)
    if name=="kernel_ridge": return KernelRidge(alpha=0.08,kernel="rbf",gamma=0.6)
    if name=="xgboost_lite": return XGBoostRegressorLite(n_estimators=120,learning_rate=0.05,max_depth=2,min_child_weight=4.,reg_lambda=8.,gamma=0.02,subsample=0.9,random_state=seed)
    if name=="gam": return make_pipeline(SplineTransformer(n_knots=4,degree=3,include_bias=False),Ridge(alpha=1.0))
    if name=="tensor_spline": return make_pipeline(TensorProductSplineTransformer(n_knots=3,degree=2),Ridge(alpha=2.0))
    if name=="local_multilinear": return LocalMultilinearRegressor(n_neighbors=20,alpha=1e-5)
    if name=="mean": return DummyRegressor(strategy="mean")
    if name=="linear_ridge": return Ridge(**p)
    if name=="lasso": return Lasso(max_iter=50000,random_state=seed,**p)
    if name=="bayesian_ridge": return BayesianRidge()
    if name=="linear_kernel_ridge": return KernelRidge(kernel="linear",**p)
    if name=="target_mlp": return MLPRegressor(hidden_layer_sizes=(32,32),activation="relu",solver="adam",max_iter=p["epochs"],learning_rate_init=p["lr"],early_stopping=False,tol=0.,n_iter_no_change=p["epochs"]+1,random_state=seed)
    if name=="tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError("Optional baseline: install the tabpfn extra, obtain weights and record their license/version") from exc
        return TabPFNRegressor(random_state=seed,device="cpu",n_estimators=4)
    raise ValueError(f"Unknown conventional model: {name}")


def resolve_models(names):
    expanded = []
    for name in names:
        selected = {"historical":HISTORICAL,"neural":NEURAL[:7],"simple":SIMPLE}.get(name,[name])
        for model in selected:
            if model not in ALL_MODELS: raise ValueError(f"Unknown model: {model}; available: {ALL_MODELS}")
            if model not in expanded: expanded.append(model)
    return expanded
