import numpy as np
import pytest

from wing_hml.baselines import HISTORICAL,build_estimator,candidates


@pytest.mark.parametrize("name",HISTORICAL+["mean","linear_ridge","lasso","bayesian_ridge","linear_kernel_ridge","fixed_rbf_gp"])
def test_baselines_handle_three_support_points_and_constant_columns(name):
    x=np.array([[0.,1.,1.],[0.,2.,1.],[0.,3.,1.]])
    y=np.array([0.,1.,2.])
    model=build_estimator(name,candidates(name,small=True)[0],3,42)
    model.fit(x,y)
    pred=np.asarray(model.predict([[0.,1.5,1.]]))
    assert pred.shape==(1,) and np.isfinite(pred).all()
    if name=="rbf": np.testing.assert_array_equal(model.centers_,x)
