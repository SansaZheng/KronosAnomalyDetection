import numpy as np

from kronos_ad.proxy import select_targets,proxy_input,restore_prediction,ordinal_stamps


def test_targets_have_only_earlier_within_group_context():
    groups=np.repeat(['first','second'],20)
    parts=np.tile(['reference']*8+['calibration']*6+['test']*6,2)
    targets,history,local=select_targets(parts,groups,lookback=4,calibration_cap=5,test_cap=7)
    assert len(targets)==12
    assert (history<targets[:,None]).all()
    assert np.array_equal(groups[history],np.repeat(groups[targets,None],4,axis=1))
    assert (local[targets]>=4).all()
    a,b,c=select_targets(parts,groups,4,5,7)
    np.testing.assert_array_equal(a,targets)
    np.testing.assert_array_equal(b,history)


def test_proxy_roundtrip_does_not_invent_volume_or_amount():
    history=np.array([[1.,2.,3.],[20.,30.,40.]])
    center=np.array([2.,20.]);scale=np.array([1.,10.])
    x,mean,std=proxy_input(history,center,scale)
    np.testing.assert_array_equal(x[:,:,0],x[:,:,3])
    assert (x[:,:,4:]==0).all()
    restored=restore_prediction(x[:,-1,3],mean,std,center,scale)
    np.testing.assert_allclose(restored,history[:,-1],rtol=1e-6)
    np.testing.assert_array_equal(history,np.array([[1.,2.,3.],[20.,30.,40.]]))


def test_input_clipping_is_limited_to_proxy_history_copy():
    history=np.zeros((1,64));history[0,-1]=1000
    x,_,_=proxy_input(history,np.array([0.]),np.array([1.]))
    assert x.max()<=5
    assert history[0,-1]==1000


def test_ordinal_timestamp_shapes_and_target_order():
    stamps=ordinal_stamps(np.array([[0,1,2],[60,61,62]]))
    assert stamps.shape==(2,3,5)
    assert stamps[0,:,0].tolist()==[0,1,2]
    assert stamps[1,:,1].tolist()==[1,1,1]


def test_cache_fingerprint_binds_context_values():
    from kronos_ad.proxy import input_fingerprint
    x=np.arange(12,dtype=float).reshape(3,4)
    before=input_fingerprint(x)
    assert before==input_fingerprint(x.copy())
    x[0,0]+=1
    assert before!=input_fingerprint(x)
