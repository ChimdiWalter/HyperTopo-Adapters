def test_dummy():
    import src.models.manifold as M
    import src.losses.hyperbolic as H
    import src.losses.topology as T
    import src.data.lesion_dataset as D
    assert hasattr(M, 'ProductManifoldHead')
    assert hasattr(H, 'info_nce_from_pairs')
    assert hasattr(D, 'LesionDataset')
