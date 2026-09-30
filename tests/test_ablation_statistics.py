import pytest

from scripts.ablation_statistics import cluster_interval


def test_macro_and_release_means_have_separate_cluster_intervals_and_stable_order():
    a=cluster_interval([1,1,0],['same','same','other'],repeats=400)
    b=cluster_interval([0,1,1],['other','same','same'],repeats=400)
    assert a==b
    assert a['release_row_mean']==pytest.approx(2/3)
    assert a['unique_question_macro_mean']==.5
    assert a['unique_question_macro_ci95']==[0,1]
