from pc_classifier.neuralnetcode import make_conv_seq, make_lin_seq


def test_make_conv_seq():
    assert make_conv_seq([3, 64, 128, 1024]) is not None


def test_make_lin_seq():
    assert make_lin_seq([1024, 512, 256]) is not None
