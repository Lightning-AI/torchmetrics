from torchmetrics.functional.text.infolm import _ALLOWED_INFORMATION_MEASURE_LITERAL, _IMEnum


def test_im_enum_literal_equivalence():
    literal_values = _ALLOWED_INFORMATION_MEASURE_LITERAL.__args__
    enum_values = tuple(im.lower() for im in _IMEnum._member_names_)
    if literal_values != enum_values:
        raise ValueError(
            "Values of `_ALLOWED_INFORMATION_MEASURE_LITERAL` and `_IMEnum` are expected to be same, but got "
            f"{literal_values} and {enum_values}"
        )
