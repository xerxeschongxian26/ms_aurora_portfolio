def test_package_imports() -> None:
    import aurora_inference

    assert aurora_inference.__name__ == "aurora_inference"


def test_type_annotation_function() -> None:
    x: int = "Nope"
    assert x.__name__ == "x"
