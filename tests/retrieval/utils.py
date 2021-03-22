def _assert_error(function, error, *args, **kwargs):
    """ Assert that `function(*args)` raises `error`. """
    try:
        function(*args, **kwargs)
        assert False  # assert exception is raised
    except Exception as e:
        assert isinstance(e, error)
