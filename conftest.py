def pytest_addoption(parser):
    parser.addoption(
        "--sub",
        action="store",
        default=None,
        help="Path to a submission CSV to validate against keyword-stuffing test",
    )
