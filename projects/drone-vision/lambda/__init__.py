"""
Marks the local *lambda* directory as a proper Python package so that

    import lambda.app

resolves to *this* folder (not the third-party “lambda” package on PyPI).

Nothing else is needed.
"""
