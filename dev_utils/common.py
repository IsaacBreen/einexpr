import libcst.matchers as m


unimplemented_function_matcher = m.FunctionDef(
    body=m.OneOf(
        # TODO: This bit should be simplified and generalised to a function definition formatted any way, not just these two ways.
        m.SimpleStatementSuite(
            body=[
                m.ZeroOrOne(m.Expr(value=m.SimpleString())),
                m.Raise(exc=m.Call(func=m.Name(value="NotImplementedError")))
            ]
        ),
        m.IndentedBlock(
            body=[
                m.ZeroOrOne(m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])),
                m.SimpleStatementLine(
                    body=[
                        m.Raise(
                            exc=m.OneOf(
                                m.Name(value="NotImplementedError"),
                                m.Call(func=m.Name(value="NotImplementedError"),
                                )
                            )
                        )
                    ]
                )
            ]
        )
    )
)