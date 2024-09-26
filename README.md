# Coder Agent

This is an experimental project to create a coder agent which uses LLMs to write Python code.

## Procedure

1. Ask the user for a problem specification.
2. Use the "programmer agent" to write one or more python functions which solves the problem.  It is
   **important** that the functions doesn't perform any side effects like printing or writing to the
   file system as that is hard to test.  The system prompt for the agent includes instructions to
   add [type annotations][2] and write docstrings for all functions.
3. Type checking is performed in the following steps:
    a) The code is run through the static type checker [Mypy][3].  If it passes type checking, then
       we are done, else goto step b).
    b) The programmer agent is passed a prompt with the original code, the problem specification
       from the user, and the type errors, it is asked to solve the type errors.
    c) Go back to step a) if not the maximum number of retries is reached, then exit.
4. Ask the "test designer agent" to write tests for the functions:
    a) Generate a so called [stub file][4] with [Stubgen][5].  A stub file is basically all function
       definitions with type annotations and docstrings but excluding function bodies.
    b) Feed the problem specification and the stub file to the test designer agent and ask it to
       write tests for all functions using [unittest][6].
5. Merge the code and the tests into the same file and type check in the same way as in step 3.
6. Run the tests.  If they succeed, we're done, else go to step 7.
7. Feed the code (but not the tests) and the test errors to the programmer agent and ask it to
   update the code so that it passes the tests.
8. Go back to step 5 and retry with the updated code.  There is of course a maximum number of
   retries here as well.

## Running

You need at least Python 3.10 and [Python Poetry][1] installed. You can then run the following
commands:

```
poetry install
poetry run coder_agent
```

You may specify arguments like `--backend`, `--temperature`, `--retries` and `--no-interactive` on
the command line. For reference run:
```
poetry run coder_agent --help
```

## Example Queries

- Write a function which sums two numbers.
- Write a function which computes the factorial of an integer.
- Write a function which computes the squared factorial of an integer.

## Limitations

One limitation is that the tests will never be improved. So if there is a bug in the tests the
programmer agent has no chance of correcting the code.

[1]: https://python-poetry.org
[2]: https://docs.python.org/3/library/typing.html
[3]: https://www.mypy-lang.org
[4]: https://mypy.readthedocs.io/en/stable/stubs.html
[5]: https://mypy.readthedocs.io/en/stable/stubgen.html
[6]: https://docs.python.org/3/library/unittest.html
