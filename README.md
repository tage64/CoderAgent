# Coder Agent

This is an experimental project to create a coder agent which uses LLMs to write Python code.

## Design

There are three agents:

* The coder agent writes some Python function based on a user query.
* The test designer agent writes tests for the code.
* The third agent reads error output from the tests and tries to improve the code.

The test designer and the third agent are run in a loop until all tests pass.

## Running

You need at least Python 3.10 and [Python Poetry][1] installed. You can then run the following
commands:

```
poetry install
poetry run coder_agent
```

You may specify arguments like `--backend`, `--temperature` and `--retries` on the command line. For
reference run:
```
poetry run coder_agent --help
```

## Example Queries

Try to input the query: "Write a function which sums two numbers"

[1]: https://python-poetry.org
