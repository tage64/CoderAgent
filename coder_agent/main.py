#!/usr/bin/env python

import argparse
import os
import subprocess
import tempfile
import textwrap

from .backend import Backend, GroqBackend, OpenaiBackend


def clean_code(code: str) -> str:
    """Clean the code by removing unwanted markdown markers or other invalid syntax.
    This will ensure the code is valid Python code.
    """
    # Remove Markdown code block markers if present
    code = code.replace("```", "")
    return code.strip()


def programmer_agent(backend: Backend, user_query: str) -> str:
    system_message = {
        "role": "system",
        "content": "You are a programmer. Write Python code to solve the user's problem. "
        "Provide only the code, do not include explanations.",
    }
    user_message = {
        "role": "user",
        "content": textwrap.dedent(f"""\
            Problem:

            {user_query}

            Encapsulate your solution in functions or classes as appropriate.
        """),
    }

    messages = [system_message, user_message]

    code = backend.chat_completion(messages)
    return clean_code(code)  # Clean the code before returning


def test_designer_agent(backend: Backend, user_query: str) -> str:
    system_message = {
        "role": "system",
        "content": "You are a test designer. Write relevant Python unit tests to "
        "verify the correctness of code solving the user's problem. Use the unit "
        "test framework in Python. Provide only the test code, do not include "
        "explanations.",
    }
    user_message = {
        "role": "user",
        "content": textwrap.dedent(f"""\
            Problem:

            {user_query}

            Assume the programmer's code is in a file named 'code.py'. Import
            the necessary functions or classes from 'code.py' to perform the tests.
        """),
    }

    messages = [system_message, user_message]

    test_code = backend.chat_completion(messages)
    return clean_code(test_code)  # Clean the test code before returning


def test_agent(backend: Backend, code: str, test_output: str, test_errors: str) -> str:
    """This is the agent that reads the output from failed tests and rewrites the code
    to pass the tests.
    """
    system_message = {
        "role": "system",
        "content": "You are a test agent. Evaluate the programmer's code based on the test "
        "results and improve the code so that it passes the tests. "
        "The tests should not be altered.",
    }
    user_message = {
        "role": "user",
        "content": textwrap.dedent(f"""\
            The programmer wrote the following code:

            {code}

            The tests produced the following output:

            {test_output}

            There were errors:

            {test_errors}

            Rewrite the code so that it passes the tests without errors.
            Write only the code and no explonations.
        """),
    }

    messages = [system_message, user_message]

    explanation = backend.chat_completion(messages)
    return explanation


def code_executor(code: str, test_code: str) -> tuple[bool, str, str]:
    """Returns a tuple (success, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tempdir:
        # Write the programmer's code to code.py
        code_file = os.path.join(tempdir, "code.py")
        with open(code_file, "w") as f:
            f.write(code)

        # Write the test code to test_code.py
        test_file = os.path.join(tempdir, "test_code.py")
        with open(test_file, "w") as f:
            f.write(test_code)

        # Run the tests using unittest
        try:
            result = subprocess.run(
                ["python", "-m", "unittest", "test_code.py"],
                cwd=tempdir,
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout
            errors = result.stderr
            return True, output, errors
        except subprocess.CalledProcessError as e:
            # Tests failed
            output = e.stdout
            errors = e.stderr
            return False, output, errors


def run(backend: Backend, n_retries: int) -> None:
    user_query: str = input("Enter your query: ")

    # Programmer writes code
    code = programmer_agent(backend, user_query)
    print("\nProgrammer's code:")
    print(code)

    # Test designer writes tests
    test_code = test_designer_agent(backend, user_query)
    print("\nTest Designer's test code:")
    print(test_code)

    # Test and correct loop:
    for i in range(n_retries):
        # Execute code and tests
        success, output, errors = code_executor(code, test_code)

        if success:
            print(f"\nAll tests passed after {i} retries!")
            break
        else:
            print("\nTests failed.")
            print("Test Output:")
            print(output)
            print("Test Errors:")
            print(errors)
            print("\n", "-" * 60)

            # Test agent evaluates the code
            code = test_agent(backend, code, output, errors)
            print("\nThe code has been rewritten to:")
            print(code)
    else:
        print(f"Failed after {n_retries} retries.")


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-b", "--backend", choices=["groq", "openai"], default="groq", help="Which backend to use."
    )
    argparser.add_argument(
        "-n",
        "--retries",
        type=int,
        default=4,
        help="Maximum number of iterations in the test-rewrite loop.",
    )
    argparser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for the model.  Should be in the range [0, 1].",
    )
    argparser.add_argument("-m", "--model", help="Name of the specific model for the backend.")
    argparser.add_argument("--max-tokens", type=int, default=1500, help="Maximum number of tokens.")
    args = argparser.parse_args()
    match args.backend:
        case "groq":
            backend = GroqBackend()
        case "openai":
            backend = OpenaiBackend()
        case x:
            exit(f"Bad backend: {x}")
    backend.max_tokens = args.max_tokens
    backend.temperature = args.temperature
    if args.model is not None:
        backend.model = args.model
    run(backend, args.retries)


if __name__ == "__main__":
    main()
