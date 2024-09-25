#!/usr/bin/env python

import argparse
import shutil
import subprocess
import textwrap
from pathlib import Path

from .backend import Backend, GroqBackend, OpenaiBackend


def clean_code(code: str) -> str:
    """Clean the code by removing unwanted markdown markers or other invalid syntax.
    This will ensure the code is valid Python code.
    """
    # Remove Markdown code block markers if present
    return textwrap.dedent(code.replace("```", "")).strip()


def programmer_agent(backend: Backend, user_query: str) -> str:
    system_message = {
        "role": "system",
        "content": "You are a programmer. Write Python code to solve the user's problem. "
        "Provide only the code, do not include explanations. "
        "Only write functions without side effects, so no main function. "
        "Add type annotations to all function definitions. "
        "Write doc strings for all functions.",
    }
    user_message = {"role": "user", "content": user_query}
    messages = [system_message, user_message]
    code = backend.chat_completion(messages)
    return clean_code(code)  # Clean the code before returning


def type_check_and_correct(
    backend: Backend, code_file: Path, user_query: str, max_retries: int
) -> Path | None:
    """In a loop: send the code to mypy, if it passes, then return, else use an
    agent to correct the code.

    returns A path to the updated (or unmodified) code if type checking passes
    in <= max_retries iterations, otherwise returns None.

    The user_query argument should be the query provided by the user, it is used to remind
    the bot about the overall purpose of the code when correcting type errors.
    """
    orig_code_file = code_file
    tries: int = 0
    while tries <= max_retries:
        # Run mypy on the code.
        try:
            subprocess.run(
                ["mypy", code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Type checking failed.
            tries += 1
            # Read the code from code_file.
            with open(code_file) as f:
                code: str = f.read()
            # Write the type errors to a file.
            with open(code_file.with_name(code_file.stem + "_type_errors.txt"), "w") as f:
                f.write(e.stdout)
            new_code = programmer_agent(
                backend,
                textwrap.dedent(f"""\
                        The specification of the code is:
                            {user_query}

                        The code was:
                        ```
                        {code}
                        ```

                        However, type checking with mypy failed with the following errors:
                            {e.stdout}

                        Please correct the code to make it pass type checking.
                    """),
            )
            # Write new_code to a file.
            code_file = orig_code_file.with_stem(orig_code_file.stem + f"_types_retry_{tries}")
            with open(code_file, "w") as f:
                f.write(new_code)
        else:
            # The type checking was successful.
            print(f"Successful type checking with {tries} retries.")
            return code_file
    print(f"Failed type checking after {tries} tries.")


def test_designer_agent(backend: Backend, code_file: Path, user_query: str) -> str:
    """Given the code from the code and the query from the user which describes
    the purpose of the code, write tests for it.

    We will first generate a stub file with all function definitions and docstrings,
    then feed that to the LLM to generate tests.

    Returns the test code, which should be in the same file as the code.
    """
    # Run stubgen to create the stub file.
    subprocess.run(
        ["stubgen", code_file, "--include-docstrings", "-o", code_file.parent],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    # Read the stub file.
    with open(code_file.with_suffix(".pyi")) as f:
        stub: str = f.read()
    # Write the tests.
    system_message = {
        "role": "system",
        "content": "You are a test designer. Write relevant Python unit tests to "
        "verify the correctness of code solving the user's problem. Use the unit "
        "test framework in Python. Provide only the test code, do not include "
        "explanations. Write type annotations for the test code. "
        "Do not replicate the function definitions.",
    }
    user_message = {
        "role": "user",
        "content": textwrap.dedent(f"""\
            Problem:
                {user_query}

            The definitions of the functions to write tests for are as follows:
            ```
            {stub}
            ```
        """),
    }
    messages = [system_message, user_message]
    test_code = backend.chat_completion(messages)
    return clean_code(test_code)  # Clean the test code before returning


def run_tests_and_correct(
    backend: Backend, code_file: Path, test_code: str, user_query: str, max_retries: int
) -> str | None:
    """Given the code and the tests, merge them into the same python file and run
    the unit tests in it. If the tests succeed, then return with the final code and tests,
    else asks the LLM to rewrite the code to conform to the tests.

    Returns the final code and tests if successful, if failed after max_retries returns None.
    """
    orig_code_file = code_file
    if (c := type_check_and_correct(backend, code_file, user_query, max_retries)) is not None:
        code_file = c
    else:
        return
    with open(code_file) as f:
        code: str = f.read()
    test_code_separator: str = "## Tests"
    tries: int = 0
    while tries <= max_retries:
        # Check that the code and the tests passes type checking.
        code_with_tests: str = code + "\n\n" + test_code_separator + "\n\n" + test_code
        code_with_tests_file = code_file.with_stem(
            orig_code_file.stem + "_with_tests" + (f"_retry_{tries}" if tries > 0 else "")
        )
        with open(code_with_tests_file, "w") as f:
            f.write(code_with_tests)
        if (
            c := type_check_and_correct(backend, code_with_tests_file, user_query, max_retries)
        ) is not None:
            code_with_tests_file = c
        else:
            return None
        with open(code_with_tests_file) as f:
            code_with_tests = f.read()
        # Split it into code and test_code.
        try:
            [code, test_code] = code_with_tests.split(test_code_separator)
        except ValueError:
            # It probably did only update the code.
            code = code_with_tests
            code_with_tests = code + "\n\n" + test_code_separator + "\n\n" + test_code
            with open(code_with_tests_file, "w") as f:
                f.write(code_with_tests)
        code = code.strip()
        test_code = test_code.strip()
        # Run the tests using unittest
        try:
            subprocess.run(
                ["python", "-m", "unittest", code_with_tests_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Tests failed
            tries += 1
            # Write test errors to a file.
            with open(
                code_with_tests_file.with_name(code_with_tests_file.stem + "_errors"), "w"
            ) as f:
                f.write(e.stdout)
            # Ask programmer agent to update the code.
            code = programmer_agent(
                backend,
                textwrap.dedent(f"""\
                        The specification of the code is:
                            {user_query}

                        The code was:
                        ```
                        {code}
                        ```

                        However, the tests failed with the following errors:
                            {e.stdout}

                        Please correct the code to make it pass the tests.
                    """),
            )
        else:
            # Tests succeeded!
            print(f"The tests passed after {tries} retries.")
            return code_with_tests
    print(f"Failed testing after {tries} tries.")


def run(backend: Backend, max_retries: int) -> None:
    user_query: str = input("Enter your query: ")

    # Programmer writes code
    code: str = programmer_agent(
        backend,
        textwrap.dedent(f"""\
            Problem:
                {user_query}

            Write functions with type annotations which could solve the user's problem.
        """),
    )
    target_dir = Path("target") / "coder_agent"
    if target_dir.is_dir():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    code_file = target_dir / "code.py"
    with open(code_file, "w") as f:
        f.write(code)
    test_code: str = test_designer_agent(backend, code_file, user_query)
    code_with_tests = run_tests_and_correct(backend, code_file, test_code, user_query, max_retries)
    print(code_with_tests)


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
