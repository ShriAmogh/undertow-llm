# Contributing to lensllm

Thank you for your interest in contributing to `lensllm`!

## Quickstart for Developers

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amogharora/lensllm.git
   cd lensllm
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[prod,dev]"
   ```

4. **Run the test suite**:
   ```bash
   # Run all unit and integration tests
   pytest tests/ -v

   # Run complete multi-phase feature verification suite
   python complete_testing.py
   ```

## Pull Request Guidelines

- Ensure all existing unit tests pass before submitting your PR.
- Add test coverage for any new parameters, backends, or features.
- Keep code concise, documented, and properly typed.
