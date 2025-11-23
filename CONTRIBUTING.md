# Contributing to ODRV2

Thank you for your interest in contributing to ODRV2! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs. actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, GPU/CPU)
- **Error messages** and stack traces

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** explaining why this enhancement would be useful
- **Possible implementation** if you have ideas
- **Examples** from other projects if applicable

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** with clear commit messages
4. **Test your changes** thoroughly
5. **Update documentation** if needed
6. **Submit a pull request**

#### Pull Request Guidelines

- Follow the existing code style (PEP 8 for Python)
- Add type hints to new functions
- Include docstrings for public APIs
- Add tests for new features
- Update README.md if adding user-facing features
- Keep pull requests focused (one feature/fix per PR)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ODRV2.git
cd ODRV2

# Create virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install ruff mypy pytest
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function signatures
- Maximum line length: 120 characters
- Use meaningful variable names

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(inference): add uncertainty quantification to predictions

- Add ensemble disagreement calculation
- Add predictive entropy metric
- Update API response to include uncertainty scores

Closes #123
```

## Testing

```bash
# Run tests
pytest tests/

# Run type checking
mypy src/ --ignore-missing-imports

# Run linting
ruff check src/ scripts/
```

## Documentation

- Update docstrings when modifying functions
- Update README.md for user-facing changes
- Update TECHNICAL_REPORT.md for methodology changes
- Add inline comments for complex logic

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] External dataset validation (IDRiD, APTOS, Messidor-2)
- [ ] Model compression and optimization
- [ ] Additional evaluation metrics (AUROC, PR curves)
- [ ] Docker container for easy deployment
- [ ] Jupyter notebook tutorials

### Medium Priority
- [ ] Additional augmentation strategies
- [ ] Alternative backbone architectures
- [ ] Hyperparameter optimization tools
- [ ] Model interpretability improvements
- [ ] CLI improvements

### Good First Issues
- [ ] Documentation improvements
- [ ] Code comments and docstrings
- [ ] Unit tests for utilities
- [ ] Example notebooks
- [ ] Bug fixes

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Cited in academic papers (for significant contributions)

## Questions?

Feel free to:
- Open a GitHub issue for discussion
- Start a thread in GitHub Discussions
- Contact the maintainers

Thank you for contributing to ODRV2! 🎉
