# Contributing

Guidelines for contributing to this project.

---

## Commit Messages

We use **Conventional Commits** for clear, searchable history.

### Format

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

### Types

| Type | Use when |
|------|----------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change, no fix/feature |
| `test` | Adding or updating tests |
| `chore` | Build, deps, tooling |

### Examples

```
feat(load_data): add FE and BA channel support
fix(load_data): handle missing RPM in older .mat files
docs: add dataset download instructions
chore: add pre-commit config
```

### Cursor IDE (✨ Generate Commit Message)

Project rules are often **not used** by Generate Commit Message. See **[docs/CURSOR_COMMIT_SETUP.md](docs/CURSOR_COMMIT_SETUP.md)** for working solutions (User Rules, Chat workflow, or pre-commit validation).

### Setup commit template (optional)

```bash
git config commit.template .gitmessage
```

Then each `git commit` opens with the template as a reminder.

### Validate commits (optional)

Install pre-commit and commitizen to enforce the format:

```bash
pip install pre-commit commitizen
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## Code Style

- **Formatter:** Black (line length 88)
- **Linter:** Ruff

### Run manually

```bash
black src/
ruff check src/
```

### Or use pre-commit

```bash
pre-commit run --all-files
```

---

## Project Setup

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

---

## Adding Code

1. Follow the structure in [PLAN.md](PLAN.md)
2. Add docstrings to public functions
3. Keep functions focused; split if they do too much
