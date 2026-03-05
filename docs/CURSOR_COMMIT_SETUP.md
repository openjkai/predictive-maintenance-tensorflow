# Cursor: Conventional Commits for ✨ Generate Message

Cursor's **Generate Commit Message** (✨) often ignores project rules. Use one of these workarounds.

---

## Option 1: Add to User Rules (recommended)

1. Open **Cursor Settings** (Ctrl+, or Cmd+,)
2. Click **Rules** in the left sidebar (under Cursor Settings)
3. In **"Rules for AI"**, add:

   ```
   When generating git commit messages, always use Conventional Commits: type(scope): description. Types: feat, fix, docs, style, refactor, test, chore. Keep subject under 50 chars. Example: feat(load_data): add channel support.
   ```

4. Save. This applies to all your projects.

---

## Option 2: "Train" with existing commits

Cursor learns from your repo's commit history. After a few manual conventional commits (e.g. `feat(load_data): add load_dataset`), the AI may start following that style.

---

## Option 3: Use Chat instead of ✨

1. Stage your changes
2. Open Cursor Chat
3. Type: `Generate a commit message for my staged changes using Conventional Commits. Format: type(scope): description`
4. Copy the suggested message into the commit box

The chat AI respects project rules (including `.cursor/rules/git-commits.mdc`).

---

## Option 4: Pre-commit validation

Install the commit-msg hook so invalid messages are rejected:

```bash
pip install pre-commit commitizen
pre-commit install --hook-type commit-msg
```

Then fix the message if Cursor generates something that doesn't match.

---

*Cursor is working on native support for commit message rules ([forum](https://forum.cursor.com/t/set-rules-for-generate-commit-message/151130)).*
