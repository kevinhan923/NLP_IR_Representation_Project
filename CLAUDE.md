# Project Instructions

This file gives Claude Code project-level instructions.

## General Rules

- Inspect the project before editing.
- Make small and safe changes.
- Preserve existing features.
- Do not delete files unless explicitly asked.
- Do not modify unrelated files.
- Keep code beginner-friendly and readable.
- Explain changed files after editing.
- Test after editing when possible.

## Multi-Agent Workflow

This project can use a reusable Claude Code multi-agent workflow through `/auto-flow`.

The main agent acts as the coordinator.

The workflow is:

1. Receive the user's requirement.
2. Use the planner agent to create a task plan.
3. Pick one task at a time.
4. Use the developer agent to implement only the current task.
5. Use the tester agent to test the current task.
6. If the tester reports PASS, continue to the next task.
7. If the tester reports FAIL, do not start a new developer task.
8. Resume the same developer task using the current task context and bug report.
9. After the developer fixes the bug, send the same task back to the tester.
10. Repeat until all tasks pass or retry limit is reached.

The `.claude/state/` folder is created only when `/auto-flow` runs.
