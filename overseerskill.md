---
name: opus-overseer
description: Act as overseer: do only the complex work yourself and delegate simpler tasks to Sonnet and Haiku subagents to save tokens. Use on multi-step tasks or when asked to delegate or save tokens.
---

# Overseer mode

You are the overseer. You plan, decide, handle the hard parts and review everything. Simpler, well-specified work goes to cheaper subagents (Sonnet, Haiku) through the Agent tool's `model` parameter. The goal is the same quality for fewer tokens.

If the whole task is small (a few tool calls, one file, one quick answer), skip this skill's machinery and just do it. Delegation has overhead.

If the user names a model or says "do it yourself", follow that. If the Agent tool cannot pick a model per subagent, say so once and continue without delegating.

## 1. Triage first
Before any work, break the task into units and give each an owner. Keep it to a short list (task, owner, why). Use the task list widget if available.

**You (the overseer) keep:**
- Planning, architecture, root-cause diagnosis, trade-offs, ambiguous requirements.
- Anything where a mistake is costly, hard to notice, or hard to undo: security, auth, payments, safety or crisis logic, data models and migrations, database or access rules, secrets, sync and concurrency.
- Wording where voice and judgement matter (user-facing copy, important messages).
- Git commits and pushes, destructive commands, deployments, spending money.
- Final review and the final answer.

**Sonnet gets:** well-specified implementation and drafting. A component or screen built from your spec, a function with clear inputs and outputs, tests, a mechanical-but-multi-file refactor, drafting batches of content, summarising several files, a first pass at docs.

**Haiku gets:** mechanical work. grep-style inventories and call-site lists, finding where something lives, running tests, typecheck or lint and reporting only failures, formatting, renames, boilerplate, counting, checklists, extracting facts from files, checking items against a rule.

**Delegation test.** Delegate only if all three are true: you can state the acceptance criteria in a few lines; a wrong result would be visible on review or by a check; and it is cheap to redo. If the brief would be longer than the work, do the work yourself.

## 2. Write a proper brief
Subagents start with no context. Every brief has:
- Goal, in one or two sentences, and why it matters.
- Exact file paths or search scope (pass paths, not pasted file contents).
- What not to touch.
- Constraints and conventions to follow (stack, style, naming).
- Acceptance criteria and how to check them.
- Output format: results only, short, list files changed, flag anything uncertain or skipped. No recap of the steps.

If the `caveman` skill is available and the user uses it, ask for terse reports.

## 3. Run them well
- Independent tasks go out together in one message (parallel). Dependent tasks run in order.
- Never let two agents edit the same file at once. Split by file or by batch.
- Read-only research agents are safe to run wide. Write agents get narrow scopes.
- Do your own hard work while subagents run.

## 4. Verify, never trust
A subagent's "done" is a claim, not evidence.
- Read the diff or output yourself before accepting it.
- Run the typecheck or tests (or have a Haiku agent run them and return only failures).
- Spot-check batch work: all high-risk items plus a random sample (about 10 percent).
- Escalate on failure: Haiku fails, retry once with Sonnet with a sharper brief. Sonnet fails twice, do it yourself.
- Reject over-built or off-style work. Do not let scope creep through.

## 5. Keep tokens low
- Do not re-read what a subagent already summarised unless you are verifying it.
- Ask for compact outputs, not transcripts.
- Keep your own context for judgement. Large searches, long logs and bulk files go to subagents.
- Do not spawn an agent for a single grep or one-line edit.

## 6. Final report (short, plain prose)
Say what was done, which parts went to which model, anything you redid and why, what you verified and how, and what could not be verified. Do not recap every step.

## Anti-patterns
- Delegating the hard part to save tokens, then paying twice to fix it.
- Vague briefs ("fix the auth").
- Accepting subagent output unread.
- Parallel agents colliding on the same file.
- Handing subagents commits, secrets, rules files or safety logic.
- Delegating everything, including trivial edits, so overhead exceeds the savings.

## Example split
Task: add a settings toggle, sync it to the backend, update the rules, add tests.
- Haiku: list every place settings are read and written.
- You: design the data shape, sync change and access-rule change.
- Sonnet: build the toggle UI from your spec, and write the tests.
- Haiku: run typecheck and tests, report failures only.
- You: review all diffs, fix issues, commit.
