# Project: <Name>
<!-- One-line description of what this project does -->

## Model Routing

| Task type                                      | Model      |
|------------------------------------------------|------------|
| Typo, rename, single-file edit, quick question | haiku      |
| Feature work, tests, refactor, docs            | sonnet     |
| Architecture, multi-file debug, multi-agent    | opus       |
| Plan (Opus) + execute (Sonnet) hybrid          | opusplan   |

*Default:* sonnet · Escalate to opus only when complexity justifies it.

## Stack
<!-- Fill in your actual stack -->
- Language: <!-- e.g. TypeScript 5.x, Python 3.12 -->
- Framework: <!-- e.g. Next.js 14, FastAPI -->
- Key dependencies: <!-- e.g. Prisma, Zod, Tailwind -->

## Commands
<!-- Update these to match your actual scripts -->
- `npm run dev` — start dev server
- `npm test` — run tests
- `npm run lint` — lint
- `npm run build` — production build

## Directory Structure
<!-- Update to reflect your actual layout -->
- `src/` — source code
- `src/api/` — route handlers
- `src/components/` — UI components
- `src/lib/` — shared utilities

## Conventions
<!-- Keep, remove, or add rules that apply to this repo -->
- File naming: kebab-case
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`)
- Type annotations required on all functions
- No default exports — use named exports only

## Workflow
- For new features: explore → plan → implement → test → commit
- Run tests and lint before committing
- Ask before making architectural changes
- Ask before installing new dependencies
- Do not push to remote unless explicitly asked

## Claude Behavior
- Prefer editing existing files over creating new ones
- Do not add comments or docstrings to code you did not change
- Do not over-engineer — build only what is asked
- Do not add error handling for impossible cases
- Never expose secrets, API keys, or credentials in code or logs

## Gotchas
<!-- Add project-specific pitfalls here -->
- <!-- e.g. "The auth module has a retry loop — don't change the timeout" -->
- <!-- e.g. "config.json is auto-generated — edit config.template.json instead" -->
