# Before you publish

Delete this file once you have worked through it. It is a handover note, not part of the course.

## 1. Placeholders you must replace

Search the repo for these and fill them in:

| Where | Placeholder | Replace with |
|---|---|---|
| `LICENSE` | `<YOUR NAME>` | your name |
| `LICENSE-DOCS` | `<YOUR NAME>` | your name |
| `README.md` | `YOUR-USERNAME` in the clone URL | your GitHub handle |
| `README.md` | `[link to the booklet on your site]` | the booklet URL on your site |
| Booklet, near the top | `[link to your GitHub repo]` | the repo URL |
| Booklet, Appendix E | the reporting sentence | your issues URL or contact |

```bash
grep -rn "YOUR NAME\|YOUR-USERNAME\|link to the booklet\|link to your GitHub" . --exclude-dir=.git
```

## 2. Decisions only you can make

**Licensing.** The repo ships MIT for code and CC BY 4.0 for prose. Both are permissive defaults, not recommendations. If you want to restrict commercial reuse of the writing, swap `LICENSE-DOCS` for CC BY-NC 4.0. Note that CC licenses are not designed for source code, which is why the two are separate.

**Attribution for AI assistance.** This material was drafted in collaboration with an AI assistant, and the code was executed and measured rather than generated blind. Whether to say so is your call. If your site or your institution has a disclosure policy, it applies here. A one-line note in the README costs nothing and pre-empts the question.

**Your name on it.** The booklet currently has no author line. Add one, and if you want the academic credit, consider a `CITATION.cff` file so the repo is citable:

```yaml
cff-version: 1.2.0
title: "Build, Train, Deploy: An LLM Booklet"
authors:
  - family-names: <SURNAME>
    given-names: <GIVEN NAME>
message: "If you use this material, please cite it."
url: "https://github.com/YOUR-USERNAME/llm_booklet"
date-released: 2026-08-06
```

## 3. Publishing the booklet on your site

The booklet is a single Markdown file, about 3,200 lines. Points to watch:

- **Table of contents anchors.** The links use GitHub-flavoured anchor slugs (`#week-1-build-a-transformer-...`). Most static site generators produce the same slugs, but not all. Check every TOC link renders and jumps correctly after conversion, or replace them with your generator's anchor syntax.
- **Code block languages.** Blocks are tagged `python`, `bash`, and `dockerfile`. Confirm your highlighter knows all three.
- **Long tables.** Several tables have four or five columns. Check them on a phone; add horizontal scrolling if your theme does not.
- **Splitting into pages.** If your site prefers one page per chapter, split on the `# Week N` headings. Keep the appendices together, and keep the front matter as the landing page, since it carries the ordering and the size rule that make the rest make sense.

## 4. Before the first push

```bash
python verify.py                    # must print 16/16
git status --short                  # confirm no *.pt, *.gguf, or input.txt staged

# Measure what git would actually publish, not the working tree. `du -sh .` counts
# ignored files too, so a local venv/ or a downloaded corpus makes it look enormous.
git ls-files -z | du -ch --files0-from=- | tail -1     # should be well under 1 MB
git count-objects -vH | grep size-pack                 # the packed repository size
```

The `.gitignore` already excludes checkpoints, downloaded corpora, and virtualenvs. Verify nothing large slipped in before the first commit, because removing a large file from git history afterwards is unpleasant.

Enable GitHub Actions so `.github/workflows/verify.yml` runs. A green badge on a repo that claims specific numbers is worth more than the numbers.

Add the badge to the top of `README.md`:

```markdown
[![verify](https://github.com/YOUR-USERNAME/llm_booklet/actions/workflows/verify.yml/badge.svg)](https://github.com/YOUR-USERNAME/llm_booklet/actions/workflows/verify.yml)
```

## 5. After publishing

**Set an expectation about drift.** The README says "last verified August 2026". This stack breaks tutorials on a monthly cadence, so either re-run `verify.py` periodically and update that date, or add a line saying the material is a snapshot and pointing at the official docs. An honest stale-date beats a silently rotting document.

**Expect the first issues to be about weeks 3 to 5.** Those chapters were not executed on the machine the booklet was written on, and Appendix E says so. The most useful contribution anyone can make is running them and reporting real numbers. The README's contributing section already asks for this.

**One thing worth doing yourself first.** Run week 3 end to end on your own GPU and add the numbers to the README's reference table. It closes the one honest gap in the document, and it takes an afternoon.
