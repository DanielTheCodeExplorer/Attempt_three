# Workflow Evolution Story

## Purpose
This note explains how the competency-scoring workflow changed over the last day of implementation. It is written as a clear narrative for the project report so the design decisions can be justified step by step.

## 1. Starting Point: Job Description as the Main Evidence Source
The starting workflow used the employee `description` field from the main dataset as the primary source of evidence. The original intention was straightforward:

- take each employee's job-description text,
- compare it against the skills they said they had,
- produce a competency-related score.

At this stage, the system treated the job description as the main textual representation of the employee's experience. This was a reasonable first step because the field was already present in the structured dataset and easy to process.

However, the output quickly showed a limitation. Many job descriptions were broad, generic, and operational. They often described responsibilities in a business context, but not in the explicit technical language needed for strong skill-level evidence. As a result, the scores were consistently low and difficult to interpret.

## 2. First Adjustment: Move Toward Biography-Based Evidence
Because the job-history descriptions were too generic, the next design move was to inspect the biography data. The biographies often contained richer, more natural descriptions of the employee's background and experience.

The reasoning was:

- biographies might contain more complete context,
- biographies might mention the kind of work people actually did,
- biographies might provide better skill signals than short operational descriptions.

This led to the creation of a separate biography pipeline. Instead of relying only on the job-description dataset, the system now also created a biography-based dataset and produced a separate score output from it.

This shift improved evidence coverage, but it did not solve the entire problem. The biographies were often still broad and professional in tone rather than explicitly skill-focused. In other words, the text was richer, but not yet structured in a way that made scoring easy or reliable.

## 3. Intermediate Problem: Raw Text Was Still Too Diffuse
Once the pipeline used both job history and biographies, a new issue became clearer. Even when the text was richer, it still tended to mix several kinds of information together:

- professional background,
- general business responsibilities,
- industries worked in,
- broad strengths,
- occasional technical evidence.

This created two problems:

1. the text was too diffuse to support clean skill-level comparison,
2. generic profile phrases could be mistaken for technical evidence.

The result was that the pipeline still struggled to distinguish between:

- a declared skill that was truly evidenced in the text,
- and a declared skill that was only loosely compatible with the person's general background.

That made it necessary to introduce a filtering and structuring stage before final scoring.

## 4. LLM Normalization Stage: Turn Broad Text into Structured Skill Evidence
The next major change was to introduce an LLM-based normalization layer. The purpose of this stage was not to invent better employee profiles. The purpose was to transform raw biography and job-history text into a more controlled, auditable evidence format.

The normalizer was designed to return:

- a short standardized summary,
- a list of matched skills,
- evidence phrases and sentences for each matched skill,
- a cleaned evidence text field,
- an evidence-strength label for each matched skill.

This was a major improvement because it introduced structure. Instead of treating the whole biography or job-history block as one undifferentiated input, the pipeline could now ask:

- which skills are actually supported,
- where is the supporting evidence,
- and how strong is that evidence?

This made the audit trail much clearer and reduced the amount of irrelevant text flowing into the scorer.

## 5. Taxonomy and TF-IDF Reintroduced a Controlled Skill Space
At an intermediate stage, the workflow moved too far toward extraction-only scoring. That made the pipeline easier to interpret, but it removed the original TF-IDF idea from the active score.

To correct that, the next design step reintroduced:

- a canonical skill taxonomy,
- skill aliases and relationships,
- TF-IDF scoring against predefined skill profiles.

This preserved the original methodological idea that competency scoring should still include a text-similarity component, while avoiding the earlier problem of comparing raw noisy text directly to ideal reference answers.

## 6. Final Design Decision: Combine TF-IDF and LLM Confidence
The final design question became:

- how can the system preserve TF-IDF similarity scoring,
- while also using the LLM to judge how strongly the evidence supports a skill?

The answer was to separate the roles:

- TF-IDF provides the source-level text similarity signal,
- the LLM provides the source-level confidence signal,
- the final competency score multiplies both together.

This gave the project a clearer and more defensible combined methodology.

## 7. Current Workflow: Two Evidence Sources, One Final Dataset
The current implementation now works as follows:

1. load employee-level job-history records,
2. load employee-level biography records,
3. keep the canonical skill list fixed,
4. score biography text against each skill profile with TF-IDF,
5. score job-history text against each skill profile with TF-IDF,
6. use the LLM normalization layer to assign confidence by source,
7. calculate source-level competency scores,
8. sum the competency scores,
9. rank the skills within each employee,
10. write one final consolidated dataset.

The score now answers a clearer question than the earlier versions.

It means:

- how strongly does this source text align with the target skill profile,
- and how confident is the evidence supporting that skill?

## 8. Final Dataset Logic
The final dataset keeps one row per:

- `talentlinkId + skill`

It includes:

- source text columns,
- biography TF-IDF score,
- job-description TF-IDF score,
- biography confidence score,
- job-description confidence score,
- averaged confidence helper score,
- summed TF-IDF helper score,
- biography competency score,
- job-description competency score,
- summed competency score,
- competency-based rank.

The active formulas are:

- `biography_competency_score = biography_tfidf_score * biography_confidence_score * 100`
- `job_description_competency_score = job_description_tfidf_score * job_description_confidence_score * 100`
- `confidence_score = ((biography_confidence_score + job_description_confidence_score) / 2) * 100`
- `sum_tfidf_score = biography_tfidf_score + job_description_tfidf_score`
- `sum_competency_score = biography_competency_score + job_description_competency_score`

Skills are then ranked within each employee using:

- `competency_score_rank`

If `sum_competency_score == 0`, then:

- `competency_score_rank = -1`

## 9. Why the Workflow Changed Repeatedly
The workflow changed several times because each stage exposed a more precise version of the real problem.

The sequence was:

- job descriptions were available but too generic,
- biographies were richer but still too diffuse,
- raw text scoring was too noisy,
- extraction-first scoring was clearer but removed TF-IDF from the active score,
- taxonomy and TF-IDF restored structured skill comparison,
- LLM confidence preserved evidence-strength judgement,
- the final dataset combined both into one consolidated output.

So the changes were not random. They were a progression toward a workflow that better matched the actual properties of the data and the methodological aims of the project.

## 10. Academic Interpretation
In methodology terms, the implementation evolved from a broad text-comparison pipeline into a controlled hybrid evidence-scoring pipeline.

The key methodological shift was:

- from comparing raw employee text directly to ideal skill descriptions,

to:

- structuring the evidence with an LLM,
- scoring text similarity with TF-IDF,
- weighting that similarity with source-level confidence,
- and combining both evidence sources in one final dataset.

This matters because it improves:

- interpretability,
- auditability,
- alignment with sparse real-world employee data,
- preservation of the original TF-IDF scoring idea,
- clearer final ranking of employee-skill evidence.

## 11. Final Position
The final workflow is not identical to the earliest benchmark-comparison idea, but it is a better fit for the data available in this project.

It preserves the core aim of competency scoring while changing the mechanism to something more realistic:

- do not rely on one raw text source,
- do not let the LLM fully replace the score,
- use biography and job history separately,
- use TF-IDF for textual alignment,
- use the LLM for evidence confidence,
- combine both into a final summed competency score and rank.

That is the current state of the implementation.
