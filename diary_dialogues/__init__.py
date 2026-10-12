"""Diary Dialogues: AI-simulated personalities x AI support models.

simulator  the steered diarist (Qwen2.5-7B-Instruct + schema vectors, Zhang et al. 2026)
responder  the response models that reply to the diaries
harness    the diarist <-> responder conversation loop
judge      scoring: schema expression, coherence, and MentalHealthBench rubrics
"""
