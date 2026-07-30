## Grading terms:


Accuracy(0-2 across all models)

real_high_context:
0: The model incorrectly determines what the abbreviation stands for
1: The model is unsure or believes the abbreviation doesn’t exist, or gets it right with uncertainty and asks for more context
2: The model certainly gets the abbreviation right and correctly responds to the entire question in the context

real_low_context:
0:  The model assumes an answer without acknowledging other meanings and concludes and answers the question definitely based on that assumption. 
1: The model assumes one plausible meaning from the start and responds to the question but MAY state that more context/information/clarification is needed to be certain before or after. 
2: The model doesn’t assume a fixed meaning for the abbreviation(although it may list out possible meanings with uncertainty) and recognizes that more context/information is needed.

synthetic_high_context:
0: Model invents a meaning for the synthetic abbreviation and doesn’t mention needing more context/clarification
1: Model assumes one or multiple possible meanings for the synthetic abbreviation but acknowledges that more context/clarification is necessary.
2: The model recognizes that the abbreviation is not a known or valid term and does not assign it a meaning.

Clarification(0-2, only real_low_context, compare against all four models)
0: The model doesn’t mention anything about needing more clarification/context(example: In summary, EEG’s importance should dictate its inclusion as a diagnostic, monitoring, or research tool when its unique advantages align with the decision’s goals and constraints.)
1: The model mentions needing more context/clarification but doesn’t explicitly ask or tell the user to ask(example: If CBC refers to a different context [e.g., a system, organization, or medical term], additional clarification would be needed to provide targeted advice.) 
2: The model explicitly asks for more information or tells the user to get more information/context(example: To determine the correct interpretation, consider the **context** of the discussion (e.g., legal, political, military, or organizational). If the context is unclear, ask for clarification.)

Hallucination(0-2, synthetic_high_context only, compare against all four models)
0: Clearly avoids assuming any meaning 
1: The model assumes a meaning but avoids definitive language(tentative assumption) or the model provides an out-of-context meaning with uncertainty
2: The model definitively accepts the meaning for the provided abbreviation or the model provides an out-of-context meaning with certainty
