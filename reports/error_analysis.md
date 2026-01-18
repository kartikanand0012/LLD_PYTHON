# Error analysis

- False positives: 3
- False negatives: 2

## Category counts
- empty_context: 1
- low_overlap: 1
- negation: 1
- numeric: 3
- uncategorized: 1
## False positives
- Example
  - Response: Support lasts 12 months.
  - Context: The plan includes 12 months of support.
  - Expected: non-hallucination
  - Predicted: hallucination
- Example
  - Response: It is not specified in the context.
  - Context: N/A
  - Expected: non-hallucination
  - Predicted: hallucination
- Example
  - Response: Orders over $40 ship free.
  - Context: Shipping is free for orders over $40.
  - Expected: non-hallucination
  - Predicted: hallucination
## False negatives
- Example
  - Response: The email was sent on January 18.
  - Context: The email was sent on January 8.
  - Expected: hallucination
  - Predicted: non-hallucination
- Example
  - Response: The release added /users, /teams, and /billing.
  - Context: The release adds two endpoints: /users and /teams.
  - Expected: hallucination
  - Predicted: non-hallucination
