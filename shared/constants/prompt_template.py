USER_PROMPT_TEMPLATE = """Task Description:
{%- if task_description %}
{{ task_description | trim }}
{%- endif %}

{%- if criterion_description %}
Criterion Description:
{{ criterion_description | trim }}
{%- endif %}

{%- if rubric %}

Rubric:

{%- for rubric_item in rubric %}

Score {{ rubric_item.get_rubric_item_score() }}: {{ rubric_item.get_rubric_item_label() }}
{{ rubric_item.get_rubric_item_description() }}
{%- if rubric_item.get_rubric_item_examples() %}

Examples:
{%- for example in rubric_item.get_rubric_item_examples() %}
{{ example }}
{%- endfor %}
{%- endif %}
{%- endfor %}
{%- endif %}

{%- if context %}

Context:
{{ context | trim }}
{%- endif %}

{%- if response_format %}

Response Format:
{{ response_format | trim }}
{%- endif %}
{%- if disagreement_context %}

{{ disagreement_context | trim }}
{%- endif %}
"""

