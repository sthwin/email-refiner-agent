from google.adk.agents import Agent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from .prompt import (
    EMAIL_OPTIMIZER_DESCRIPTION,
    TONE_STYLIST_DESCRIPTION,
    CLARITY_EDITOR_DESCRIPTION,
    LITERARY_CRITIC_DESCRIPTION,
    EMAIL_SYNTHESIZER_DESCRIPTION,
    PERSUASION_STRATEGIST_DESCRIPTION,
    TONE_STYLIST_INSTRUCTION,
    CLARITY_EDITOR_INSTRUCTION,
    LITERARY_CRITIC_INSTRUCTION,
    EMAIL_SYNTHESIZER_INSTRUCTION,
    PERSUASION_STRATEGIST_INSTRUCTION,
)

MODEL = LiteLlm(model="openai/gpt-4o")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
    output_key="clarity_output",
    model=MODEL,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
    output_key="tone_output",
    model=MODEL,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
    output_key="persuasion_output",
    model=MODEL,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
    output_key="synthesized_output",
    model=MODEL,
)


# LoopAgent의 내부에서 escalate플래그를 True로 설정하면 루프를 탈출함
# 만약에 상위에 분류 에이전트가 있고, 하위 에이전트가 많이 있는 계층적인 구조에서 escalate플래그를 True로 설정하면
# 상위 에이전트로 제어가 넘어감. 또한 하위 에이전트 내에 또다른 에이진트가 있고 거기서 escalate플래그를 True로 설정하면
# 그 상위(한단계 위) 에이전트로 제어가 넘어감.
def escalate_email_complete(tool_context: ToolContext):
    """Use this tool only when the email is good to go"""
    tool_context.actions.escalate = True
    return "Email optimization complete."


literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
    tools=[escalate_email_complete],
    model=MODEL,
)

# LoopAgent는 sub_agents에 추가된 순서대로 에이전트를 실행시킴
email_refiner_agent = LoopAgent(
    name="EmailRefinerAgent",
    max_iterations=50,
    description=EMAIL_OPTIMIZER_DESCRIPTION,
    sub_agents=[
        clarity_agent,
        tone_stylist_agent,
        persuation_agent,
        email_synthesizer_agent,
        literary_critic_agent,
    ],
)

root_agent = email_refiner_agent
