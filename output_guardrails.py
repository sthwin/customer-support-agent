from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail, Runner
from numpy.strings import str_len
from models import TechnicalOutputGuardRailOutput, UserAccountContext

technical_output_guardrail_agent = Agent(
    name="Technical Support Guardrail",
    instructions="""
    Analyze the technical support response to check if it inappropriately contains:
    
    - Billing information (payments, refunds, charges, subscriptions)
    - Order information (shipping, tracking, delivery, returns)
    - Account management info (passwords, email changes, account settings)
    
    Technical agents should ONLY provide technical troubleshooting, diagnostics, and product support.
    Return true for any field that contains inappropriate content for a technical support response.
    """,
    output_type=TechnicalOutputGuardRailOutput,
)

@output_guardrail
async def technical_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
    output: str,
):
    result = await Runner.run(
        starting_agent=technical_output_guardrail_agent,
        input=output,
        context=wrapper.context,
    )
    
    validation = result.final_output
    
    triggered = (
        validation.contains_off_topic
        or validation.contains_billing_data
        or validation.contains_account_data
    )
    
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )
