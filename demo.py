import os

# Force the Hugging Face or OpenAI usage. 
# If using Hugging Face, set these:
os.environ["SWARM_CLIENT_BACKEND"] = "huggingface"
os.environ["HF_API_TOKEN"] = ""
os.environ["HF_DEFAULT_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"  # or your model

from swarm import Swarm, Agent

# =========================
# 1. Utility Functions
# =========================
def create_shipping_label(order_id: str):
    print(f">>> Creating shipping label for order ID: {order_id}")
    return f"Shipping label created for order {order_id}"

def issue_refund(order_id: str):
    print(f">>> Refunding order ID: {order_id}")
    return f"Refund completed for order {order_id}"

def transfer_to_shipping_agent():
    return shipping_agent

def transfer_to_refund_agent():
    return refund_agent


# =========================
# 2. Agents
# =========================

# --- Shipping Agent ---

shipping_agent = Agent(
    name="Shipping Agent",
    instructions="""
You handle shipping and delivery issues.
If you need to create a shipping label, do so only once. 
After creating the shipping label, finalize with:
{
  "name": "none",
  "arguments": {}
}
Then reply in nature language. Never loop calling.
""",
    functions=[create_shipping_label],
)


# --- Refund Agent ---
refund_agent = Agent(
    name="Refund Agent",
    instructions="""
You handle refund and return requests.
You can call `issue_refund(order_id: str)` to process a refund.
If you need to create a issue_refund, do so only once. 
After creating the issue_refund, finalize with
{
  "name": "none",
  "arguments": {}
}
Then reply in nature language. Never loop calling.
""",
    functions=[issue_refund],
)

# --- Main Agent (the key) ---
main_agent = Agent(
    name="Main Customer Service Agent",
    # This is the critical "system instructions" to force a JSON tool call:
    instructions="""
You are the primary customer service agent for an e-commerce store. You can use transfer_to_shipping_agent and transfer_to_refund_agent. 

IMPORTANT: 
If you want to transfer, You can only respond in valid JSON. Never respond in plain text.
If you do any transfer, only do it once. Neve stack in a loop. 
After finish one transter, responde with nature language. Never loop calling.
""",
    functions=[transfer_to_shipping_agent, transfer_to_refund_agent]
)

# =========================
# 3. Initialize Swarm
# =========================
huggingface_client = Swarm()  # picks up from environment

# =========================
# 4. Conversation Loop
# =========================
user_messages = [
    "Hello, I'd like to know if my order has shipped.",
    "Actually, I'd like to return it. My order number is 54321.",
    "No shipping or refund question here, just hello!"
]

history_messages = []
for user_text in user_messages:
    print(f"\nUser: {user_text}")
    history_messages.append({"role": "user", "content": user_text})

    response = huggingface_client.run(
        agent=main_agent,
        messages=history_messages,
        debug=True
    )

    # The response is from whichever agent ended up being active.
    last_msg = response.messages[-1]["content"]
    agent_name = response.agent.name

    print(f"{agent_name} says: {last_msg}")
    history_messages.append({"role": "assistant", "content": last_msg})
