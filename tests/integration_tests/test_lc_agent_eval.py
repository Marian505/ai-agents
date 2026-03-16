import pytest
from langsmith import AsyncClient
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from langchain_core.messages import HumanMessage, AIMessage
from langsmith.evaluation import aevaluate

from lc_agent import agent

# wrong test gent does not contaiinweb search
# @pytest.mark.asyncio
# @pytest.mark.langsmith
# async def test_trajectory_quality():
#     evaluator = create_trajectory_llm_as_judge(
#         model="claude-sonnet-4-5-20250929",
#         prompt=TRAJECTORY_ACCURACY_PROMPT,
#     )
#     result = await agent.ainvoke({"messages": [HumanMessage(content="What's the weather in Seattle?")]})
#     evaluation = evaluator(outputs=result["messages"])
#     # print(f"evaluation: {evaluation}")
#     assert evaluation["score"] is True

@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_evaluate():
    aclient = AsyncClient()

    dataset = await aclient.read_dataset(dataset_name="my_dataset4")
    await aclient.create_example(
        inputs={"messages": [HumanMessage(content="What is Java?")]},
        outputs={"messages": [AIMessage("Java is programming language.")]},
        dataset_id=dataset.id
    )

    trajectory_evaluator = create_trajectory_llm_as_judge(
        model="claude-sonnet-4-5-20250929",
        prompt=TRAJECTORY_ACCURACY_PROMPT,
    )

    async def run_agent(inputs):
        """Your agent function that returns trajectory messages."""
        result = await agent.ainvoke(inputs)
        return result["messages"]

    experiment_results = await aevaluate(
        run_agent,
        data=dataset.id,
        evaluators=[trajectory_evaluator]
    )

    print(f"experiment_results: {experiment_results}")