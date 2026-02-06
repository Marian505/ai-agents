import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool, ToolRuntime
import httpx 
import base64
from rich.pretty import pprint  # noqa: F401
import aiofiles
from dataclasses import dataclass
import os

load_dotenv()

PDF_GEN_URL = os.getenv("PDF_GEN_URL", "http://fastapi.dev.svc.cluster.local:8000/api/html-to-pdf")

fast_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
smart_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.0)

@dataclass
class ContextSchema:
    pdf_path: str | None

@tool
async def html_to_pdf(html: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """Tool for create PDF from HTML.
    Function parameter has to be just valid HTML in string.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            PDF_GEN_URL,
            content=html,
            headers={"Content-Type": "text/html"}
        ) 
        if response.status_code != 200:
            raise ValueError(f"PDF API failed: {response.status_code}")

    # TODO: no prod version, for prod store in S3
    if runtime.context.pdf_path is not None:
        pdf_path = runtime.context.pdf_path
        pdf_base64 = response.json()['pdf_base64']
        pdf_bytes = await asyncio.to_thread(base64.b64decode, pdf_base64)
        async with aiofiles.open(pdf_path, 'wb') as f:
            await f.write(pdf_bytes)
    
    return f"PDF generated, response code: {response.status_code}"

system_prompt="""
    You are CV assistant. 
    You will read and analyze user CV in PDF. 
    Generate valid HTML and generate PDF on user demand.
    For PDF generation use tool html_to_pdf on user demand.
"""

class CustomState(AgentState):
    pass

agent = create_agent(
    model=fast_model,
    tools=[html_to_pdf],
    system_prompt=system_prompt,
    context_schema=ContextSchema,
    state_schema=CustomState
)
