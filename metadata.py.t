import os
import uuid
import logging
import time
from typing import Optional

import openai
import mlflow
from tenacity import retry, stop_after_attempt, wait_exponential

from schemas import JiraAnalysis
from llm_client import get_enterprise_openai_client
from token_jwt import get_token
from config_loader import get_prompts, get_config


prompts = get_prompts()
config = get_config()

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def extract_jira_metadata(jira_description: str) -> Optional[JiraAnalysis]:

    if len(jira_description) > 15000:
        logger.error(f"Input too long ({len(jira_description)} chars). Dropping request.")
        return None

    client = get_enterprise_openai_client()
    jwt = get_token()

    request_headers = {
        "X-HSBC-E2E-Trust-Token": jwt,
        "x-correlation-id": str(uuid.uuid4()),
        "x-usersession-id": str(uuid.uuid4())
    }

    try:

        start_time = time.time()

        completion = client.beta.chat.completions.parse(
            model=os.getenv("ENTERPRISE_MODEL_NAME", config["model_name"]),
            messages=[
                {"role": "system", "content": prompts[PROMPT_VERSION]["main_prompt"]},
                {"role": "user", "content": jira_description}
            ],
            temperature=0.0,
            response_format=JiraAnalysis,
            user=config["user_id"],
            extra_headers=request_headers
        )

        latency = time.time() - start_time
        mlflow.log_metric("latency_sec", latency)

        usage = completion.usage
        if usage:

            logger.info(
                "Token usage | sent(prompt)=%s, received(completion)=%s, total=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

            mlflow.log_metric("prompt_tokens", usage.prompt_tokens)
            mlflow.log_metric("completion_tokens", usage.completion_tokens)
            mlflow.log_metric("total_tokens", usage.total_tokens)

        else:
            logger.warning("No token usage returned by API.")

        if completion.choices[0].message.refusal:
            logger.warning(f"Model refusal: {completion.choices[0].message.refusal}")
            return None

        parsed = completion.choices[0].message.parsed

        parsed.agile_standard = (
            parsed.who.identified
            and parsed.what.identified
            and parsed.why.identified
            and parsed.ac_defined.presence_ac
        )

        mlflow.log_metric("agile_standard", int(parsed.agile_standard))

        return parsed

    except openai.LengthFinishReasonError:
        logger.error("Token limit exceeded during generation.")
        return None

    except openai.APIStatusError as e:
        logger.error(f"Enterprise API Error: {e.status_code} - {e.response.text}")
        raise

    except openai.APIConnectionError as e:
        logger.error(f"Network routing/SSL error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected Exception: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    test_description = (
        """We want to add a new column in the Project class,
        so that the application can store and manage additional data
        related to the project. (Delivery Yes/No)."""
    )

    prompt_text = prompts[PROMPT_VERSION]["main_prompt"]

    with mlflow.start_run(run_name=f"{PROMPT_VERSION}_{config['model_name']}"):

        mlflow.log_param("model", config["model_name"])
        mlflow.log_param("prompt_version", PROMPT_VERSION)

        mlflow.log_text(prompt_text, "prompt.txt")

        result = extract_jira_metadata(test_description)

        if hasattr(result, "model_dump_json"):

            output_json = result.model_dump_json(indent=2)

            mlflow.log_text(output_json, "output.json")

            print(output_json)
