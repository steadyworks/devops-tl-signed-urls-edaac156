import os
from pathlib import Path
from typing import cast

from google import genai
from google.genai import types
from google.genai.client import AsyncClient

from backend.lib.utils.common import none_throws


class Gemini:
    def __init__(self) -> None:
        self.__client = genai.Client(
            vertexai=True,
            project=none_throws(os.getenv("GOOGLE_VERTEX_AI_PROJECT")),
            location="global",
        )
        self.model = "gemini-2.5-flash-lite-preview-06-17"

    def get_client(self) -> AsyncClient:
        return self.__client.aio

    def build_gemini_content_from_image_understanding_job(
        self, user_instruction: str, image_paths: list[Path]
    ) -> list[types.Content]:
        parts: list[types.Part] = []

        # Build structured prompt content with image parts
        parts.append(types.Part.from_text(text="<request>\n<photos>\n"))

        for idx, path in enumerate(image_paths):
            print(idx, path)
            with open(path, "rb") as f:
                raw_bytes = f.read()

            # FIXME: hardecoded mime
            image_part = types.Part.from_bytes(data=raw_bytes, mime_type="image/png")

            parts.append(types.Part.from_text(text=f"<photo><id>{idx}</id><img>"))
            parts.append(image_part)
            parts.append(types.Part.from_text(text="</img></photo>\n"))

        parts.append(types.Part.from_text(text="</photos>\n<instruction>\n"))
        parts.append(types.Part.from_text(text=user_instruction))
        parts.append(types.Part.from_text(text="\n</instruction>\n</request>"))

        return [types.Content(role="user", parts=parts)]

    def build_gemini_config_from_image_understanding_job(
        self,
    ) -> types.GenerateContentConfig:
        sys_prompt = """The user will give you a structured XML like request that specifies some photos (n = 1 - 100) and their metadata, as well as some instructions, such as
<request>
  <photos>
  <photo><id>1</id><img>[image bytes]</img></photo>
  <photo><id>2</id><img>[image bytes]</img></photo>
  <photo><id>3</id><img>[image bytes]</img></photo>
  </photos>
  <instruction>
    I'm creating a photo book to celebrate a memory with my girlfriend. 
  </instruction>
</request>

With the request, the user is trying to create a photobook. Use all that you can infer from the uploaded photos and do the following.
    1. Group the photos into pages. Each page can have 1-6 photos.  You should group by subject, location, time, or anything you see fit. Each page should have a meaningful and coherent theme.
    2. For each page, optionally write a message in 1-3 sentences to celebrate the occasion identified by the photos you chose on that page if you see fit. Tone: Casual, celebratory, romantic, don't use words too fancy; Remember: The message should sound super natural as if the user is trying to convey the message to the photobook viewer. 

To recap, your job is to understand the user instructions, identify the grouping and return an XML in the following example format:
<response>
<page>
    <photo><id>1</id></photo>
    <photo><id>2</id></photo>
    <img>[page_message]</img>
</page>

<page>
    <photo><id>3</id></photo>
    <photo><id>5</id></photo>
    <img>[page_message]</img>
</page>
</response>"""

        return types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=65535,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ],
            system_instruction=[types.Part.from_text(text=sys_prompt)],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    async def run_image_understanding_job(
        self, user_instruction: str, image_paths: list[Path]
    ) -> str:
        contents = self.build_gemini_content_from_image_understanding_job(
            user_instruction, image_paths
        )
        config = self.build_gemini_config_from_image_understanding_job()

        # Stream and collect output
        chunks = await self.get_client().models.generate_content_stream(
            model=self.model,
            contents=cast("types.ContentListUnion", contents),
            config=config,
        )
        response_text = ""
        async for chunk in chunks:
            if chunk.text is not None:
                response_text += chunk.text
        return response_text
