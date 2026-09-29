"""The answer log (every answer served) and user feedback on those answers."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from lexeu.generation.answer import Answer
from lexeu.infra.db import AnswerRow, FeedbackRow


class AnswerLog:
    def __init__(self, engine: AsyncEngine) -> None:
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    async def record(
        self,
        answer_id: str,
        answer: Answer,
        retrieval: str,
        api_key_id: int | None = None,
        cache_hit: bool = False,
    ) -> None:
        row = AnswerRow(
            id=answer_id,
            api_key_id=api_key_id,
            question=answer.question,
            lang=answer.lang,
            answer=answer.text,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            citations=[
                {"n": c.n, "provision_key": c.provision_key, "citation": c.citation}
                for c in answer.citations
            ],
            model=answer.model,
            prompt_version=answer.prompt_version,
            retrieval=retrieval,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cost_usd=answer.cost_usd,
            latency_ms=answer.timings_ms,
            cache_hit=cache_hit,
        )
        async with self._session.begin() as s:
            s.add(row)

    async def feedback(self, answer_id: str, rating: int, comment: str | None) -> bool:
        """Store (or replace) the feedback on an answer. False if the answer does not exist."""
        async with self._session.begin() as s:
            if await s.scalar(select(AnswerRow.id).where(AnswerRow.id == answer_id)) is None:
                return False
            stmt = insert(FeedbackRow).values(answer_id=answer_id, rating=rating, comment=comment)
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=[FeedbackRow.answer_id],
                    set_={"rating": rating, "comment": comment},
                )
            )
        return True
