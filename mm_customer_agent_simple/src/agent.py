import re
from typing import Any, Dict, List, Optional

from .config import Settings
from .customer_qa_retriever import CustomerQARetriever
from .generator import AnswerGenerator
from .image_tool import ImageUnderstandingTool
from .schema import RetrievedChunk
from .tools import SearchManualTool

FIGURE_REGEXES = [
    r"air_conditioner_\d+",
    r"air_fryer_\d+",
    r"Blower_\d+",
    r"Boat_\d+",
    r"Camera_\d+",
    r"Dish_washer_\d+",
    r"drill\d+_\d+",
    r"earphones_\d+",
    r"eReader_\d+",
    r"exercise_bikes_\d+",
    r"fax_\d+",
    r"fitness_trackers_\d+",
    r"fridge_\d+",
    r"function_keyboard_\d+",
    r"generator_\d+",
    r"hybrid_instant_camera_\d+",
    r"jetski_\d+",
    r"landline_\d+",
    r"lawn_mower_\d+",
    r"Manual\d+_\d+",
    r"multi\-use_pressure_cooker_and_air_fryer_\d+",
    r"oven_\d+",
    r"pump_\d+",
    r"rideon_motorcycle_\d+",
    r"Security_Camera_\d+",
    r"snowmobile_\d+",
    r"television\d+_\d+",
    r"thermostat_\d+",
    r"toothbrush\d+_\d+",
    r"vr_\d+",
    r"washing_machine\d+_\d+",
]
FIGURE_ID_PATTERN = "|".join(f"(?:{pattern})" for pattern in FIGURE_REGEXES)
PIC_MARK_RE = re.compile(
    rf"<PIC\s*>|\[PIC\s*\]|[<\[]PIC[:：]\s*({FIGURE_ID_PATTERN})?\s*[>\]]"
)
FIGURE_ID_RE = re.compile(rf"({FIGURE_ID_PATTERN})")
FIGURE_ID_WRAPPED_RE = re.compile(rf"[<\[]({FIGURE_ID_PATTERN})[>\]]")

ROUTE_CONFIDENCE_THRESHOLD = 0.55


class AgentTraceLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: List[str] = []

    def log(self, actor: str, action: str, detail: str = "") -> None:
        line = f"[{actor}] {action}"
        if detail:
            line += f": {self._preview(detail)}"
        self.events.append(line)
        if self.enabled:
            print(line)

    @staticmethod
    def _preview(text: str, max_chars: int = 240) -> str:
        value = " ".join(str(text).split())
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "..."


class AgentHelpers:
    @staticmethod
    def clean_question_turn(text: str) -> str:
        value = text.strip()
        if value.endswith(","):
            value = value[:-1].strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        return value.replace('""', '"').strip()

    @classmethod
    def split_question_turns(cls, question: str) -> List[str]:
        turns = []
        for line in question.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            turn = cls.clean_question_turn(line)
            if turn:
                turns.append(turn)
        if turns:
            return turns
        question = cls.clean_question_turn(question)
        return [question] if question else []

    @staticmethod
    def normalize_pic_tokens(
        answer: str, available_image_ids: List[str] | None = None
    ) -> tuple[str, List[str]]:
        available_image_ids = available_image_ids or []
        image_ids: List[str] = []
        next_image_idx = 0

        def next_available_image_id() -> str:
            nonlocal next_image_idx
            if next_image_idx >= len(available_image_ids):
                return ""
            image_id = available_image_ids[next_image_idx]
            next_image_idx += 1
            return image_id

        def replace_pic_mark(match: re.Match) -> str:
            image_id = match.group(1) or next_available_image_id()
            if not image_id:
                return ""
            image_ids.append(image_id)
            return "<PIC>"

        text = PIC_MARK_RE.sub(replace_pic_mark, answer)

        def replace_figure_id(match: re.Match) -> str:
            image_ids.append(match.group(1))
            return "<PIC>"

        text = FIGURE_ID_WRAPPED_RE.sub(replace_figure_id, text)
        text = AgentHelpers.replace_figure_ids(text, replace_figure_id)

        known_ids = [
            image_id for image_id in available_image_ids if image_id not in image_ids
        ]
        if known_ids:
            bare_id_re = re.compile(
                r"("
                + "|".join(
                    re.escape(image_id)
                    for image_id in sorted(known_ids, key=len, reverse=True)
                )
                + r")"
            )

            def replace_bare_id(match: re.Match) -> str:
                image_ids.append(match.group(1))
                return "<PIC>"

            text = AgentHelpers.replace_figure_ids(text, replace_bare_id, bare_id_re)

        return text, image_ids

    @staticmethod
    def replace_figure_ids(
        text: str, replace_func, pattern: re.Pattern = FIGURE_ID_RE
    ) -> str:
        result: List[str] = []
        pos = 0
        last_match_end = -1
        while pos < len(text):
            match = pattern.match(text, pos)
            if not match:
                result.append(text[pos])
                pos += 1
                continue

            prev_char = text[pos - 1] if pos > 0 else ""
            next_char = text[match.end()] if match.end() < len(text) else ""
            prev_ok = (
                pos == 0
                or pos == last_match_end
                or not re.match(r"[A-Za-z0-9_\-]", prev_char)
            )
            next_ok = (
                match.end() == len(text)
                or not re.match(r"[A-Za-z0-9_\-]", next_char)
                or bool(pattern.match(text, match.end()))
            )
            if prev_ok and next_ok:
                result.append(replace_func(match))
                last_match_end = match.end()
                pos = match.end()
            else:
                result.append(text[pos])
                pos += 1
        return "".join(result)

    @staticmethod
    def collect_chunk_image_ids(chunks: List[RetrievedChunk]) -> List[str]:
        image_ids: List[str] = []
        for chunk in chunks:
            for image_id in chunk.image_ids:
                if image_id not in image_ids:
                    image_ids.append(image_id)
        return image_ids

    @staticmethod
    def rule_customer_service_question(question: str, question_id: str = "") -> bool:
        try:
            if question_id and int(question_id) < 64:
                return True
        except ValueError:
            pass

        text = question.lower()
        manual_terms = [
            "如何",
            "怎么",
            "步骤",
            "操作",
            "使用",
            "安装",
            "拆卸",
            "更换",
            "调节",
            "调整",
            "启动",
            "关闭",
            "清洁",
            "连接",
            "设置",
            "充电",
            "故障排除",
            "troubleshooting",
            "steps",
            "how to",
            "what are",
            "manual",
            "snowmobile",
            "television",
            "toothbrush",
            "吹风机",
            "空调",
            "遥控器",
            "摩托艇",
            "水泵",
            "温控器",
            "vr",
            "键盘",
            "电动摩托车",
            "鼠标",
            "烤箱",
            "相机",
            "电钻",
            "冰箱",
            "牙刷",
        ]
        customer_terms = [
            "7天无理由",
            "退货",
            "换货",
            "退款",
            "运费",
            "发票",
            "物流",
            "快递",
            "包装破损",
            "投诉",
            "虚假宣传",
            "少发",
            "补发",
            "保质期",
            "临期",
            "客服",
            "售后",
            "维修服务",
            "赔偿",
            "赔付",
            "订单",
            "付款",
            "签收",
            "人工客服",
            "假货",
            "二手",
            "划痕",
            "瑕疵",
            "色差",
            "上门安装服务",
            "试用装",
            "优惠券",
            "以旧换新",
        ]

        has_customer_term = any(term in text for term in customer_terms)
        has_manual_term = any(term in text for term in manual_terms)
        if has_customer_term and not has_manual_term:
            return True
        if has_customer_term and any(
            term in text
            for term in ["收到", "购买", "你们", "商品", "订单", "退款", "投诉"]
        ):
            return True
        return False


class CustomerServiceAgent(AgentHelpers):
    def __init__(self, settings: Settings, generator: AnswerGenerator | None = None):
        self.settings = settings
        self.generator = generator or AnswerGenerator(settings)
        self._qa_retriever: Optional[CustomerQARetriever] = None

    def __del__(self):
        try:
            if self._qa_retriever is not None:
                self._qa_retriever.close()
        except Exception:
            pass

    @property
    def qa_retriever(self) -> CustomerQARetriever:
        if self._qa_retriever is None:
            self._qa_retriever = CustomerQARetriever(self.settings)
        return self._qa_retriever

    def run(
        self,
        query: str,
        return_debug: bool = False,
        trace: AgentTraceLogger | None = None,
        turn_index: int = 1,
        turn_total: int = 1,
        original_query: str = "",
    ) -> Dict:
        trace = trace or AgentTraceLogger(enabled=False)
        original_query = original_query or query
        trace.log("子agent:customer_service", "接收", f"query={query}")
        trace.log(
            "子agent:customer_service",
            "执行",
            "faq_retrieval=enabled"
            if self.settings.enable_customer_qa_retrieval
            else "faq_retrieval=disabled",
        )

        qa_result = self._try_customer_qa(
            query,
            turn_index=turn_index,
            turn_total=turn_total,
            original_query=original_query,
        )
        hit = bool(qa_result and qa_result.get("hit"))
        answer = qa_result.get("answer", "") if qa_result else ""
        source = qa_result.get("strategy", "") if qa_result else "direct_llm"
        final_reason = qa_result.get("final_reason", "") if qa_result else "faq_disabled"
        route = "customer_qa_retrieval" if hit else "direct_llm_required"
        q2q = qa_result.get("query2query", {}) if qa_result else {}
        q2a = qa_result.get("query2answer", {}) if qa_result else {}

        trace.log(
            "子agent:customer_service",
            "输出",
            f"hit={hit} source={source or 'none'} "
            f"q2q_hit={q2q.get('hit', False)} q2q_candidates={len(q2q.get('candidates', []) or [])} "
            f"q2a_hit={q2a.get('hit', False)} q2a_candidates={len(q2a.get('candidates', []) or [])} "
            f"fallback={'' if hit else 'direct_llm'}",
        )

        result = {
            "query": query,
            "question": query,
            "question_type": "customer_service",
            "route": route,
            "hit": hit,
            "answer": answer,
            "answer_source": source if hit else "direct_llm",
            "fallback": "" if hit else "direct_llm",
            "final_reason": final_reason,
            "sub_agent": "customer_service",
        }
        if return_debug and qa_result is not None:
            result["customer_qa_retrieval"] = qa_result
        return result

    def _try_customer_qa(
        self,
        query: str,
        turn_index: int,
        turn_total: int,
        original_query: str,
    ) -> Optional[Dict]:
        if not self.settings.enable_customer_qa_retrieval:
            return None
        return self.qa_retriever.answer(
            query,
            turn_index=turn_index,
            turn_total=turn_total,
            original_query=original_query,
        )


class ManualQAAgent(AgentHelpers):
    def __init__(self, settings: Settings, generator: AnswerGenerator | None = None):
        self.settings = settings
        self._search_tool: Optional[SearchManualTool] = None
        self.generator = generator or AnswerGenerator(settings)

    @property
    def search_tool(self) -> SearchManualTool:
        if self._search_tool is None:
            self._search_tool = SearchManualTool(self.settings)
        return self._search_tool

    def run(
        self,
        query: str,
        return_debug: bool = False,
        trace: AgentTraceLogger | None = None,
        round_index: int = 0,
        task_id: int | str = "",
    ) -> Dict:
        trace = trace or AgentTraceLogger(enabled=False)
        trace.log("子agent:manual", "接收", f"query={query}")
        initial_result = self._search_once(query, trace=trace, round_index=round_index)
        initial_chunks = list(initial_result.get("chunks") or [])
        all_search_chunks = list(initial_chunks)
        candidate_limit = max(self.settings.final_context_top_k * 2, 12)
        candidate_state = self.prepare_candidates(
            all_search_chunks,
            candidate_limit=candidate_limit,
            trace=trace,
            task_id=task_id,
        )
        merged_chunks = list(candidate_state["merged_chunks"])
        followup_chunks: List[RetrievedChunk] = []
        followup_queries: List[str] = []
        followup_searches: List[Dict] = []
        retrieval_plans: List[Dict] = []

        plan_context = merged_chunks[: self.settings.final_context_top_k]
        for round_idx in range(2):
            trace.log("子agent:manual", "规划", f"round={round_idx + 1}")
            plan = self.generator.plan_retrieval(query, plan_context)
            retrieval_plans.append(plan)
            trace.log(
                "子agent:manual",
                "规划结果",
                f"round={round_idx + 1} "
                f"sufficient={plan.get('evidence_sufficient')} "
                f"queries={len(plan.get('search_queries', []) or [])} "
                f"context={len(plan_context)}",
            )
            if plan.get("evidence_sufficient"):
                break

            queries = [
                item
                for item in plan.get("search_queries", [])[:3]
                if item and item not in followup_queries
            ]
            if not queries:
                break

            round_chunks: List[RetrievedChunk] = []
            for followup_query in queries:
                followup_queries.append(followup_query)
                trace.log(
                    "子agent:manual",
                    "补充检索",
                    f"round={round_idx + 1} query={followup_query}",
                )
                search_result = self._search_once(
                    followup_query,
                    trace=trace,
                    round_index=round_idx + 1,
                )
                hits = list(search_result.get("chunks") or [])
                followup_searches.append(
                    {
                        "round": round_idx + 1,
                        "query": followup_query,
                        "chunks": hits,
                    }
                )
                round_chunks.extend(hits)

            followup_chunks.extend(round_chunks)
            all_search_chunks.extend(round_chunks)
            candidate_state = self.prepare_candidates(
                all_search_chunks,
                candidate_limit=candidate_limit,
                trace=trace,
                task_id=task_id,
            )
            merged_chunks = list(candidate_state["merged_chunks"])
            plan_context = merged_chunks[: self.settings.final_context_top_k]

        candidate_state = self.prepare_candidates(
            all_search_chunks,
            candidate_limit=candidate_limit,
            trace=trace,
            task_id=task_id,
        )
        candidate_chunks = list(candidate_state["candidate_chunks"])
        trace.log(
            "子agent:manual",
            "选证",
            f"candidates={len(candidate_chunks)} max_final={self.settings.final_context_top_k}",
        )
        evidence_selection = self.generator.select_evidence(
            query,
            candidate_chunks,
            max_chunks=self.settings.final_context_top_k,
        )
        final_chunks = self._chunks_from_selection(
            candidate_chunks,
            evidence_selection,
            max_chunks=self.settings.final_context_top_k,
        )
        selected_count = len(evidence_selection.get("selected_chunk_ids", []) or [])
        trace.log(
            "子agent:manual",
            "选证结果",
            f"selected={selected_count} final_chunks={len(final_chunks)} "
            f"fallback={evidence_selection.get('fallback', False)}",
        )
        answer = self.generator.generate(query, final_chunks)
        available_image_ids = self.collect_chunk_image_ids(final_chunks)
        answer, image_ids = self.normalize_pic_tokens(answer, available_image_ids)
        trace.log(
            "子agent:manual",
            "输出",
            f"answer_source=manual_rag chunks={len(final_chunks)} image_ids={image_ids}",
        )

        return {
            "query": query,
            "question": query,
            "question_type": "manual",
            "route": "agentic_retrieval",
            "initial_query": query,
            "initial_chunks": initial_chunks,
            "retrieval_plans": retrieval_plans,
            "followup_queries": followup_queries,
            "followup_searches": followup_searches,
            "followup_chunks": followup_chunks,
            "candidate_chunks": candidate_chunks,
            "evidence_selection": evidence_selection,
            "chunks": final_chunks,
            "answer": answer,
            "image_ids": image_ids,
            "answer_source": "manual_rag",
            "sub_agent": "manual",
            "child_result": self._serialize_search_result(initial_result, return_debug),
        }

    def _search_once(
        self,
        query: str,
        trace: AgentTraceLogger,
        round_index: int = 0,
    ) -> Dict:
        trace.log(
            "子agent:manual",
            "执行",
            f"hybrid_search top_k={self.settings.rerank_top_k}",
        )
        chunks = self.search_tool.search(
            query,
            top_k=self.settings.rerank_top_k,
        )
        top_hit = self._chunk_brief(chunks[0]) if chunks else "none"
        trace.log(
            "子agent:manual",
            "输出",
            f"hits={len(chunks)} top={top_hit}",
        )

        result = {
            "query": query,
            "question": query,
            "question_type": "manual",
            "route": "manual_search",
            "round": round_index,
            "chunks": chunks,
            "sub_agent": "manual",
        }
        result["chunk_count"] = len(chunks)
        return result

    @staticmethod
    def _serialize_search_result(result: Dict, include_debug: bool) -> Dict:
        data = {
            "query": result.get("query", ""),
            "question": result.get("question", ""),
            "question_type": result.get("question_type", ""),
            "route": result.get("route", ""),
            "round": result.get("round", 0),
            "sub_agent": result.get("sub_agent", ""),
            "chunk_count": result.get("chunk_count", 0),
        }
        if include_debug:
            data["chunks"] = [
                chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                for chunk in result.get("chunks") or []
            ]
        return data

    def prepare_candidates(
        self,
        chunks: List[RetrievedChunk],
        candidate_limit: int,
        trace: AgentTraceLogger | None = None,
        task_id: int | str = "",
    ) -> Dict:
        trace = trace or AgentTraceLogger(enabled=False)
        raw_count = len(chunks)
        merged_chunks = self._merge_chunks([], chunks)
        candidate_chunks = merged_chunks[:candidate_limit]
        trace.log(
            "子agent:manual",
            "整理",
            f"task={task_id} raw_hits={raw_count} "
            f"unique_chunks={len(merged_chunks)} "
            f"candidates={len(candidate_chunks)} limit={candidate_limit}",
        )
        return {
            "raw_count": raw_count,
            "unique_count": len(merged_chunks),
            "candidate_limit": candidate_limit,
            "merged_chunks": merged_chunks,
            "candidate_chunks": candidate_chunks,
        }

    @staticmethod
    def _chunk_brief(chunk: RetrievedChunk) -> str:
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
        return f"{chunk.manual_id}/{chunk.section_path} score={score}"

    @staticmethod
    def _chunks_from_selection(
        candidate_chunks: List[RetrievedChunk],
        selection: Dict,
        max_chunks: int,
    ) -> List[RetrievedChunk]:
        by_id = {chunk.chunk_id: chunk for chunk in candidate_chunks}
        selected_chunks: List[RetrievedChunk] = []
        for chunk_id in selection.get("selected_chunk_ids", []):
            chunk = by_id.get(chunk_id)
            if chunk is not None and chunk.chunk_id not in {
                item.chunk_id for item in selected_chunks
            }:
                selected_chunks.append(chunk)
            if len(selected_chunks) >= max_chunks:
                break

        min_chunks = min(4, len(candidate_chunks), max_chunks)
        if len(selected_chunks) < min_chunks:
            selected_ids = {chunk.chunk_id for chunk in selected_chunks}
            selected_manual_ids = {chunk.manual_id for chunk in selected_chunks}
            for chunk in candidate_chunks:
                if chunk.chunk_id in selected_ids:
                    continue
                if selected_manual_ids and chunk.manual_id not in selected_manual_ids:
                    continue
                selected_chunks.append(chunk)
                selected_ids.add(chunk.chunk_id)
                if len(selected_chunks) >= min_chunks:
                    break

        return selected_chunks[:max_chunks]

    @staticmethod
    def _merge_chunks(
        existing: List[RetrievedChunk], new_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        by_id: Dict[str, RetrievedChunk] = {}
        first_seen_order: Dict[str, int] = {}

        for chunk in [*existing, *new_chunks]:
            if chunk.chunk_id not in first_seen_order:
                first_seen_order[chunk.chunk_id] = len(first_seen_order)
            current = by_id.get(chunk.chunk_id)
            if current is None or ManualQAAgent._chunk_rank_score(
                chunk
            ) > ManualQAAgent._chunk_rank_score(current):
                by_id[chunk.chunk_id] = chunk

        merged = list(by_id.values())
        merged.sort(
            key=lambda chunk: (
                ManualQAAgent._chunk_rank_score(chunk),
                -first_seen_order[chunk.chunk_id],
            ),
            reverse=True,
        )
        return merged

    @staticmethod
    def _chunk_rank_score(chunk: RetrievedChunk) -> float:
        if chunk.rerank_score is not None:
            return float(chunk.rerank_score)
        return float(chunk.score or 0.0)


class MainAgent(AgentHelpers):
    def __init__(
        self,
        settings: Settings,
        generator: AnswerGenerator | None = None,
        customer_agent: CustomerServiceAgent | None = None,
        manual_agent: ManualQAAgent | None = None,
    ):
        self.settings = settings
        self.generator = generator or AnswerGenerator(settings)
        self.customer_agent = customer_agent or CustomerServiceAgent(
            settings, self.generator
        )
        self.manual_agent = manual_agent or ManualQAAgent(settings, self.generator)
        self.image_tool = ImageUnderstandingTool(settings)

    def answer(
        self,
        question: str,
        return_debug: bool = False,
        question_id: str = "",
        flow_log: bool = True,
        image_paths: List[str] | None = None,
        image_base64: List[str] | None = None,
        rewrite_history: List[Dict[str, Any]] | None = None,
        turn_mode: str = "auto_newline",
    ) -> Dict:
        image_observation = {}
        image_paths = image_paths or []
        image_base64 = image_base64 or []
        trace = AgentTraceLogger(enabled=flow_log)
        original_question = question
        image_rewrite: Dict[str, Any] = {}
        actionability: Dict[str, Any] = {}
        history_rewritten_query = ""
        final_query = question
        if image_paths or image_base64:
            image_observation = self.image_tool.observe(
                image_paths=image_paths,
                image_base64=image_base64,
                user_question=question,
            )
            image_rewrite = self.generator.rewrite_query_with_image_context(
                question,
                image_observation,
            )
            final_query = str(image_rewrite.get("query", "")).strip() or question
            trace.log(
                "main_agent",
                "image_observation",
                f"tools={','.join(image_observation.get('tools', []))} "
                f"image_count={image_observation.get('image_count', 0)} "
                f"summary={image_observation.get('summary', '')} "
                f"vlm_summary={image_observation.get('vlm_summary', '')} "
                f"suggested_query={image_observation.get('suggested_query', '')}",
            )
            trace.log(
                "main_agent",
                "image_rewrite",
                f"image_rewritten_query={final_query} reason={image_rewrite.get('reason', '')}",
            )
        else:
            final_query = question

        if rewrite_history:
            history_rewritten_query = self.generator.rewrite_current_turn(
                rewrite_history,
                final_query,
            )
            if history_rewritten_query:
                final_query = history_rewritten_query

        if image_observation:
            actionability = self.generator.assess_query_actionability(
                original_question,
                final_query,
                image_observation,
            )
            trace.log(
                "main_agent",
                "actionability",
                f"can_proceed={actionability.get('can_proceed')} "
                f"reason={actionability.get('reason', '')} "
                f"missing_info={actionability.get('missing_info', [])}",
            )
            if not actionability.get("can_proceed", True):
                result = self._build_image_followup_result(
                    original_question=original_question,
                    image_rewritten_query=final_query,
                    image_observation=image_observation,
                    image_rewrite=image_rewrite,
                    actionability=actionability,
                    trace=trace,
                )
                result["agent_trace"] = trace.events
                return result

        turns = self._split_turns_by_mode(final_query, turn_mode)
        question_text = "\n".join(turns)
        trace.log(
            "main_agent",
            "final_query",
            f"turn_mode={turn_mode} original_query={original_question} "
            f"final_query={question_text} history_rewritten_query={history_rewritten_query}",
        )
        trace.log("主agent", "接收", f"question_id={question_id or '-'} query={question_text}")

        if image_observation:
            trace.log(
                "main_agent",
                "image_observation",
                f"tools={','.join(image_observation.get('tools', []))} image_count={image_observation.get('image_count', 0)} "
                f"summary={image_observation.get('summary', '')} "
                f"suggested_query={image_observation.get('suggested_query', '')}",
            )

        rewritten_turns, task_origins, decompose_mode = self._decompose_turns(turns)
        trace.log(
            "主agent",
            "拆解",
            f"mode={decompose_mode} turns={len(turns)} tasks={len(rewritten_turns)} {rewritten_turns}",
        )

        tasks = self._build_tasks(
            turns,
            rewritten_turns,
            question_id,
            trace,
            task_origins=task_origins,
        )
        task_results = [
            self._run_task(task, return_debug=return_debug, trace=trace)
            for task in tasks
        ]
        selected_sub_agent = self._selected_sub_agent(tasks)
        question_type = selected_sub_agent if selected_sub_agent != "mixed" else "mixed"

        trace.log("主agent", "整合", f"tasks={len(task_results)} type={question_type}")
        result = self._build_final_result(
            question_text=question_text,
            turns=turns,
            rewritten_turns=rewritten_turns,
            tasks=tasks,
            task_results=task_results,
            selected_sub_agent=selected_sub_agent,
            return_debug=return_debug,
        )
        trace.log(
            "主agent",
            "输出",
            f"answer={result.get('answer', '')} image_ids={result.get('image_ids', [])}",
        )
        result["agent_trace"] = trace.events
        if image_observation:
            result["original_question"] = original_question
            result["image_rewritten_query"] = image_rewrite.get("query", "")
            result["image_rewrite_reason"] = image_rewrite.get("reason", "")
            result["history_rewritten_query"] = history_rewritten_query
            result["final_query"] = question_text
            result["actionability"] = actionability
            result["image_observation"] = image_observation
        return result

    @classmethod
    def _split_turns_by_mode(cls, question: str, turn_mode: str) -> List[str]:
        mode = (turn_mode or "auto_newline").strip().lower()
        if mode in {"single", "single_turn"}:
            turn = cls.clean_question_turn(question)
            return [turn] if turn else []
        if mode not in {"auto_newline", "auto", "newline"}:
            raise ValueError("turn_mode must be 'auto_newline' or 'single'")
        return cls.split_question_turns(question)

    def _build_image_followup_result(
        self,
        original_question: str,
        image_rewritten_query: str,
        image_observation: Dict[str, Any],
        image_rewrite: Dict[str, Any],
        actionability: Dict[str, Any],
        trace: AgentTraceLogger,
    ) -> Dict:
        answer = str(actionability.get("followup_question", "")).strip()
        if not answer:
            answer = self._fallback_image_followup(original_question, image_observation)
        trace.log("main_agent", "followup", f"answer={answer}")
        task = {
            "task_id": 1,
            "turn_index": 1,
            "turn_total": 1,
            "original_query": original_question,
            "rewritten_turn": image_rewritten_query,
            "query": image_rewritten_query or original_question,
            "sub_agent": "clarification",
            "route_plan": {
                "intent": "clarification",
                "confidence": 1.0,
                "reason": actionability.get("reason", ""),
                "source": "actionability",
            },
        }
        return {
            "question": image_rewritten_query or original_question,
            "rewritten_turns": [image_rewritten_query or original_question],
            "main_plan": {
                "turns": [image_rewritten_query or original_question],
                "rewritten_turns": [image_rewritten_query or original_question],
                "task_origins": [
                    {
                        "task_id": 1,
                        "turn_index": 1,
                        "original_query": original_question,
                        "rewritten_turn": image_rewritten_query,
                    }
                ],
                "task_count": 1,
                "selected_sub_agent": "clarification",
            },
            "tasks": [task],
            "task_results": [],
            "turn_results": [],
            "answers": [answer],
            "answer": answer,
            "image_ids": [],
            "question_type": "image_clarification",
            "route": "image_clarification",
            "route_plan": task["route_plan"],
            "selected_sub_agent": "clarification",
            "sub_agent_result_type": "image_clarification",
            "answer_sources": ["image_clarification"],
            "original_question": original_question,
            "image_rewritten_query": image_rewritten_query,
            "image_rewrite_reason": image_rewrite.get("reason", ""),
            "history_rewritten_query": "",
            "final_query": image_rewritten_query or original_question,
            "actionability": actionability,
            "image_observation": image_observation,
        }

    @staticmethod
    def _fallback_image_followup(
        original_question: str,
        image_observation: Dict[str, Any],
    ) -> str:
        is_zh = bool(re.search(r"[\u4e00-\u9fff]", original_question))
        product = str(image_observation.get("product_type", "")).strip()
        code = str(image_observation.get("visible_error_code", "")).strip()
        if is_zh:
            clue = "\u3001".join(item for item in [product, code] if item)
            if clue:
                return (
                    f"\u6211\u770b\u5230\u56fe\u7247\u4e2d\u6709 {clue} "
                    "\u8fd9\u4e9b\u7ebf\u7d22\uff0c\u4f46\u8fd8\u9700\u8981\u60a8\u8865\u5145"
                    "\u4ea7\u54c1\u540d\u79f0\u3001\u578b\u53f7\u6216\u60f3\u5904\u7406\u7684"
                    "\u5177\u4f53\u76ee\u6807\uff0c\u6211\u518d\u5e2e\u60a8\u7ee7\u7eed\u5224\u65ad\u3002"
                )
            return (
                "\u6211\u770b\u5230\u60a8\u4e0a\u4f20\u4e86\u56fe\u7247\uff0c\u4f46\u8fd8\u9700"
                "\u8981\u60a8\u8865\u5145\u5177\u4f53\u60f3\u54a8\u8be2\u7684\u95ee\u9898\u3001"
                "\u4ea7\u54c1\u7c7b\u578b\u6216\u4f7f\u7528\u573a\u666f\uff0c\u6211\u518d"
                "\u5e2e\u60a8\u7ee7\u7eed\u5904\u7406\u3002"
            )
        if is_zh:
            clue = "、".join(item for item in [product, code] if item)
            if clue:
                return f"我看到了图片中的 {clue} 线索，但还需要您补充具体产品型号或想处理的目标，才能继续判断。"
            return "我看到了您上传的图片，但还需要您补充具体想咨询的问题或产品场景，才能继续处理。"
        clue = ", ".join(item for item in [product, code] if item)
        if clue:
            return f"I can see {clue} in the image, but I still need the product model or your exact goal before proceeding."
        return "I can see the uploaded image, but I need your exact question or product context before proceeding."

    def _build_tasks(
        self,
        turns: List[str],
        rewritten_turns: List[str],
        question_id: str,
        trace: AgentTraceLogger,
        task_origins: List[Dict] | None = None,
    ) -> List[Dict]:
        tasks: List[Dict] = []
        turn_total = len(rewritten_turns)
        for idx, query in enumerate(rewritten_turns, start=1):
            origin = task_origins[idx - 1] if task_origins and idx <= len(task_origins) else {}
            original_turn = origin.get("original_query") or (
                turns[idx - 1] if idx <= len(turns) else query
            )
            route_plan = self._route(query, question_id)
            task = {
                "task_id": idx,
                "turn_index": origin.get("turn_index", idx),
                "turn_total": turn_total,
                "original_query": original_turn,
                "rewritten_turn": origin.get("rewritten_turn", query),
                "query": query,
                "sub_agent": route_plan["intent"],
                "route_plan": route_plan,
            }
            trace.log(
                "主agent",
                "调度",
                f"task={idx} -> 子agent={route_plan['intent']} query={query}",
            )
            tasks.append(task)
        return tasks

    def _decompose_turns(self, turns: List[str]) -> tuple[List[str], List[Dict], str]:
        if len(turns) <= 1:
            base_turns = turns
            mode = "single_query_decompose"
        else:
            base_turns = self.generator.rewrite_customer_turns(turns)
            if len(base_turns) != len(turns):
                base_turns = turns
            mode = "multi_turn_rewrite_then_decompose"

        decomposed_turns: List[str] = []
        origins: List[Dict] = []
        for idx, turn in enumerate(base_turns, start=1):
            pieces = self.generator.decompose_user_query(turn)
            pieces = [piece.strip() for piece in pieces if piece.strip()]
            if not pieces:
                pieces = [turn]
            original_turn = turns[idx - 1] if idx <= len(turns) else turn
            pieces = self._repair_decomposition_coverage(original_turn, turn, pieces)
            for piece in pieces:
                decomposed_turns.append(piece)
                origins.append(
                    {
                        "turn_index": idx,
                        "original_query": original_turn,
                        "rewritten_turn": turn,
                    }
                )
        return decomposed_turns, origins, mode

    @classmethod
    def _repair_decomposition_coverage(
        cls,
        original_turn: str,
        rewritten_turn: str,
        pieces: List[str],
    ) -> List[str]:
        repaired = list(pieces)
        for candidate in cls._rule_extract_manual_questions(original_turn):
            if not cls._pieces_cover(candidate, repaired):
                repaired.append(candidate)
        for candidate in cls._rule_extract_manual_questions(rewritten_turn):
            if not cls._pieces_cover(candidate, repaired):
                repaired.append(candidate)
        return repaired

    @classmethod
    def _rule_extract_manual_questions(cls, text: str) -> List[str]:
        manual_terms = [
            "如何",
            "怎么",
            "怎样",
            "步骤",
            "安装",
            "清洁",
            "更换",
            "使用",
            "操作",
            "设置",
            "启动",
            "关闭",
            "充电",
            "滤网",
            "节能模式",
            "energy saving",
            "clean",
            "install",
            "replace",
            "use",
            "operate",
            "filter",
        ]
        customer_terms = [
            "发票",
            "退款",
            "退货",
            "退换货",
            "运费",
            "售后",
            "维修政策",
            "refund",
            "invoice",
            "after-sales",
            "return policy",
        ]
        separators = [
            "另外",
            "此外",
            "同时",
            "并且",
            "而且",
            "还有",
            "顺便",
            "also",
            "and also",
        ]
        lower_text = text.lower()
        if not any(term in lower_text for term in manual_terms):
            return []
        if not any(term in lower_text for term in customer_terms):
            return []

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        segments = [normalized]
        for sep in separators:
            next_segments: List[str] = []
            for segment in segments:
                parts = re.split(re.escape(sep), segment, flags=re.IGNORECASE)
                next_segments.extend(parts)
            segments = next_segments

        clause_segments: List[str] = []
        for segment in segments:
            clause_segments.extend(re.split(r"[？?。.;；,，]", segment))
        segments = clause_segments

        results: List[str] = []
        for segment in segments:
            candidate = cls.clean_question_turn(segment)
            candidate = candidate.strip(" ，,。.;；")
            if not candidate:
                continue
            candidate_lower = candidate.lower()
            if any(term in candidate_lower for term in manual_terms):
                if not candidate.endswith(("？", "?", "。", ".")):
                    candidate += "？" if cls._contains_cjk(candidate) else "?"
                results.append(candidate)
        return cls._dedupe_strings(results)

    @staticmethod
    def _pieces_cover(candidate: str, pieces: List[str]) -> bool:
        candidate_tokens = MainAgent._coverage_tokens(candidate)
        if not candidate_tokens:
            return True
        for piece in pieces:
            piece_tokens = MainAgent._coverage_tokens(piece)
            overlap = candidate_tokens & piece_tokens
            if len(overlap) >= max(1, min(3, len(candidate_tokens))):
                return True
        return False

    @staticmethod
    def _coverage_tokens(text: str) -> set[str]:
        lower = text.lower()
        tokens = set(re.findall(r"[a-zA-Z0-9]+", lower))
        tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", lower))
        return {token for token in tokens if len(token) >= 2}

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _dedupe_strings(items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _run_task(
        self,
        task: Dict,
        return_debug: bool,
        trace: AgentTraceLogger,
    ) -> Dict:
        if task["sub_agent"] == "customer_service":
            return self._run_customer_task(task, return_debug=return_debug, trace=trace)
        return self._run_manual_task(task, return_debug=return_debug, trace=trace)

    def _run_customer_task(
        self,
        task: Dict,
        return_debug: bool,
        trace: AgentTraceLogger,
    ) -> Dict:
        child_result = self.customer_agent.run(
            task["query"],
            return_debug=return_debug,
            trace=trace,
            turn_index=task["turn_index"],
            turn_total=task["turn_total"],
            original_query=task["original_query"],
        )
        if child_result.get("hit"):
            answer = child_result.get("answer", "")
            analysis = {}
            source = child_result.get("answer_source", "customer_qa")
        else:
            trace.log("主agent", "客服分析", f"task={task['task_id']}")
            analysis = self.generator.analyze_customer_turn(task["query"])
            answer = self.generator.generate_direct(task["query"], analysis)
            source = "direct_llm"

        answer, image_ids = self.normalize_pic_tokens(answer)
        return {
            "task_id": task["task_id"],
            "turn_index": task["turn_index"],
            "original_query": task["original_query"],
            "rewritten_turn": task.get("rewritten_turn", task["query"]),
            "query": task["query"],
            "question_type": "customer_service",
            "route": child_result.get("route", "direct_llm"),
            "answer": answer,
            "image_ids": image_ids,
            "answer_source": source,
            "customer_analysis": analysis,
            "child_result": self._serialize_child_result(child_result, return_debug),
        }

    def _run_manual_task(
        self,
        task: Dict,
        return_debug: bool,
        trace: AgentTraceLogger,
    ) -> Dict:
        child_result = self.manual_agent.run(
            task["query"],
            return_debug=return_debug,
            trace=trace,
            task_id=task["task_id"],
        )

        return {
            "task_id": task["task_id"],
            "turn_index": task["turn_index"],
            "original_query": task["original_query"],
            "rewritten_turn": task.get("rewritten_turn", task["query"]),
            "query": task["query"],
            "question_type": "manual",
            "route": child_result.get("route", "agentic_retrieval"),
            "initial_query": task["query"],
            "initial_chunks": child_result.get("initial_chunks", []),
            "retrieval_plans": child_result.get("retrieval_plans", []),
            "followup_queries": child_result.get("followup_queries", []),
            "followup_searches": child_result.get("followup_searches", []),
            "followup_chunks": child_result.get("followup_chunks", []),
            "candidate_chunks": child_result.get("candidate_chunks", []),
            "evidence_selection": child_result.get("evidence_selection", {}),
            "chunks": child_result.get("chunks", []),
            "answer": child_result.get("answer", ""),
            "image_ids": child_result.get("image_ids", []),
            "answer_source": child_result.get("answer_source", "manual_rag"),
            "child_result": self._serialize_child_result(child_result, return_debug),
        }

    def _build_final_result(
        self,
        question_text: str,
        turns: List[str],
        rewritten_turns: List[str],
        tasks: List[Dict],
        task_results: List[Dict],
        selected_sub_agent: str,
        return_debug: bool,
    ) -> Dict:
        turn_results = self._build_turn_results(turns, task_results)
        answers = [result.get("answer", "") for result in turn_results]
        answer = "\n".join(answer for answer in answers if answer)
        image_ids = self._collect_result_image_ids(task_results)
        question_type = selected_sub_agent if selected_sub_agent != "mixed" else "mixed"
        route = self._result_route(question_type, task_results)

        result = {
            "question": question_text,
            "rewritten_turns": rewritten_turns,
            "main_plan": {
                "turns": turns,
                "rewritten_turns": rewritten_turns,
                "task_origins": [
                    {
                        "task_id": task.get("task_id"),
                        "turn_index": task.get("turn_index"),
                        "original_query": task.get("original_query", ""),
                        "rewritten_turn": task.get("rewritten_turn", ""),
                    }
                    for task in tasks
                ],
                "task_count": len(tasks),
                "selected_sub_agent": selected_sub_agent,
            },
            "tasks": tasks,
            "task_results": [
                self._serialize_task_result(item, include_debug=return_debug)
                for item in task_results
            ],
            "turn_results": turn_results,
            "answers": answers,
            "answer": answer,
            "image_ids": image_ids,
            "question_type": question_type,
            "route": route,
            "route_plan": self._combined_route_plan(tasks, selected_sub_agent),
            "selected_sub_agent": selected_sub_agent,
            "sub_agent_result_type": question_type,
            "answer_sources": [
                result.get("answer_source", "") for result in task_results
            ],
        }
        self._attach_legacy_debug_fields(
            result,
            task_results,
            include_debug=return_debug,
        )
        return result

    def _build_turn_results(
        self,
        turns: List[str],
        task_results: List[Dict],
    ) -> List[Dict]:
        grouped: Dict[int, List[Dict]] = {}
        for result in task_results:
            turn_index = int(result.get("turn_index") or result.get("task_id") or 1)
            grouped.setdefault(turn_index, []).append(result)

        turn_results: List[Dict] = []
        for turn_index in sorted(grouped):
            results = grouped[turn_index]
            results.sort(key=lambda item: int(item.get("task_id") or 0))
            turn_answer = "\n".join(
                item.get("answer", "") for item in results if item.get("answer")
            )
            turn_results.append(
                {
                    "turn_index": turn_index,
                    "original_query": turns[turn_index - 1]
                    if 0 <= turn_index - 1 < len(turns)
                    else results[0].get("original_query", ""),
                    "task_ids": [item.get("task_id") for item in results],
                    "question_types": [
                        item.get("question_type", "") for item in results
                    ],
                    "answer": turn_answer,
                    "image_ids": self._collect_result_image_ids(results),
                    "answer_sources": [
                        item.get("answer_source", "") for item in results
                    ],
                }
            )
        return turn_results

    @staticmethod
    def _selected_sub_agent(tasks: List[Dict]) -> str:
        agents = [task.get("sub_agent", "") for task in tasks]
        unique_agents = {agent for agent in agents if agent}
        if len(unique_agents) == 1:
            return agents[0] if agents else "manual"
        return "mixed"

    @staticmethod
    def _result_route(question_type: str, task_results: List[Dict]) -> str:
        routes = {result.get("route", "") for result in task_results}
        if question_type == "mixed":
            return "mixed_orchestration"
        if question_type == "manual":
            return "agentic_retrieval"
        if any(route == "customer_qa_retrieval" for route in routes):
            return "customer_qa_retrieval"
        return "direct_llm"

    @staticmethod
    def _combined_route_plan(tasks: List[Dict], selected_sub_agent: str) -> Dict:
        if len(tasks) == 1:
            return tasks[0].get("route_plan", {})
        return {
            "intent": selected_sub_agent,
            "source": "main_agent",
            "task_routes": [
                {
                    "task_id": task.get("task_id"),
                    "intent": task.get("sub_agent"),
                    "query": task.get("query"),
                    "route_plan": task.get("route_plan", {}),
                }
                for task in tasks
            ],
        }

    @classmethod
    def _collect_result_image_ids(cls, task_results: List[Dict]) -> List[str]:
        image_ids: List[str] = []
        for result in task_results:
            for image_id in result.get("image_ids") or []:
                if image_id not in image_ids:
                    image_ids.append(image_id)
        return image_ids

    def _attach_legacy_debug_fields(
        self,
        result: Dict,
        task_results: List[Dict],
        include_debug: bool,
    ) -> None:
        customer_results = [
            item for item in task_results if item.get("question_type") == "customer_service"
        ]
        manual_results = [
            item for item in task_results if item.get("question_type") == "manual"
        ]
        if customer_results:
            result["customer_analyses"] = [
                item.get("customer_analysis", {}) for item in customer_results
            ]
            qa_retrievals = []
            for item in customer_results:
                child_result = item.get("child_result", {})
                qa = child_result.get("customer_qa_retrieval")
                if qa:
                    qa_retrievals.append(qa)
            if qa_retrievals:
                result["customer_qa_retrieval"] = qa_retrievals
        if not manual_results or not include_debug:
            return
        result["initial_query"] = manual_results[0].get("initial_query", "")
        result["initial_chunks"] = self._dump_chunks(
            self._flatten_chunks(manual_results, "initial_chunks")
        )
        result["retrieval_plans"] = [
            plan
            for item in manual_results
            for plan in item.get("retrieval_plans", [])
        ]
        result["followup_queries"] = [
            query
            for item in manual_results
            for query in item.get("followup_queries", [])
        ]
        result["followup_searches"] = [
            {
                "round": search.get("round"),
                "query": search.get("query"),
                "chunks": self._dump_chunks(search.get("chunks") or []),
            }
            for item in manual_results
            for search in item.get("followup_searches", [])
        ]
        result["followup_chunks"] = self._dump_chunks(
            self._flatten_chunks(manual_results, "followup_chunks")
        )
        result["candidate_chunks"] = self._dump_chunks(
            self._flatten_chunks(manual_results, "candidate_chunks")
        )
        if len(manual_results) == 1:
            result["evidence_selection"] = manual_results[0].get(
                "evidence_selection", {}
            )
        else:
            result["evidence_selection"] = {
                "per_task": [
                    {
                        "task_id": item.get("task_id"),
                        "selection": item.get("evidence_selection", {}),
                    }
                    for item in manual_results
                ]
            }
        result["chunks"] = self._dump_chunks(
            self._flatten_chunks(manual_results, "chunks")
        )

    @staticmethod
    def _flatten_chunks(task_results: List[Dict], key: str) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        for item in task_results:
            chunks.extend(item.get(key) or [])
        return chunks

    @staticmethod
    def _dump_chunks(chunks: List[RetrievedChunk]) -> List[Dict]:
        return [
            chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            for chunk in chunks
        ]

    def _serialize_task_result(self, result: Dict, include_debug: bool) -> Dict:
        data = {
            "task_id": result.get("task_id"),
            "turn_index": result.get("turn_index"),
            "original_query": result.get("original_query", ""),
            "rewritten_turn": result.get("rewritten_turn", ""),
            "query": result.get("query"),
            "question_type": result.get("question_type"),
            "route": result.get("route"),
            "answer": result.get("answer", ""),
            "image_ids": result.get("image_ids", []),
            "answer_source": result.get("answer_source", ""),
        }
        if result.get("customer_analysis") is not None:
            data["customer_analysis"] = result.get("customer_analysis", {})
        if include_debug:
            for key in [
                "initial_query",
                "retrieval_plans",
                "followup_queries",
                "evidence_selection",
                "child_result",
            ]:
                if key in result:
                    data[key] = result[key]
            for key in [
                "initial_chunks",
                "followup_chunks",
                "candidate_chunks",
                "chunks",
            ]:
                if key in result:
                    data[key] = self._dump_chunks(result[key])
            if "followup_searches" in result:
                data["followup_searches"] = [
                    {
                        "round": search.get("round"),
                        "query": search.get("query"),
                        "chunks": self._dump_chunks(search.get("chunks") or []),
                    }
                    for search in result["followup_searches"]
                ]
        return data

    def _serialize_child_result(self, result: Dict, include_debug: bool) -> Dict:
        data = {}
        for key, value in result.items():
            if key in {
                "initial_chunks",
                "followup_chunks",
                "candidate_chunks",
                "chunks",
            }:
                if include_debug:
                    data[key] = self._dump_chunks(value)
                continue
            if key == "followup_searches":
                if include_debug:
                    data[key] = [
                        {
                            "round": search.get("round"),
                            "query": search.get("query"),
                            "chunks": self._dump_chunks(search.get("chunks") or []),
                        }
                        for search in value
                    ]
                continue
            if key == "customer_qa_retrieval" and not include_debug:
                continue
            data[key] = value
        return data

    def _route(self, question: str, question_id: str = "") -> Dict:
        llm_plan = self.generator.route_intent(question, question_id)
        rule_intent = self._rule_intent(question, question_id)
        source = "llm"
        rule_override = False

        if self._is_strong_customer_id(question_id):
            intent = "customer_service"
            source = "rule"
            rule_override = bool(llm_plan and llm_plan.get("intent") != intent)
        elif not llm_plan:
            intent = rule_intent
            source = "rule"
        elif float(llm_plan.get("confidence", 0.0)) < ROUTE_CONFIDENCE_THRESHOLD:
            intent = rule_intent
            source = "rule"
            rule_override = llm_plan.get("intent") != intent
        else:
            intent = llm_plan["intent"]
            if rule_intent == "customer_service" and intent != rule_intent:
                intent = rule_intent
                source = "rule"
                rule_override = True

        return {
            "intent": intent,
            "confidence": float(llm_plan.get("confidence", 0.0)) if llm_plan else 0.0,
            "reason": llm_plan.get("reason", "") if llm_plan else "LLM 路由失败，使用规则兜底。",
            "source": source,
            "llm_plan": llm_plan or {},
            "rule_intent": rule_intent,
            "rule_override": rule_override,
        }

    @classmethod
    def _rule_intent(cls, question: str, question_id: str = "") -> str:
        if cls.rule_customer_service_question(question, question_id):
            return "customer_service"
        return "manual"

    @staticmethod
    def _is_strong_customer_id(question_id: str = "") -> bool:
        try:
            return bool(question_id and int(question_id) < 64)
        except ValueError:
            return False

    @staticmethod
    def _is_customer_service_question(question: str, question_id: str = "") -> bool:
        return AgentHelpers.rule_customer_service_question(question, question_id)


class CustomerAgent(MainAgent):
    """Backward-compatible public agent name used by CLI and batch scripts."""

    pass
