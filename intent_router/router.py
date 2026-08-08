from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI

from .policy import fuse_boundary_decisions


DEFAULT_MODEL = "qwen3-32b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CONFIG_PATH = Path.home() / "bailian_api.txt"
_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_SHORT_ENGLISH_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,23}$")
_LATEST_TRUNCATION_MARKER = "\n...[当前用户消息中间内容已截断]...\n"
_HISTORY_TRUNCATION_MARKER = "\n...[历史消息中间内容已截断]...\n"
_REFERENCE_CONTEXT_PATTERN = re.compile(
    r"(?:这个|那个|这些|那些|这款|那款|这几款|那几款|"
    r"这[一二三四五六七八九十百\d]+款|那[一二三四五六七八九十百\d]+款|"
    r"它|它们|他们|她们|其中|就它|同款|上面|上述|前面|刚才|"
    r"你(?:刚才)?(?:说|提|推荐|列|展示|找)|"
    r"第[一二三四五六七八九十百\d]+(?:个|款|件|项|只|台|种))"
)

_STOCK_IDS = frozenset(
    {"StockRoute", "StockAdvice", "StockResearch", "StockOperation", "OtherFinance", "NoStock"}
)
_PRODUCT_IDS = frozenset(
    {
        "ProductRoute",
        "ProductKnowledge",
        "ProductFollowup",
        "NonRetail",
        "MultiProduct",
        "NoProduct",
    }
)
_PRODUCTION_CANDIDATE_ORDER = (
    "StockInfo",
    "StockAdvice",
    "StockResearch",
    "StockOther",
    "FinanceOther",
    "Ecommerce",
    "ProductInfo",
    "ProductOther",
    "NonRetail",
    "MultiProduct",
    "ChitChat",
    "NoRequest",
    "NoAvailable",
)

# Both prefixes are deliberately short, stable and free of benchmark examples.
_STOCK_SYSTEM_PROMPT = """你是股票行情能力的边界分类器，不回答用户问题。以当前用户请求为准；历史只用于解析明确指代，assistant/tool只提供事实。按语义而非关键词选择：
- StockRoute：用户请求股票、上市公司证券、股票指数、股票市场或行业板块的当前/历史公开信息、市场概念、行情解释或非决策性市场分析。
- StockAdvice：预测未来证券结果、条件推演、选股荐股或买卖持仓决策。
- StockResearch：产业竞争地位、股权持有人、股东行为或长期因果研究。
- StockOperation：证券账户、委托交易、资金操作或证券软件本身。
- OtherFinance：基金、债券、期货、外汇、贵金属、加密资产或银行产品自身；若最终询问其代表的股票指数或板块则不属。
- NoStock：没有完整的股票领域请求，或股票内容只是陈述、背景、引用或待处理文字。
只输出一个上述id，不得解释。"""

_PRODUCT_SYSTEM_PROMPT = """你是普通零售商品发现能力的边界分类器，不回答用户问题。以当前用户请求为准；历史只用于解析明确指代，assistant/tool只提供事实。按最终要获得的对象和动作选择：
- ProductRoute：用户确实要获得一个待购普通零售商品，动作是搜索、推荐、筛选、展示、平台查找、购买、复购、找同款/替换件，或为购买进行场景选型和同类比较。无需指定平台。
- ProductKnowledge：只问已知商品的属性、价格、图片、评价、真伪、适用性、统计或兼容，或无具体购买约束的一般品牌、品类、规格、样式知识。
- ProductFollowup：已经完成选品后的加购/支付，已有订单的物流售后，或已有商品的使用故障。
- NonRetail：最终对象是机动车整车、房产、药品、保险、票旅住宿餐饮或专业服务，而不是普通零售商品。
- MultiProduct：分别发现两个及以上互不相关的普通商品。
- NoProduct：没有完整的商品请求，或商品内容只是陈述、背景、引用、纯名称或无法恢复对象的残句。
主题、用途、平台和媒介不能改变最终对象的类型。只输出一个上述id，不得解释。"""


class RoutingStatus(str, Enum):
    """Operationally distinct outcomes of one routing request."""

    ROUTED = "routed"
    NO_ROUTE = "no_route"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    API_ERROR = "api_error"


@dataclass(frozen=True)
class IntentCandidate:
    """One model-visible name plus its private backend mapping."""

    candidate_id: str
    name: str
    intent_label: str | None
    description: str
    examples: tuple[str, ...] = ()
    category: str | None = None
    excludes: tuple[str, ...] = ()

    @property
    def is_virtual(self) -> bool:
        return self.intent_label is None


@dataclass(frozen=True)
class RouterCredentials:
    api_key: str
    base_url: str


@dataclass(frozen=True)
class PresentedCandidate:
    candidate: IntentCandidate


@dataclass(frozen=True)
class PreparedRoutingRequest:
    """Two prompt-ready boundary requests and their private output mapping."""

    # ``api_messages`` remains the stock request for source compatibility.
    api_messages: tuple[dict[str, str], ...]
    product_api_messages: tuple[dict[str, str], ...]
    presented_candidates: tuple[PresentedCandidate, ...]
    normalized_messages: tuple[dict[str, str], ...]
    order_seed: int

    @property
    def stock_api_messages(self) -> tuple[dict[str, str], ...]:
        return self.api_messages

    @property
    def name_to_candidate(self) -> dict[str, IntentCandidate]:
        return {item.candidate.name: item.candidate for item in self.presented_candidates}


@dataclass(frozen=True)
class IntentDecision:
    """Public result. ``intent_label`` is set only for a valid real route."""

    status: RoutingStatus
    intent_label: str | None
    selected_candidate_id: str | None
    selected_candidate_name: str | None
    output_name: str | None
    latency_ms: float
    model: str
    prompt_sha256: str | None = None
    candidate_order_seed: int | None = None
    raw_model_output: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    selected_probability: float | None = None
    error_type: str | None = None
    decision_reason: str | None = None

    @property
    def should_route(self) -> bool:
        return self.status is RoutingStatus.ROUTED

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["should_route"] = self.should_route
        return result


def default_category_descriptions() -> dict[str, str]:
    return {
        "Stock": "股票与证券市场边界。",
        "Product": "普通零售商品发现边界。",
    }


def default_candidates() -> tuple[IntentCandidate, ...]:
    """Return two real outputs and generic virtual boundary outputs."""

    return (
        IntentCandidate(
            "stock_market_information",
            "StockInfo",
            "SearchStockQuotes",
            "股票市场当前或历史公开信息及非决策性分析。",
            category="Stock",
        ),
        IntentCandidate(
            "no_route_stock_advice",
            "StockAdvice",
            None,
            "证券预测、选择或交易决策。",
            category="Stock",
        ),
        IntentCandidate(
            "no_route_stock_research",
            "StockResearch",
            None,
            "产业、股权、股东行为或长期因果研究。",
            category="Stock",
        ),
        IntentCandidate(
            "no_route_stock_other",
            "StockOther",
            None,
            "证券账户、交易操作或证券软件本身。",
            category="Stock",
        ),
        IntentCandidate(
            "no_route_other_finance",
            "FinanceOther",
            None,
            "非股票金融产品自身的信息或决策。",
            category="Stock",
        ),
        IntentCandidate(
            "ecommerce_product_recommendation",
            "Ecommerce",
            "RecommendProduct",
            "一个待购普通零售商品的发现或购买。",
            category="Product",
        ),
        IntentCandidate(
            "no_route_product_information",
            "ProductInfo",
            None,
            "已知商品或一般品类知识，不产生商品发现结果。",
            category="Product",
        ),
        IntentCandidate(
            "no_route_product_other",
            "ProductOther",
            None,
            "选品后的履约、订单处理或已有商品处理。",
            category="Product",
        ),
        IntentCandidate(
            "no_route_non_retail",
            "NonRetail",
            None,
            "不属于普通零售商品的对象或服务。",
            category="Product",
        ),
        IntentCandidate(
            "no_route_multi_product",
            "MultiProduct",
            None,
            "分别发现多个互不相关的商品。",
            category="Product",
        ),
        IntentCandidate(
            "no_route_chitchat",
            "ChitChat",
            None,
            "闲聊。",
        ),
        IntentCandidate(
            "no_route_no_request",
            "NoRequest",
            None,
            "没有完整请求。",
        ),
        IntentCandidate(
            "no_route_no_available",
            "NoAvailable",
            None,
            "其它不匹配的任务或边界冲突。",
        ),
    )


def load_bailian_credentials(
    config_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> RouterCredentials:
    """Load credentials without logging them."""

    resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    resolved_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("QWEN_BASE_URL")
    if base_url:
        resolved_url = base_url

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    file_key: str | None = None
    file_url: str | None = None
    if (not resolved_key or not resolved_url) and path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip().strip("'\"")
                if key in {"DASHSCOPE_API_KEY", "OPENAI_API_KEY", "API_KEY"}:
                    file_key = value
                elif key in {"DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "BASE_URL"}:
                    file_url = value
            elif re.match(r"^https?://", line, flags=re.IGNORECASE):
                file_url = line
            elif file_key is None:
                file_key = line.strip("'\"")

    resolved_key = resolved_key or file_key or os.getenv("OPENAI_API_KEY")
    resolved_url = (resolved_url or file_url or DEFAULT_BASE_URL).rstrip("/")
    if not resolved_key:
        raise ValueError("未找到百炼 API Key；请设置 DASHSCOPE_API_KEY，或提供 ~/bailian_api.txt")
    if not re.match(r"^https?://", resolved_url, flags=re.IGNORECASE):
        raise ValueError("base_url 必须以 http:// 或 https:// 开头")
    return RouterCredentials(api_key=resolved_key, base_url=resolved_url)


class IntentRouter:
    """Two-boundary, fail-closed intent selector for Qwen3-32B."""

    def __init__(
        self,
        *,
        credentials: RouterCredentials | None = None,
        config_path: str | Path | None = None,
        model: str = DEFAULT_MODEL,
        candidates: Sequence[IntentCandidate] | None = None,
        category_descriptions: Mapping[str, str] | None = None,
        timeout_seconds: float = 10.0,
        max_history_messages: int = 16,
        max_history_chars: int = 12_000,
        max_assistant_history_chars: int = 1_200,
        min_route_probability: float | None = None,
        default_order_seed: int | None = 0,
        client: Any | None = None,
        fail_closed: bool = True,
    ) -> None:
        self.model = model
        self.candidates = tuple(candidates or default_candidates())
        self.category_descriptions = dict(
            default_category_descriptions()
            if category_descriptions is None
            else category_descriptions
        )
        self.timeout_seconds = timeout_seconds
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars
        self.max_assistant_history_chars = max_assistant_history_chars
        self.min_route_probability = min_route_probability
        self.default_order_seed = default_order_seed
        self.fail_closed = fail_closed
        self._validate_candidates(self.candidates, self.category_descriptions)
        if min(max_history_messages, max_history_chars, max_assistant_history_chars) <= 0:
            raise ValueError("历史消息和字符上限必须大于 0")
        if min_route_probability is not None and not 0 < min_route_probability <= 1:
            raise ValueError("min_route_probability 必须位于 (0, 1]，或设为 None")

        if client is not None:
            self.client = client
        else:
            resolved = credentials or load_bailian_credentials(config_path)
            self.client = OpenAI(
                api_key=resolved.api_key,
                base_url=resolved.base_url,
                timeout=timeout_seconds,
                max_retries=0,
            )

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[IntentCandidate],
        category_descriptions: Mapping[str, str],
    ) -> None:
        if not candidates:
            raise ValueError("候选集不能为空")
        ids = [candidate.candidate_id for candidate in candidates]
        names = [candidate.name for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id 必须唯一")
        if len(names) != len(set(names)):
            raise ValueError("候选英文名称必须唯一")
        if any(_SHORT_ENGLISH_NAME.fullmatch(name) is None for name in names):
            raise ValueError("候选名称必须是 2-24 位的简短英文/数字标识")
        labels = [candidate.intent_label for candidate in candidates if candidate.intent_label]
        if len(labels) != len(set(labels)):
            raise ValueError("真实候选的 intent_label 必须唯一")
        if not any(candidate.is_virtual for candidate in candidates):
            raise ValueError("至少需要一个虚拟候选")
        if any(not candidate.description.strip() for candidate in candidates):
            raise ValueError("候选 description 不能为空")
        required = set(_PRODUCTION_CANDIDATE_ORDER)
        missing = required - set(names)
        if missing:
            raise ValueError(f"缺少路由器所需候选: {sorted(missing)}")
        referenced = {candidate.category for candidate in candidates if candidate.category}
        missing_categories = referenced - set(category_descriptions)
        if missing_categories:
            raise ValueError(f"候选引用了未定义的 category: {sorted(missing_categories)}")

    def prepare(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        order_seed: int | None = None,
    ) -> PreparedRoutingRequest:
        normalized = self._normalize_and_trim_messages(messages)
        if order_seed is not None:
            seed = order_seed
        elif self.default_order_seed is not None:
            seed = self.default_order_seed
        else:
            seed = self._conversation_seed(normalized)
        user_prompt = self._build_user_prompt(normalized)
        presented = self._present_candidates()
        return PreparedRoutingRequest(
            api_messages=(
                {"role": "system", "content": _STOCK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ),
            product_api_messages=(
                {"role": "system", "content": _PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ),
            presented_candidates=presented,
            normalized_messages=normalized,
            order_seed=seed,
        )

    def route(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        order_seed: int | None = None,
    ) -> IntentDecision:
        prepared = self.prepare(messages, order_seed=order_seed)
        prompt_sha256 = hashlib.sha256(
            (_STOCK_SYSTEM_PROMPT + "\n" + _PRODUCT_SYSTEM_PROMPT).encode("utf-8")
        ).hexdigest()
        started = time.perf_counter()
        stock_response: Any | None = None
        product_response: Any | None = None
        stock_responses: list[Any] = []
        product_responses: list[Any] = []
        try:
            stock_response = self._create_completion(prepared.stock_api_messages)
            stock_responses.append(stock_response)
            product_response = self._create_completion(prepared.product_api_messages)
            product_responses.append(product_response)

            stock_raws = [self._extract_content(stock_response)]
            product_raws = [self._extract_content(product_response)]
            stock_id = self._parse_boundary_id(stock_raws[-1], _STOCK_IDS)
            product_id = self._parse_boundary_id(product_raws[-1], _PRODUCT_IDS)
            if stock_id is None:
                stock_response = self._retry_format(prepared.stock_api_messages)
                stock_responses.append(stock_response)
                stock_raws.append(self._extract_content(stock_response))
                stock_id = self._parse_boundary_id(stock_raws[-1], _STOCK_IDS)
            if product_id is None:
                product_response = self._retry_format(prepared.product_api_messages)
                product_responses.append(product_response)
                product_raws.append(self._extract_content(product_response))
                product_id = self._parse_boundary_id(product_raws[-1], _PRODUCT_IDS)
        except Exception as exc:
            if not self.fail_closed:
                raise
            return self._error_decision(
                RoutingStatus.API_ERROR,
                started,
                prepared,
                prompt_sha256,
                type(exc).__name__,
            )

        raw_output = json.dumps(
            {
                "stock": stock_raws[0] if len(stock_raws) == 1 else stock_raws,
                "product": product_raws[0] if len(product_raws) == 1 else product_raws,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        all_responses = [*stock_responses, *product_responses]
        prompt_tokens = self._sum_usage(all_responses, "prompt_tokens")
        completion_tokens = self._sum_usage(all_responses, "completion_tokens")
        if stock_id is None or product_id is None:
            return IntentDecision(
                status=RoutingStatus.INVALID_MODEL_OUTPUT,
                intent_label=None,
                selected_candidate_id=None,
                selected_candidate_name=None,
                output_name=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                model=self.model,
                prompt_sha256=prompt_sha256,
                candidate_order_seed=prepared.order_seed,
                raw_model_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_type="unrecognized_candidate_name",
                decision_reason="invalid_boundary_output",
            )

        fusion = fuse_boundary_decisions(prepared.normalized_messages, stock_id, product_id)
        selected = prepared.name_to_candidate.get(fusion.output_name)
        if selected is None:
            selected = prepared.name_to_candidate["NoAvailable"]

        selected_probability = None
        if fusion.intent_label == "SearchStockQuotes":
            selected_probability = self._boundary_probability(stock_response, stock_id)
        elif fusion.intent_label == "RecommendProduct":
            selected_probability = self._boundary_probability(product_response, product_id)

        intent_label = selected.intent_label if fusion.intent_label is not None else None
        if intent_label is None:
            status = RoutingStatus.NO_ROUTE
            error_type = None
        elif self.min_route_probability is not None and (
            selected_probability is None or selected_probability < self.min_route_probability
        ):
            status = RoutingStatus.LOW_CONFIDENCE
            intent_label = None
            error_type = (
                "missing_logprobs"
                if selected_probability is None
                else "below_route_threshold"
            )
        else:
            status = RoutingStatus.ROUTED
            error_type = None

        return IntentDecision(
            status=status,
            intent_label=intent_label,
            selected_candidate_id=selected.candidate_id,
            selected_candidate_name=selected.name,
            output_name=selected.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            model=self.model,
            prompt_sha256=prompt_sha256,
            candidate_order_seed=prepared.order_seed,
            raw_model_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            selected_probability=selected_probability,
            error_type=error_type,
            decision_reason=fusion.reason,
        )

    def route_text(self, text: str) -> IntentDecision:
        return self.route(({"role": "user", "content": text},))

    def _create_completion(self, api_messages: Sequence[Mapping[str, str]]) -> Any:
        options: dict[str, Any] = {
            "model": self.model,
            "messages": list(api_messages),
            "temperature": 0,
            "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if self.min_route_probability is not None:
            options["logprobs"] = True
        return self.client.chat.completions.create(**options)

    def _retry_format(self, api_messages: Sequence[Mapping[str, str]]) -> Any:
        corrected = [dict(message) for message in api_messages]
        corrected[-1]["content"] += "\n上次格式错误。仅输出最匹配的一个id。 /no_think"
        return self._create_completion(corrected)

    def _error_decision(
        self,
        status: RoutingStatus,
        started: float,
        prepared: PreparedRoutingRequest,
        prompt_sha256: str,
        error_type: str,
    ) -> IntentDecision:
        return IntentDecision(
            status=status,
            intent_label=None,
            selected_candidate_id=None,
            selected_candidate_name=None,
            output_name=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            model=self.model,
            prompt_sha256=prompt_sha256,
            candidate_order_seed=prepared.order_seed,
            error_type=error_type,
            decision_reason="upstream_error",
        )

    def _normalize_and_trim_messages(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[dict[str, str], ...]:
        if not messages:
            raise ValueError("messages 不能为空")

        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            raw_role = message.get("role", "")
            raw_content = message.get("content", "")
            if not isinstance(raw_role, str) or not isinstance(raw_content, str):
                raise ValueError(f"messages[{index}] 的 role/content 必须是字符串")
            role = raw_role.strip()
            content = raw_content.strip()
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"messages[{index}].role 仅支持 system/user/assistant/tool")
            if not content:
                raise ValueError(f"messages[{index}].content 不能为空")
            if role != "system":
                normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("忽略 system message 后没有剩余对话")
        if normalized[-1]["role"] != "user":
            raise ValueError("最后一条消息必须来自 user")

        current = dict(normalized[-1])
        current["content"] = self._truncate_content(
            current["content"], self.max_history_chars, marker=_LATEST_TRUNCATION_MARKER
        )
        remaining_chars = self.max_history_chars - len(current["content"])
        history_slots = max(0, self.max_history_messages - 1)
        needs_reference = self._needs_reference_context(current["content"])
        eligible = [
            message
            for message in normalized[:-1]
            if message["role"] == "user" or needs_reference
        ]
        recent = eligible[-history_slots:] if history_slots else []

        kept_reversed: list[dict[str, str]] = []
        for original in reversed(recent):
            if remaining_chars <= 0:
                break
            message = dict(original)
            limit = remaining_chars
            if message["role"] in {"assistant", "tool"}:
                limit = min(limit, self.max_assistant_history_chars)
            message["content"] = self._truncate_content(
                message["content"], limit, marker=_HISTORY_TRUNCATION_MARKER
            )
            kept_reversed.append(message)
            remaining_chars -= len(message["content"])
        return tuple([*reversed(kept_reversed), current])

    @staticmethod
    def _needs_reference_context(current_user_request: str) -> bool:
        return _REFERENCE_CONTEXT_PATTERN.search(current_user_request) is not None

    @staticmethod
    def _truncate_content(content: str, limit: int, *, marker: str) -> str:
        if len(content) <= limit:
            return content
        if limit <= len(marker):
            return content[:limit]
        available = limit - len(marker)
        head_size = (available * 2) // 3
        return content[:head_size] + marker + content[-(available - head_size) :]

    @staticmethod
    def _conversation_seed(messages: Sequence[Mapping[str, str]]) -> int:
        canonical = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return int.from_bytes(
            hashlib.sha256(canonical.encode("utf-8")).digest()[:8], "big"
        )

    def _present_candidates(self) -> tuple[PresentedCandidate, ...]:
        by_name = {candidate.name: candidate for candidate in self.candidates}
        ordered = [by_name[name] for name in _PRODUCTION_CANDIDATE_ORDER]
        ordered.extend(
            candidate for candidate in self.candidates if candidate.name not in _PRODUCTION_CANDIDATE_ORDER
        )
        return tuple(PresentedCandidate(candidate) for candidate in ordered)

    @staticmethod
    def _build_user_prompt(messages: Sequence[Mapping[str, str]]) -> str:
        payload: dict[str, Any] = {"current_user_request": messages[-1]["content"]}
        prior = [message for message in messages[:-1] if message["role"] == "user"]
        reference = [message for message in messages[:-1] if message["role"] != "user"]
        if prior:
            payload["prior_user_turns"] = prior
        if reference:
            payload["assistant_tool_reference"] = reference
        conversation = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return (
            f"<conversation_json>{conversation}</conversation_json>\n"
            "输出谓词结果： /no_think"
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def _parse_boundary_id(raw_output: str, allowed: frozenset[str]) -> str | None:
        normalized = unicodedata.normalize("NFKC", raw_output or "").strip()
        return normalized if normalized in allowed else None

    @staticmethod
    def _sum_usage(responses: Sequence[Any], field: str) -> int | None:
        values: list[int] = []
        for response in responses:
            usage = getattr(response, "usage", None)
            value = getattr(usage, field, None) if usage is not None else None
            if isinstance(value, int):
                values.append(value)
        return sum(values) if values else None

    def _boundary_probability(self, response: Any, selected_id: str) -> float | None:
        if self.min_route_probability is None:
            return None
        try:
            positions = response.choices[0].logprobs.content
        except (AttributeError, IndexError, TypeError):
            return None
        meaningful = [
            position
            for position in positions or ()
            if str(getattr(position, "token", "")).strip()
        ]
        generated = unicodedata.normalize(
            "NFKC", "".join(str(position.token) for position in meaningful)
        ).strip()
        if generated != selected_id or not meaningful:
            return None
        try:
            mean_logprob = sum(float(position.logprob) for position in meaningful) / len(meaningful)
        except (AttributeError, TypeError, ValueError):
            return None
        return round(min(max(math.exp(mean_logprob), 0.0), 1.0), 6)
