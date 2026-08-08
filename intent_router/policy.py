from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


_STOCK_POSITIVE_IDS = frozenset({"StockInfo", "StockRoute"})
_PRODUCT_POSITIVE_IDS = frozenset({"Ecommerce", "ProductRoute", "RetailFind"})

_STOCK_APP = re.compile(
    r"(?:股票|炒股|看盘|盯盘).{0,10}(?:软件|app|应用)|"
    r"(?:软件|app|应用).{0,10}(?:股票|炒股|看盘|盯盘)",
    re.IGNORECASE,
)
_APP_AS_OBJECT = re.compile(
    r"(?:推荐|换|哪个好|哪个.{0,6}好|好用|流畅|比较|介绍|做什么|支持模拟|"
    r"(?:什么|哪些|有).{0,16}(?:软件|app|应用))",
    re.IGNORECASE,
)
_ATTACHED_APP_TASK = re.compile(
    r"(?:看|查|查询|了解).+?(?:再|同时|顺便|然后).+?(?:软件|app|应用)",
    re.IGNORECASE,
)
_RETAIL_DISCOVERY_ACTION = re.compile(r"(?:推荐|买|购买|找|搜|搜索|来一单|下单)")
_NON_RETAIL_OBJECT = re.compile(
    r"(?:药品|乳膏|药膏|处方药|酒店|住宿|民宿|摄影工作室|维修服务|"
    r"家政服务|律师|培训课程|保险合同|房子|住宅)"
)
_NON_RETAIL_THEME = re.compile(r"(?:汽车|轿车|摩托车|房产|旅行|保险)")
_ORDINARY_GOOD_AFTER_THEME = re.compile(
    r"(?:配件|用品|模型|玩具|收纳|箱|包|袋|套|垫|架|膜|线|充电器|"
    r"支架|文件|书|麦克风|显示器|靠垫|插头|柜|窝)"
)
_STOCK_INFORMATION_OBJECT = re.compile(
    r"(?:股票|股市|大盘|股价|行情|指数|板块|成交量|股票代码|股东人数|"
    r"上市公司公告|股票公告)"
)
_STOCK_INFORMATION_ACTION = re.compile(
    r"(?:查|查询|看|查看|打开|找|了解|分析|概览|多少|什么|怎么样|情况|现价|"
    r"最新|今日|今天|历史|过去|近期|请|给我)"
)
_STOCK_DECISION_OR_RESEARCH = re.compile(
    r"(?:预测|推测|预判|未来|明天|下周|目标价|买入|卖出|加仓|减仓|推荐股票|"
    r"选股|龙头股|赛道|持股|减持|增持|股权|ETF|基金|债券|期货|外汇)"
)
_STOCK_TEXT_OR_NEGATION = re.compile(
    r"(?:不要查|不查|别查|不是查|无需查|分析句子|主谓宾|翻译|改写|文案|笑话)"
)
_EXPLICIT_STOCK_ADVICE_TASK = re.compile(
    r"(?:预测|推测|预判|预计).{0,20}(?:股票|A股|大盘|指数|走势|涨跌)|"
    r"(?:股票|A股|大盘|指数).{0,20}(?:未来|明天|下周|涨跌|目标价)"
)
_PRODUCT_INFORMATION_ATTRIBUTE = re.compile(
    r"(?:参数|容量|续航|尺寸|重量|材质|功能|多少钱|价格|评价|像素|适合吗|怎么样)"
)
_PRODUCT_TEXT_OR_FULFILMENT = re.compile(
    r"(?:分析句子|主谓宾|翻译|改写|文案|刚才推荐|你.{0,6}推荐|这些|那些|"
    r"这[二三四五六七八九十\d]+款|维修|服务|师傅|课程|酒店|房子|药品|乳膏)"
)
_PHYSICAL_DISCOVERY = re.compile(
    r"(?:推荐|搜|搜索|找|看下|看看|看一下|想买|要买).{0,20}"
    r"(?:手机|电脑|平板|电视|冰箱|空调|相机|耳机|手表|键盘|鼠标|显示器|"
    r"电池|电源|充电器|存储卡|NFC卡|定位器|书|教材|文件袋|保险柜|麦克风|"
    r"靠垫|插头|路由器|投影仪|净化器|洗衣机|牙膏|衣|裤|鞋|包)"
)


@dataclass(frozen=True)
class FusionResult:
    """Backend decision produced from two independent boundary classifiers."""

    output_name: str
    intent_label: str | None
    reason: str


def _current_user_request(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _prior_user_turns(messages: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    turns = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user"
    ]
    return tuple(turns[:-1])


def _is_stock_app(text: str) -> bool:
    return _STOCK_APP.search(text) is not None


def _app_is_final_object(text: str) -> bool:
    return _is_stock_app(text) and _APP_AS_OBJECT.search(text) is not None


def _ordinary_good_follows_non_retail_theme(text: str) -> bool:
    for match in _NON_RETAIL_THEME.finditer(text):
        if _ORDINARY_GOOD_AFTER_THEME.search(text[match.end() :]):
            return True
    return False


def _stock_veto(
    text: str,
    product_id: str,
    has_prior_user_turn: bool,
) -> tuple[str, str] | None:
    if product_id == "MultiProduct" and _RETAIL_DISCOVERY_ACTION.search(text):
        return "NoAvailable", "mixed_stock_and_product"
    if has_prior_user_turn and re.search(
        r"(?:两个|两项|两件).{0,12}(?:一起|都要|都做)|(?:一起做|两个一起)",
        text,
    ):
        return "NoAvailable", "explicit_history_merge"
    if re.search(
        r"(?:翻译|改写|润色|写成|文案|英文).{0,60}(?:股票|股价|大盘|行情|指数)|"
        r"(?:股票|股价|大盘|行情|指数).{0,60}(?:翻译|改写|润色|文案|英文)",
        text,
        re.IGNORECASE,
    ):
        return "NoAvailable", "quoted_or_transformed_text"
    if _app_is_final_object(text) and _ATTACHED_APP_TASK.search(text) is None:
        return "StockOther", "stock_application_is_object"
    if re.search(
        r"(?:推测|预测|预判|预计).{0,18}(?:明天|下周|未来|后市|走势|涨跌|价格)|"
        r"(?:明天|下周|未来|后市).{0,24}(?:会|可能|涨|跌|走势|价格|点位)",
        text,
    ):
        return "StockAdvice", "future_prediction"
    if re.search(
        r"(?:如果|假如|假设|一旦).{0,35}(?:大盘|股价|股票|指数|板块).{0,35}"
        r"(?:会|可能|先|涨|跌|下挫|影响)",
        text,
    ):
        return "StockAdvice", "conditional_prediction"
    if re.search(
        r"(?:龙头股).{0,10}(?:是哪个|有哪些|选哪|推荐)|"
        r"(?:选|推荐|买入|卖出|加仓|减仓).{0,12}(?:股票|个股|龙头股)",
        text,
    ):
        return "StockAdvice", "security_selection"
    if re.search(
        r"(?:属于|是不是|是否是).{0,15}(?:龙头|赛道)|"
        r"(?:有|是否|有没有).{0,8}(?:减持|增持)|"
        r"(?:业绩|经营).{0,35}股价.{0,30}(?:同时|先|后|因果)",
        text,
    ):
        return "StockResearch", "company_research"
    if re.search(
        r"(?:ETF|基金).{0,20}(?:选哪些|选哪个|哪只|标的|净值|价格|价位|收益)",
        text,
        re.IGNORECASE,
    ) and not re.search(
        r"(?:指数|板块|大盘|市场).{0,16}(?:点位|位置|振幅|行情)", text
    ):
        return "FinanceOther", "other_financial_instrument"
    if re.search(
        r"股价.{0,16}(?:反应|反映).{0,16}(?:中报|年报|业绩|预期)", text
    ):
        return "StockAdvice", "valuation_judgement"

    request_signal = re.search(
        r"(?:查|查询|看|瞅|了解|分析|多少|什么|怎么|怎样|为何|为什么|吗|呢|？|"
        r"给我|帮我|打开|概览|表现|如何|原因|哪些|啥|摘要|股价|行情|走势|价格|"
        r"点位|K线|收盘|开盘|涨|跌|市值|成交|代码|指标|股息率|市盈率|市净率|"
        r"分红|财报|营收|股东人数)",
        text,
    )
    if request_signal is None and re.search(r"(?:资金|市场|板块|股票|股价)", text):
        return "NoRequest", "declarative_market_text"
    if not has_prior_user_turn and re.fullmatch(r"是.{1,24}", text):
        return "NoRequest", "request_fragment"
    return None


def _product_veto(
    text: str,
    stock_id: str,
    has_prior_user_turn: bool,
) -> tuple[str, str] | None:
    if _app_is_final_object(text):
        return "StockOther", "stock_application_is_object"
    if re.search(
        r"(?:第[一二三四五六七八九十\d]+(?:个|款|件)|就这个|就它|选这个).{0,18}"
        r"(?:支付|付款|下单|提交|加入购物车|加购)",
        text,
    ):
        return "ProductOther", "post_selection_fulfilment"
    if _NON_RETAIL_OBJECT.search(text):
        return "NonRetail", "non_retail_object"
    if (
        re.search(r"(?:汽车|轿车|摩托车|整车)", text)
        and not _ordinary_good_follows_non_retail_theme(text)
        and re.search(r"(?:不是真车|非真车)", text) is None
    ):
        return "NonRetail", "vehicle_object"
    if re.search(r"(?:找到|看中|拍了|买了|我有|现有)", text) and re.search(
        r"(?:合适|怎么样|值不值|值得|正品|适合|认为|价格|像素|功能)", text
    ):
        return "ProductInfo", "known_product_information"
    if re.search(
        r"(?:哪个|什么|哪些).{0,8}品牌.{0,10}(?:好|有名|靠谱)|"
        r"品牌.{0,10}(?:哪个好|有哪些)",
        text,
    ) and not re.search(
        r"(?:预算|用途|场景|平台|京东|淘宝|天猫|拼多多|搜|找)", text
    ):
        return "ProductInfo", "general_brand_knowledge"
    if not has_prior_user_turn and re.search(
        r"(?:这些|那些|这[二三四五六七八九十\d]+款|那[二三四五六七八九十\d]+款)",
        text,
    ):
        return "NoRequest", "unresolved_reference"
    if not has_prior_user_turn and (
        re.match(r"^(?:是买|是).{0,35}$", text) or re.match(r"^的", text)
    ):
        return "NoRequest", "request_fragment"
    if re.fullmatch(
        r"(?:给我|帮我|请)?(?:推荐|推荐一下|买|购买|看看|看一下|搜|搜索|找)"
        r"(?:给我|一下)?[啊呀吗吧]?",
        text,
    ) or re.fullmatch(r"(?:有|还有)?什么(?:好物|东西)?推荐[啊呀吗吧]?", text):
        return "NoRequest", "missing_product_object"
    if re.fullmatch(r"你(?:买|购买|找|搜|推荐).{0,8}", text):
        return "NoRequest", "malformed_request"
    if (
        stock_id == "StockAdvice"
        and re.search(r"应该买哪个|该买哪个|买哪一个", text)
        and re.search(
            r"(?:手机|电脑|平板|电视|冰箱|空调|相机|耳机|手表|汽车|商品|型号|款)",
            text,
        )
        is None
    ):
        return "StockAdvice", "security_selection"
    request_signal = re.search(
        r"(?:我想|我要|我需要|给我|帮我|请|买|购买|推荐|搜|搜索|找|看|查看|"
        r"查|打开|来一单|下单|多少钱|价格|怎么样|适合|好不好|功能|参数|图片|"
        r"评价|对比|比较|介绍|讲|有吗|有没有|吗|呢|？|选|挑|要个|来个|整个|"
        r"有啥|哪里|链接|直接给|筛选|送到|配|值得|应该|怎么|哪款|哪些|哪个|什么)",
        text,
    )
    if (
        not has_prior_user_turn
        and len(text) <= 24
        and request_signal is None
        and re.search(r"[A-Za-z0-9]", text)
    ):
        return "NoRequest", "bare_product_name"
    if (
        re.search(r"的那种[啊呀吗吧]?$", text)
        and re.search(r"(?:给我|帮我|推荐|搜|搜索|找|想买|要买|有没有|有没)", text)
        is None
    ):
        return "NoRequest", "request_fragment"
    if (
        re.search(r"选.{1,12}还是.{1,12}(?:好|合适)[啊呀吗吧？?]?$", text)
        and re.search(r"(?:买|购买|推荐|预算|平台|京东|淘宝|天猫|拼多多)", text)
        is None
    ):
        return "ProductInfo", "general_option_knowledge"
    if (
        re.search(
            r"(?:只有|一共|总共).{0,8}(?:三|四|五|六|七|八|九|十|\d+)(?:个|件|种|类)",
            text,
        )
        and len(re.findall(r"(?:一个|一台|一件|一款|一种)", text)) >= 2
        and _RETAIL_DISCOVERY_ACTION.search(text)
    ):
        return "MultiProduct", "multiple_unrelated_products"
    if (
        re.search(
            r"(?:这[二三四五六七八九十\d]+款|那[二三四五六七八九十\d]+款).{0,40}"
            r"(?:价格|多少钱)",
            text,
        )
        and not has_prior_user_turn
    ):
        return "NoRequest", "unresolved_price_comparison"
    return None


def _recover_route(text: str, stock_id: str, product_id: str) -> FusionResult | None:
    stock_recoverable = stock_id in {"NoStock", "StockOperation", "StockResearch"}
    if (
        stock_recoverable
        and _STOCK_INFORMATION_OBJECT.search(text)
        and _STOCK_INFORMATION_ACTION.search(text)
        and _STOCK_DECISION_OR_RESEARCH.search(text) is None
        and _STOCK_TEXT_OR_NEGATION.search(text) is None
    ):
        return FusionResult("StockInfo", "SearchStockQuotes", "explicit_stock_information")
    if (
        stock_id in {"NoStock", "StockOperation", "StockResearch", "StockAdvice"}
        and re.search(
            r"(?:分析|了解|概览).{0,18}(?:市场|证券|资金|炒作)|"
            r"(?:市场|证券|资金).{0,18}(?:分析|概览|怎么样)",
            text,
        )
        and _STOCK_TEXT_OR_NEGATION.search(text) is None
    ):
        return FusionResult("StockInfo", "SearchStockQuotes", "stock_market_discussion")
    if (
        product_id in {"NoProduct", "NonRetail", "MultiProduct"}
        and re.match(r"^(?:购买|买|下单)", text)
        and not re.search(
            r"(?:股票|基金|ETF|保险|药品|乳膏|汽车|整车|房产|机票|酒店|服务)",
            text,
            re.IGNORECASE,
        )
    ):
        return FusionResult("Ecommerce", "RecommendProduct", "explicit_retail_purchase")
    if product_id in {"NoProduct", "ProductKnowledge"} and re.search(
        r"(?:给我|帮我|找|要).{0,10}购买链接", text
    ):
        return FusionResult("Ecommerce", "RecommendProduct", "explicit_retail_discovery")
    if product_id in {"NoProduct", "ProductKnowledge"} and re.search(
        r"(?:多大|多少).{0,12}(?:适合|够用).{0,16}(?:平|人|房|客厅|卧室)", text
    ):
        return FusionResult("Ecommerce", "RecommendProduct", "scenario_based_selection")
    if (
        product_id in {"NoProduct", "ProductKnowledge", "NonRetail"}
        and _PHYSICAL_DISCOVERY.search(text)
        and _PRODUCT_INFORMATION_ATTRIBUTE.search(text) is None
        and _PRODUCT_TEXT_OR_FULFILMENT.search(text) is None
        and not (
            stock_id == "StockAdvice" and _EXPLICIT_STOCK_ADVICE_TASK.search(text)
        )
    ):
        return FusionResult("Ecommerce", "RecommendProduct", "explicit_retail_discovery")
    return None


def _virtual_output(stock_id: str, product_id: str) -> str:
    stock_mapping = {
        "StockAdvice": "StockAdvice",
        "StockResearch": "StockResearch",
        "StockOperation": "StockOther",
        "StockOther": "StockOther",
        "OtherFinance": "FinanceOther",
        "FinanceOther": "FinanceOther",
    }
    product_mapping = {
        "ProductKnowledge": "ProductInfo",
        "ProductInfo": "ProductInfo",
        "ProductFollowup": "ProductOther",
        "ProductOther": "ProductOther",
        "NonRetail": "NonRetail",
        "MultiProduct": "MultiProduct",
    }
    return stock_mapping.get(stock_id) or product_mapping.get(product_id) or "NoAvailable"


def _is_chitchat(text: str) -> bool:
    return re.fullmatch(
        r"(?:你好|您好|嗨|哈喽|hello|hi|早上好|下午好|晚上好|谢谢|感谢|再见|拜拜|"
        r"辛苦了|很高兴见到你|你好吗)[啊呀哦呢！!。,.， ]*",
        text,
        re.IGNORECASE,
    ) is not None


def fuse_boundary_decisions(
    messages: Sequence[Mapping[str, str]],
    stock_id: str,
    product_id: str,
) -> FusionResult:
    """Fuse independent closed-set decisions, then apply generic safety checks."""

    text = _current_user_request(messages)
    has_prior = bool(_prior_user_turns(messages))
    stock_positive = stock_id in _STOCK_POSITIVE_IDS
    product_positive = product_id in _PRODUCT_POSITIVE_IDS

    if stock_positive and not product_positive:
        veto = _stock_veto(text, product_id, has_prior)
        if veto is None:
            return FusionResult("StockInfo", "SearchStockQuotes", "stock_boundary")
        output_name, reason = veto
        return FusionResult(output_name, None, reason)

    if product_positive and not stock_positive:
        veto = _product_veto(text, stock_id, has_prior)
        if veto is None:
            return FusionResult("Ecommerce", "RecommendProduct", "product_boundary")
        output_name, reason = veto
        return FusionResult(output_name, None, reason)

    recovered = _recover_route(text, stock_id, product_id)
    if recovered is not None:
        return recovered
    if not stock_positive and not product_positive and _is_chitchat(text):
        return FusionResult("ChitChat", None, "chitchat")
    reason = "positive_conflict" if stock_positive and product_positive else "no_positive_boundary"
    return FusionResult(_virtual_output(stock_id, product_id), None, reason)
