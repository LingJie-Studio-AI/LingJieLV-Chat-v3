# -*- coding: utf-8 -*-
"""凌界工具运行时：安全表达式求值 + 真实取时，以及 agent 生成循环。"""
import ast, json, operator, datetime as dt
from zoneinfo import ZoneInfo
import torch
import torch.nn.functional as F

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError('不支持的表达式')

def run_calculator(expression):
    tree = ast.parse(expression, mode='eval')
    v = safe_eval(tree)
    if isinstance(v, float) and v == int(v):
        v = int(v)
    return {'result': v}

def run_get_time(timezone='Asia/Shanghai'):
    tz = ZoneInfo(timezone)
    now = dt.datetime.now(tz)
    wk = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][now.weekday()]
    return {'datetime': now.strftime('%Y-%m-%d %H:%M:%S'), 'weekday': wk}

TOOL_FNS = {'calculator': lambda a: run_calculator(a.get('expression', '')),
            'get_time': lambda a: run_get_time(a.get('timezone', 'Asia/Shanghai'))}

class Agent:
    def __init__(self, model, tok, system):
        self.m = model; self.t = tok; self.system = system
        # 兼容带视觉塔的 MiniMoEVL（语言模型在 .lm）与纯文本 MiniMoE
        self.lm = getattr(model, 'lm', model)
        self.ids = [tok.sys] + tok.enc(system) + [tok.end]

    def _step(self, ids):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h = self.lm.forward_emb(self.lm.tok_emb(torch.tensor([ids[-768:]]).cuda()))[0]
                lg = F.linear(h, self.lm.tok_emb.weight)[0, -1].float()
        return int(torch.argmax(lg))

    @staticmethod
    def extract_call(text):
        # 从首个 { 开始用 raw_decode 提取完整 JSON，容忍中间不完整/跨行
        i = text.find('{')
        while i != -1:
            try:
                obj, end = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(obj, dict) and 'name' in obj:
                    return obj
                i = text.find('{', i + end)
            except json.JSONDecodeError:
                return None
        return None

    def run(self, prompt, max_new=500, max_calls=4):
        self.ids += [self.t.user] + self.t.enc(prompt) + [self.t.end, self.t.asst]
        out = []
        phase = 'call'
        n_calls = 0
        consumed = 0  # 已处理到的 token 位置，之后只在新生成片段里找调用
        for _ in range(max_new):
            nid = self._step(self.ids)
            if nid == self.t.end:
                break
            self.ids.append(nid); out.append(nid)
            if phase == 'call':
                new_text = self.t.t.decode(out[consumed:])
                if '【工具调用】' in new_text:
                    call = self.extract_call(new_text.split('【工具调用】', 1)[1])
                    if call is not None:
                        try:
                            result = TOOL_FNS[call['name']](call.get('arguments', {}))
                        except Exception as e:
                            result = {'error': str(e)}
                        inj = f'\n【工具结果】\n{json.dumps(result, ensure_ascii=False)}\n'
                        inj_ids = self.t.enc(inj)
                        self.ids += inj_ids; out += inj_ids
                        consumed = len(out)  # 本次调用已消费，后续只看新片段
                        n_calls += 1
                        # 错误且未超调用上限：保持调用态，让模型重试；成功或超限才进入回答
                        if 'error' not in result or n_calls >= max_calls:
                            phase = 'answer'
        return self.t.t.decode(out)
