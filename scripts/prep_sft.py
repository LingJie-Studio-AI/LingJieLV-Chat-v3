import argparse, json, os, re, time, hashlib, zlib, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.presets import get_preset

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--src', default='')
p.add_argument('--out', default='')
p.add_argument('--format', choices=['auto', 'minimind', 'alpaca'], default='auto')
p.add_argument('--maxlen', type=int, default=1250)
p.add_argument('--max_n', type=int, default=0, help='flash 截取条数，0 表示不截断')
args = p.parse_args()

pr = get_preset(args.variant)
sys_prompt = pr['sys_prompt']
out = args.out or os.path.join(ROOT, pr['sft_dir'], 'sft_v4.jsonl')
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

SYS_OFF = sys_prompt + '思考模式：关闭。直接给出最终回答，禁止输出【思考】、【推理】或任何中间推理过程。'
SYS_ON = sys_prompt + '思考模式：开启。先在【思考】中推理，再用【回答】给出最终结论。'

TOOL_STRUCT = re.compile(r'(tool_call|function_call|<\|tool|"arguments"\s*:|\{\s*"name"\s*:)', re.I)
TOOL_NAMES = re.compile(r'(?i)(calculat\w*|arithmetic|\bmath\b|\bcalc\b|get_time|current_time|\bdate\b|\btime\b|\bclock\b|\bnow\b)')
TOOL_CN = re.compile(r'(计算器|计算工具|时间工具|日期工具|工具调用|调用工具|调用函数)')
TOOL_DROP = re.compile(r'(?i)(\bweather\b|web_search|\bsearch_web\b|\bbrowse\b|\bcrawl\b|\bhttp\b|\bapi\b|read_file|write_file|code_exec|python_exec|get_news)')


def norm(s): return re.sub(r'\s+', ' ', s).strip()


def sh(t):
    a = 0
    for g in (t[i:i + 3] for i in range(len(t) - 2)):
        a ^= zlib.crc32(g.encode())
    return a & ((1 << 64) - 1)


def ham(a, b): return bin(a ^ b).count('1')


def rewrite(u, a):
    a = re.sub(r'(?i)minimind', '凌界', a)
    a = re.sub(r'由\s*OpenAI\s*开发', '由凌界团队训练', a)
    return a


def detect_format(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        head = f.read(4096)
    return 'alpaca' if '"instruction"' in head or '"output"' in head else 'minimind'


# ---------------- Flash：Alpaca-GPT4-ZH 精简链路 ----------------
def prep_alpaca(src, out_path, max_n):
    raw = json.load(open(src, encoding='utf-8'))
    data = raw.get('default', raw) if isinstance(raw, dict) else raw
    seen, rows = set(), []
    for d in data:
        if not isinstance(d, dict):
            continue
        ins = (d.get('instruction') or '').strip()
        inp = (d.get('input') or '').strip()
        ans = (d.get('output') or '').strip()
        if not ins or not ans:
            continue
        if len(ins) + len(inp) > 400 or len(ans) > 700:
            continue
        key = hashlib.md5((ins + inp + ans).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        u = ins + ('\n' + inp if inp else '')
        rows.append({'system': SYS_OFF, 'prompt': u, 'response': rewrite(u, ans), 'src': 'alpaca'})
    random.Random(42).shuffle(rows)
    if max_n:
        rows = rows[:max_n]
    with open(out_path, 'w', encoding='utf-8') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'ALPACA written {len(rows)}', flush=True)


# ---------------- Pro：minimind conversations 清洗去重链路 ----------------
def prep_minimind(src, out_path, maxlen):
    IDX = {}

    def near_dup(h, thr=3):
        for b in range(4):
            for o in IDX.get((b, (h >> (b * 16)) & 0xFFFF), ()):
                if ham(h, o) <= thr:
                    return True
        return False

    def add(h):
        for b in range(4):
            IDX.setdefault((b, (h >> (b * 16)) & 0xFFFF), []).append(h)

    t0 = time.time(); n = keep = 0
    d_tool = d_len = d_bad = d_dup = 0; cot = 0; seen = set()
    with open(src, encoding='utf-8', errors='ignore') as f, open(out_path, 'w', encoding='utf-8') as w:
        for ln in f:
            n += 1
            if n % 200000 == 0:
                print(f'[{time.strftime("%H:%M:%S")}] lines={n} keep={keep} tool_drop={d_tool} dup={d_dup}', flush=True)
            try:
                d = json.loads(ln)
            except Exception:
                d_bad += 1; continue
            conv = d.get('conversations') or []
            us = [m.get('content') or '' for m in conv if m.get('role') == 'user']
            asst = [m for m in conv if m.get('role') == 'assistant']
            if not us or not asst:
                d_bad += 1; continue
            u = norm(us[0]); a = norm(asst[-1].get('content') or '')
            rc = (asst[-1].get('reasoning_content') or '').strip()
            if not a:
                d_bad += 1; continue
            body = u + a + rc
            if TOOL_STRUCT.search(body) or TOOL_CN.search(body):
                if not (TOOL_NAMES.search(body) and not TOOL_DROP.search(body)):
                    d_tool += 1; continue
            L = len(u) + len(a) + len(rc)
            if L > maxlen or L < 4:
                d_len += 1; continue
            a2 = rewrite(u, a); rc2 = rewrite(u, rc) if rc else ''
            if rc2:
                variants = [(SYS_ON, f'【思考】{rc2}\n【回答】{a2}', 'cot_on'),
                            (SYS_OFF, a2, 'cot_off')]
            else:
                variants = [(SYS_OFF, a2, 'sft')]
            for sysp, resp, src_tag in variants:
                key = hashlib.blake2b((sysp + '|' + u + '|' + resp).encode(), digest_size=16).hexdigest()
                if key in seen:
                    d_dup += 1; continue
                if len(resp) >= 80:
                    h = sh(u + '|' + resp)
                    if near_dup(h):
                        d_dup += 1; continue
                    add(h)
                seen.add(key)
                w.write(json.dumps({'system': sysp, 'prompt': u, 'response': resp, 'src': src_tag}, ensure_ascii=False) + '\n')
                keep += 1
                if rc2: cot += 1
    print(f'DONE keep={keep} cot={cot} lines={n} tool_drop={d_tool} len={d_len} dup={d_dup} bad={d_bad} sec={round(time.time()-t0,1)}', flush=True)


def main():
    fmt = args.format
    if fmt == 'auto':
        fmt = detect_format(args.src) if args.src else 'minimind'
    if fmt == 'alpaca':
        prep_alpaca(args.src, out, args.max_n or 12000)
    else:
        prep_minimind(args.src, out, args.maxlen)


main()
