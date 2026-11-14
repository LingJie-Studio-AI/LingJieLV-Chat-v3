# -*- coding: utf-8 -*-
"""凌界本地推理网页服务（标准库实现，无第三方 Web 依赖）。

用法（在仓库根目录）：
    python3 web/serve.py            # 默认 http://127.0.0.1:8000
    python3 web/serve.py --port 9000

网页支持：上传 mm_pack.pt 权重（可再传 tokenizer.json）、文字提问、上传图片、
KV 缓存开关、多轮对话与清空。权重只在本机内存加载，不上传任何外部服务器。
"""
import argparse, base64, io, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from PIL import Image  # noqa: E402
from src.kv_infer import load_pack, KVChat  # noqa: E402

WEB_DIR = os.path.join(REPO_ROOT, 'web')
STATE = {'chat': None, 'variant': None, 'device': None}
LOCK = threading.Lock()


def parse_multipart(body, boundary):
    """返回 [(name, filename, content), ...]。"""
    delim = b'--' + boundary.encode('latin1')
    out = []
    for part in body.split(delim):
        if not part or part in (b'--', b'--\r\n'):
            continue
        part = part.lstrip(b'\r\n')
        sep = part.find(b'\r\n\r\n')
        if sep < 0:
            continue
        headers = part[:sep].decode('latin1')
        content = part[sep + 4:]
        if content.endswith(b'\r\n'):
            content = content[:-2]
        name = filename = None
        for line in headers.split('\r\n'):
            if line.lower().startswith('content-disposition'):
                for seg in line.split(';'):
                    seg = seg.strip()
                    if seg.startswith('name='):
                        name = seg[5:].strip('"')
                    elif seg.startswith('filename='):
                        filename = seg[9:].strip('"')
        out.append((name, filename, content))
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype='application/json; charset=utf-8'):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(WEB_DIR, 'index.html'), 'rb') as f:
                self._send(f.read(), ctype='text/html; charset=utf-8')
        elif self.path == '/api/status':
            self._send({'loaded': STATE['chat'] is not None, 'variant': STATE['variant'],
                        'device': STATE['device']})
        else:
            self._send({'error': 'not found'}, 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n)
        try:
            if self.path == '/api/load':
                return self._load(body)
            if self.path == '/api/chat':
                return self._chat(body)
            if self.path == '/api/reset':
                with LOCK:
                    if STATE['chat']:
                        STATE['chat'].reset()
                return self._send({'ok': True})
            self._send({'error': 'not found'}, 404)
        except Exception as e:
            self._send({'error': f'{type(e).__name__}: {e}'}, 500)

    def _load(self, body):
        ct = self.headers.get('Content-Type', '')
        boundary = ct.split('boundary=')[-1]
        files = {}
        for name, filename, content in parse_multipart(body, boundary):
            if filename:
                files[name] = (filename, content)
        if 'pack' not in files:
            return self._send({'error': '缺少权重文件 mm_pack.pt'}, 400)

        tmp = os.path.join(WEB_DIR, '.tmp')
        os.makedirs(tmp, exist_ok=True)
        pack_path = os.path.join(tmp, files['pack'][0])
        with open(pack_path, 'wb') as f:
            f.write(files['pack'][1])
        tok_path = None
        if 'tokenizer' in files:
            tok_path = os.path.join(tmp, files['tokenizer'][0])
            with open(tok_path, 'wb') as f:
                f.write(files['tokenizer'][1])

        with LOCK:
            m, tok, sp, variant = load_pack(pack_path, tok_path)
            chat = KVChat(m, tok, sp)
            lt, la = m.lm.stats()
            pack_total = sum(p.numel() for p in m.parameters())
            STATE.update(chat=chat, variant=variant, device=str(chat.device))
        self._send({'ok': True, 'variant': variant, 'device': str(chat.device),
                    'lm_total': round(lt / 1e6, 2), 'lm_active': round(la / 1e6, 2),
                    'pack_total': round(pack_total / 1e6, 2)})

    def _chat(self, body):
        req = json.loads(body)
        if not STATE['chat']:
            return self._send({'error': '请先上传模型权重'}, 400)
        image = None
        if req.get('image'):
            raw = req['image'].split(',', 1)[-1]
            image = Image.open(io.BytesIO(base64.b64decode(raw)))
        max_new = int(req.get('max_new', 200))
        use_kv = bool(req.get('use_kv', True))
        with LOCK:
            t0 = time.time()
            answer, ids = STATE['chat'].generate(
                req['message'], image=image, max_new=max_new, use_kv=use_kv)
            dt = time.time() - t0
        self._send({'ok': True, 'answer': answer, 'tokens': len(ids),
                    'elapsed': round(dt, 2),
                    'tok_s': round(len(ids) / dt, 1) if dt and ids else 0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--no-browser', action='store_true')
    args = p.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://{args.host}:{args.port}'
    print(f'凌界推理服务已启动：{url}  （Ctrl+C 停止）')
    if not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    srv.serve_forever()


if __name__ == '__main__':
    main()
