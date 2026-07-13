"""Generate a static HTML report from session state.

Reads a session_state.json and produces a self-contained (mostly) HTML dashboard
with figure cards, detail modals, and a chatbox that generates copy-pasteable
CLI commands. No backend server needed — open the HTML in a browser.

Usage:
    python -m geoseg.generate_report runs/sessions/batch_20260527.json
    python -m geoseg.generate_report runs/sessions/batch_20260527.json \
        --output=runs/reports/my_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from geoseg.session_state import load_session, FigureStatus


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def _make_thumbnail(img_path: str, max_size: int = 400) -> str | None:
    """Return base64 data URI of a thumbnail, or None if PIL unavailable."""
    try:
        from PIL import Image

        p = Path(img_path)
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        img.thumbnail((max_size, max_size))
        thumb_path = Path("/tmp") / f"geoseg_thumb_{p.stem}.jpg"
        img.save(thumb_path, "JPEG", quality=75)
        data = thumb_path.read_bytes()
        thumb_path.unlink(missing_ok=True)
        b64 = base64.b64encode(data).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _img_src(path: str | None, thumbnail: bool = False) -> str:
    """Return image src: base64 thumbnail if possible, else file:// path."""
    if not path:
        return ""
    if thumbnail:
        thumb = _make_thumbnail(path)
        if thumb:
            return thumb
    p = Path(path).resolve()
    return f"file://{p}"


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
    *{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
    body{background:#f5f5f7;color:#1d1d1f}
    .header{background:#fff;border-bottom:1px solid #e5e5e5;padding:16px 24px;position:sticky;top:0;z-index:40}
    .header-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;justify-content:space-between}
    .logo{display:flex;align-items:center;gap:10px}
    .logo-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#1e40af,#06b6d4);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:14px}
    .logo-text h1{font-size:16px;font-weight:600}
    .logo-text p{font-size:11px;color:#86868b}
    .stats{max-width:1200px;margin:0 auto;padding:20px 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
    .stat-card{background:#fff;border-radius:12px;padding:16px;border:1px solid #e5e5e5}
    .stat-label{font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.5px}
    .stat-value{font-size:28px;font-weight:700;margin-top:4px}
    .stat-value.exported{color:#10b981}
    .stat-value.segmented{color:#f59e0b}
    .stat-value.skipped{color:#9ca3af}
    .stat-value.avg{color:#3b82f6}
    .filters{max-width:1200px;margin:0 auto;padding:0 24px 16px;display:flex;gap:8px;flex-wrap:wrap}
    .filter-btn{padding:6px 14px;border-radius:20px;border:1px solid #e5e5e5;background:#fff;font-size:13px;cursor:pointer;transition:all .15s}
    .filter-btn:hover{border-color:#d1d5db}
    .filter-btn.active{background:#1d1d1f;color:#fff;border-color:#1d1d1f}
    .grid{max-width:1200px;margin:0 auto;padding:0 24px 40px;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
    .card{background:#fff;border-radius:14px;border:1px solid #e5e5e5;overflow:hidden;cursor:pointer;transition:all .2s}
    .card:hover{transform:translateY(-2px);box-shadow:0 8px 30px -10px rgba(0,0,0,.12)}
    .card.warn{border-color:#fbbf24;box-shadow:0 0 0 1px #fef3c7}
    .card.skip{opacity:.55}
    .card-img{position:relative;height:180px;background:#f0f0f2;display:flex;align-items:center;justify-content:center;overflow:hidden}
    .card-img img{width:100%;height:100%;object-fit:cover}
    .card-img .no-img{color:#86868b;font-size:13px}
    .card-badge{position:absolute;top:10px;right:10px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;color:#fff;backdrop-filter:blur(8px)}
    .card-badge.exported{background:rgba(16,185,129,.9)}
    .card-badge.segmented{background:rgba(245,158,11,.9)}
    .card-badge.skipped{background:rgba(156,163,175,.9)}
    .card-badge.error{background:rgba(239,68,68,.9)}
    .card-score{position:absolute;bottom:10px;left:10px;padding:3px 8px;border-radius:6px;font-size:12px;font-weight:500;font-family:monospace}
    .card-score.good{background:rgba(0,0,0,.55);color:#fff}
    .card-score.warn{background:rgba(245,158,11,.85);color:#fff}
    .card-score.bad{background:rgba(239,68,68,.8);color:#fff}
    .card-body{padding:14px}
    .card-title{display:flex;align-items:start;justify-content:space-between;gap:8px}
    .card-title h3{font-size:14px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .card-title p{font-size:12px;color:#86868b;margin-top:2px}
    .card-engine{font-size:11px;font-weight:500;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:6px;white-space:nowrap}
    .card-meta{display:flex;align-items:center;gap:12px;margin-top:10px;padding-top:10px;border-top:1px solid #f3f4f6;font-size:12px;color:#6b7280}
    .card-meta svg{width:14px;height:14px;color:#9ca3af;flex-shrink:0}
    .card-alert{display:flex;align-items:center;gap:5px;margin-top:8px;font-size:12px;color:#d97706}
    .card-alert svg{width:14px;height:14px}
    .modal{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:50;display:none;align-items:center;justify-content:center;padding:20px}
    .modal.show{display:flex}
    .modal-box{background:#fff;border-radius:16px;max-width:900px;width:100%;max-height:90vh;overflow:hidden;display:flex;flex-direction:column;animation:modalIn .2s ease}
    @keyframes modalIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
    .modal-header{padding:16px 20px;border-bottom:1px solid #f3f4f6;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
    .modal-header h2{font-size:16px;font-weight:600}
    .modal-header p{font-size:13px;color:#86868b;margin-top:2px}
    .modal-close{width:32px;height:32px;border-radius:8px;border:none;background:none;cursor:pointer;display:flex;align-items:center;justify-content:center}
    .modal-close:hover{background:#f3f4f6}
    .modal-body{padding:20px;overflow-y:auto;flex:1}
    .compare{ display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
    .compare-box{position:relative}
    .compare-box p{font-size:11px;font-weight:600;color:#86868b;margin-bottom:6px}
    .compare-box .img-wrap{aspect-ratio:16/10;background:#f0f0f2;border-radius:10px;overflow:hidden;display:flex;align-items:center;justify-content:center}
    .compare-box .img-wrap img{width:100%;height:100%;object-fit:contain}
    .compare-box .img-wrap .no-img{color:#86868b;font-size:13px}
    .meta-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
    .meta-item{background:#f9fafb;border-radius:8px;padding:12px}
    .meta-item p:first-child{font-size:11px;color:#86868b}
    .meta-item p:last-child{font-size:14px;font-weight:600;margin-top:2px}
    .export-box{border:1px solid #e5e5e5;border-radius:10px;padding:14px}
    .export-box p{font-size:11px;font-weight:600;color:#86868b;margin-bottom:8px}
    .export-file{display:flex;align-items:center;gap:6px;font-size:13px;color:#4b5563;margin-bottom:4px}
    .export-file svg{width:16px;height:16px;color:#9ca3af}
    /* Chatbox */
    .chatbox{margin-top:16px;border:1px solid #e5e5e5;border-radius:10px;overflow:hidden}
    .chatbox-header{padding:10px 14px;background:#fafafa;border-bottom:1px solid #e5e5e5;font-size:13px;font-weight:600;color:#4b5563}
    .chatbox-body{padding:14px;max-height:200px;overflow-y:auto}
    .chat-msg{margin-bottom:10px}
    .chat-msg.user{text-align:right}
    .chat-msg.user .bubble{background:#1d1d1f;color:#fff}
    .chat-msg.system{text-align:left}
    .chat-msg.system .bubble{background:#f3f4f6;color:#374151}
    .chat-msg .bubble{display:inline-block;padding:8px 12px;border-radius:12px;font-size:13px;max-width:80%;line-height:1.4}
    .chatbox-input{display:flex;gap:8px;padding:10px 14px;border-top:1px solid #e5e5e5;background:#fafafa}
    .chatbox-input input{flex:1;padding:8px 12px;border:1px solid #d1d5db;border-radius:8px;font-size:13px;outline:none}
    .chatbox-input input:focus{border-color:#3b82f6}
    .chatbox-input button{padding:8px 16px;background:#1d1d1f;color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:500;cursor:pointer}
    .chatbox-input button:hover{background:#374151}
    .cli-cmd{background:#1e1e1e;color:#d4d4d4;padding:10px 14px;border-radius:8px;font-family:'SF Mono',monospace;font-size:12px;margin-top:8px;display:flex;align-items:center;justify-content:space-between;gap:10px}
    .cli-cmd code{word-break:break-all}
    .copy-btn{background:#374151;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;white-space:nowrap}
    .copy-btn:hover{background:#4b5563}
    .copy-btn.copied{background:#10b981}
"""


def _generate_card(entry: dict[str, Any], idx: int) -> str:
    """Generate HTML for one figure card."""
    status = entry.get("status", "pending")
    fig_id = entry["figure_id"]
    source = entry["source_path"]
    classification = entry.get("classification") or {}
    segmentation = entry.get("segmentation") or {}
    skip_reason = entry.get("skip_reason", "")

    # Determine card style
    card_class = "card"
    badge_class = status
    score = segmentation.get("quality_score")
    if status == "segmented":
        card_class += " warn"
    if status == "skipped":
        card_class += " skip"

    # Score display
    score_html = ""
    if score is not None:
        score_cls = "good" if score >= 0.75 else "warn" if score >= 0.6 else "bad"
        score_html = f'<span class="card-score {score_cls}">{score:.2f}</span>'

    # Image
    img_src = _img_src(source, thumbnail=True)
    img_html = (
        f'<img src="{img_src}" alt="{fig_id}" loading="lazy">'
        if img_src
        else '<span class="no-img">No image</span>'
    )

    # Overlay preview (if available)
    overlay_path = segmentation.get("overlay_path")
    overlay_src = _img_src(overlay_path, thumbnail=True) if overlay_path else None
    if overlay_src:
        img_html = f'<img src="{overlay_src}" alt="{fig_id} overlay" loading="lazy">'

    # Engine tag
    engine = segmentation.get("engine", "")
    engine_html = f'<span class="card-engine">{engine}</span>' if engine else ""

    # Figure type
    fig_type = classification.get("figure_type", "unknown") if classification else "unknown"
    confidence = classification.get("confidence") if classification else None
    type_html = fig_type
    if confidence is not None:
        type_html += f" · {confidence:.2f}"

    # Layer count
    n_layers = segmentation.get("n_layers", 0)
    layers_html = f"{n_layers} layers" if n_layers else ""

    # Alert for low quality
    alert_html = ""
    if score is not None and score < 0.7 and status == "segmented":
        alert_html = (
            '<div class="card-alert">'
            '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">'
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>'
            '</svg><span>建议 review</span></div>'
        )

    # Skip reason
    if status == "skipped":
        alert_html = f'<div class="card-alert" style="color:#86868b">{skip_reason or "Skipped"}</div>'

    return f"""
    <div class="{card_class}" onclick="openModal({idx})" data-status="{status}">
        <div class="card-img">
            {img_html}
            <span class="card-badge {badge_class}">{status.upper()}</span>
            {score_html}
        </div>
        <div class="card-body">
            <div class="card-title">
                <div style="min-width:0">
                    <h3 title="{fig_id}">{fig_id}</h3>
                    <p>{type_html}</p>
                </div>
                {engine_html}
            </div>
            <div class="card-meta">
                <span>{layers_html}</span>
            </div>
            {alert_html}
        </div>
    </div>
    """


def _generate_modal(entry: dict[str, Any], idx: int) -> str:
    """Generate HTML for one figure detail modal."""
    fig_id = entry["figure_id"]
    source = entry["source_path"]
    status = entry.get("status", "pending")
    classification = entry.get("classification") or {}
    panels = entry.get("panels") or {}
    segmentation = entry.get("segmentation") or {}
    export = entry.get("export") or {}

    fig_type = classification.get("figure_type", "—") if classification else "—"
    confidence = classification.get("confidence")
    confidence_str = f"{confidence:.2f}" if confidence is not None else "—"

    score = segmentation.get("quality_score")
    score_str = f"{score:.2f}" if score is not None else "—"

    engine = segmentation.get("engine", "—")
    n_layers = segmentation.get("n_layers", 0)
    n_layers_str = str(n_layers) if n_layers else "—"

    target_panel = panels.get("target_panel_id", 0) if panels else 0
    panel_str = f"#{target_panel}"

    # Images
    orig_src = _img_src(source)
    overlay_path = segmentation.get("overlay_path")
    overlay_src = _img_src(overlay_path) if overlay_path else ""

    orig_img = (
        f'<img src="{orig_src}" alt="original">'
        if orig_src
        else '<span class="no-img">原始图不可用</span>'
    )
    overlay_img = (
        f'<img src="{overlay_src}" alt="overlay">'
        if overlay_src
        else '<span class="no-img">Overlay 不可用</span>'
    )

    # Export files
    export_html = ""
    if export:
        tomo = export.get("tomo_xyz")
        parfile = export.get("parfile_snippet")
        if tomo or parfile:
            files = []
            if tomo:
                files.append(f'<div class="export-file"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>{Path(tomo).name}</div>')
            if parfile:
                files.append(f'<div class="export-file"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>{Path(parfile).name}</div>')
            export_html = f"""
            <div class="export-box">
                <p>导出文件</p>
                {''.join(files)}
            </div>
            """

    return f"""
    <div id="modal-{idx}" class="modal" onclick="closeModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div class="modal-header">
                <div>
                    <h2>{fig_id}</h2>
                    <p>{fig_type} · {engine} · {n_layers_str} layers</p>
                </div>
                <button class="modal-close" onclick="closeModal()">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
            </div>
            <div class="modal-body">
                <div class="compare">
                    <div class="compare-box">
                        <p>原始图</p>
                        <div class="img-wrap">{orig_img}</div>
                    </div>
                    <div class="compare-box">
                        <p>分割 overlay</p>
                        <div class="img-wrap">{overlay_img}</div>
                    </div>
                </div>
                <div class="meta-grid">
                    <div class="meta-item"><p>Figure 类型</p><p>{fig_type}</p></div>
                    <div class="meta-item"><p>置信度</p><p>{confidence_str}</p></div>
                    <div class="meta-item"><p>质量评分</p><p>{score_str}</p></div>
                    <div class="meta-item"><p>分割引擎</p><p>{engine}</p></div>
                    <div class="meta-item"><p>层数</p><p>{n_layers_str}</p></div>
                    <div class="meta-item"><p>目标 Panel</p><p>{panel_str}</p></div>
                </div>
                {export_html}
                <div class="chatbox">
                    <div class="chatbox-header">反馈（实时发送到 CLI）</div>
                    <div class="chatbox-body" id="chat-body-{idx}">
                        <div class="chatmsg system">
                            <div class="bubble">输入自然语言修改意见，将实时发送到 Claude Code CLI 中。确保 feedback_bridge 已启动。</div>
                        </div>
                    </div>
                    <div class="chatbox-input">
                        <input type="text" id="chat-input-{idx}" placeholder="例如：去掉右上角颜色条，底层分两层..."
                            onkeydown="if(event.key==='Enter')sendFeedback({idx},'{fig_id}')">
                        <button onclick="sendFeedback({idx},'{fig_id}')">发送</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(session_path: str, output_path: str | None = None) -> Path:
    """Generate HTML report from session state file."""
    session = load_session(session_path)

    # Determine output path
    if output_path is None:
        session_name = Path(session_path).stem
        output_path = f"runs/reports/{session_name}.html"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build stats
    total = len(session.workset)
    exported = sum(1 for e in session.workset if e.status == FigureStatus.EXPORTED)
    segmented = sum(1 for e in session.workset if e.status == FigureStatus.SEGMENTED)
    skipped = sum(1 for e in session.workset if e.status == FigureStatus.SKIPPED)
    scores = [
        e.segmentation.quality_score
        for e in session.workset
        if e.segmentation and e.segmentation.quality_score is not None
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Build cards
    cards_html = "\n".join(
        _generate_card(e.model_dump(), i) for i, e in enumerate(session.workset)
    )

    # Build modals
    modals_html = "\n".join(
        _generate_modal(e.model_dump(), i) for i, e in enumerate(session.workset)
    )

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geoseg — {session.session_id}</title>
<style>{_css()}</style>
</head>
<body>
<header class="header">
    <div class="header-inner">
        <div class="logo">
            <div class="logo-icon">gs</div>
            <div class="logo-text">
                <h1>geoseg</h1>
                <p>Session Dashboard</p>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:13px;color:#86868b;background:#f5f5f7;padding:5px 12px;border-radius:20px">{session.session_id}</span>
        </div>
    </div>
</header>

<div class="stats">
    <div class="stat-card">
        <div class="stat-label">总图数</div>
        <div class="stat-value">{total}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">已导出</div>
        <div class="stat-value exported">{exported}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">待 review</div>
        <div class="stat-value segmented">{segmented}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">已跳过</div>
        <div class="stat-value skipped">{skipped}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">平均质量</div>
        <div class="stat-value avg">{avg_score:.2f}</div>
    </div>
</div>

<div class="filters">
    <button class="filter-btn active" onclick="filter('all',this)">全部</button>
    <button class="filter-btn" onclick="filter('exported',this)">已导出</button>
    <button class="filter-btn" onclick="filter('segmented',this)">待 review</button>
    <button class="filter-btn" onclick="filter('skipped',this)">已跳过</button>
</div>

<div class="grid" id="card-grid">
{cards_html}
</div>

{modals_html}

<script>
function openModal(idx) {{
    document.getElementById('modal-'+idx).classList.add('show');
    document.body.style.overflow='hidden';
}}
function closeModal(e) {{
    if(!e||e.target===e.currentTarget){{
        document.querySelectorAll('.modal').forEach(m=>m.classList.remove('show'));
        document.body.style.overflow='';
    }}
}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeModal()}});

function filter(status,btn){{
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.card').forEach(card=>{{
        if(status==='all'||card.dataset.status===status){{
            card.style.display='';
        }}else{{
            card.style.display='none';
        }}
    }});
}}

function sendFeedback(idx, figId) {{
    const input = document.getElementById('chat-input-'+idx);
    const body = document.getElementById('chat-body-'+idx);
    const text = input.value.trim();
    if(!text) return;

    // User message
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.innerHTML = '<span class="bubble">' + escapeHtml(text) + '</span>';
    body.appendChild(userDiv);

    input.value = '';
    body.scrollTop = body.scrollHeight;

    // Send to feedback bridge
    fetch('http://127.0.0.1:8765/feedback', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{figure_id: figId, text: text}})
    }})
    .then(r => r.json())
    .then(data => {{
        const sysDiv = document.createElement('div');
        sysDiv.className = 'chat-msg system';
        if (data.status === 'ok') {{
            sysDiv.innerHTML = '<span class="bubble" style="color:#10b981">✓ 已发送至 CLI</span>';
        }} else {{
            sysDiv.innerHTML = '<span class="bubble" style="color:#ef4444">✗ 发送失败: ' + escapeHtml(data.error || 'unknown') + '</span>';
        }}
        body.appendChild(sysDiv);
        body.scrollTop = body.scrollHeight;
    }})
    .catch(err => {{
        const sysDiv = document.createElement('div');
        sysDiv.className = 'chat-msg system';
        sysDiv.innerHTML = '<span class="bubble" style="color:#ef4444">✗ 无法连接 bridge<br><small>请确保 feedback_bridge 已启动</small></span>';
        body.appendChild(sysDiv);
        body.scrollTop = body.scrollHeight;
    }});
}}

function escapeHtml(t) {{
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}}
</script>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from session state")
    parser.add_argument("session", help="Path to session_state.json")
    parser.add_argument("--output", "-o", help="Output HTML path (default: runs/reports/{session_name}.html)")
    args = parser.parse_args()

    out = generate(args.session, args.output)
    print(f"Report generated: {out}")


if __name__ == "__main__":
    main()
