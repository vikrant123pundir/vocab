"""District order matrix.

Usage:  python district_matrix.py "path/to/tondai_matrix.json"

Reads the data file, pulls every section in the number range from the index files,
refuses to run if any district x section cell is missing, then scores every allowed
order of the districts on sound, flow and data, and writes <data>.html and <data>.csv
next to the data file.

Needs: numpy, shapely, svgelements  (pip install numpy shapely svgelements)
"""
import csv, html, itertools, json, os, re, sys

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from svgelements import SVG, Path


# ---------- sections from the index files ----------
def load_sections(index_dir, lo, hi):
    found = []
    for f in sorted(os.listdir(index_dir)):
        if not f.endswith('_index.md'):
            continue
        for line in open(os.path.join(index_dir, f), encoding='utf-8'):
            m = re.match(r'\s*- ([A-Z_0-9]+) (\d+)([A-Z]*) > (.*?)\.?:\s*$', line)
            if m and lo <= int(m.group(2)) <= hi:
                found.append({'key': f"{m.group(1)} {m.group(2)}{m.group(3)}", 'subject': m.group(1),
                              'num': int(m.group(2)), 'code': m.group(2) + m.group(3), 'title': m.group(4)})
    order = {'COI': 0, 'BNS': 1, 'BNSS': 2, 'BSA': 3, 'LIM': 4, 'CPC': 5}
    found.sort(key=lambda s: (s['num'], s['code'] != str(s['num']), order.get(s['subject'], 9), s['code']))
    return found


# ---------- validation: no empty cell allowed ----------
def validate(data, sections):
    errors = []
    districts, facts, ratings = data['districts'], data['facts'], data['ratings']
    keys = [s['key'] for s in sections]
    for k in keys:
        if k not in ratings:
            errors.append(f"missing row for section {k}")
    for k in ratings:
        if k not in keys:
            errors.append(f"row {k} is not a section in the index range")
    for k, row in ratings.items():
        sc = row.get('scores', [])
        if len(sc) != len(districts):
            errors.append(f"{k}: {len(sc)} scores, need {len(districts)}")
            continue
        why = row.get('why', {})
        for d, v in zip(districts, sc):
            if v not in (0, 1, 2):
                errors.append(f"{k} / {d}: score {v} is not 0, 1 or 2")
            if v and d not in why:
                errors.append(f"{k} / {d}: score {v} without a fact")
            if not v and d in why:
                errors.append(f"{k} / {d}: fact given but score is 0")
        for d, w in why.items():
            fid = w.split('|')[0].strip()
            if d not in districts:
                errors.append(f"{k}: unknown district {d}")
            elif fid not in facts:
                errors.append(f"{k} / {d}: unknown fact id {fid}")
            elif facts[fid]['d'] != d:
                errors.append(f"{k} / {d}: fact {fid} belongs to {facts[fid]['d']}")
    for o, seq in data.get('compare_orders', {}).items():
        if sorted(seq) != sorted(districts):
            errors.append(f"compare order '{o}' does not use every district exactly once")
    return errors


# ---------- map gaps from the svg ----------
def district_gaps(svg_path, districts):
    shapes = {}
    for e in SVG.parse(svg_path).elements():
        if isinstance(e, Path) and e.id in districts:
            rings = []
            for sp in e.as_subpaths():
                p = Path(sp)
                n = max(int(p.length() / 0.5), 20)
                pts = [(p.point(i / n).x, p.point(i / n).y) for i in range(n)]
                if len(pts) >= 3:
                    rings.append(Polygon(pts).buffer(0))
            shapes[e.id] = unary_union(rings)
    missing = [d for d in districts if d not in shapes]
    if missing:
        sys.exit(f"svg has no path for: {', '.join(missing)}")
    n = len(districts)
    gap = np.zeros((n, n))
    for i, a in enumerate(districts):
        for j, b in enumerate(districts):
            if i < j:
                A, B = shapes[a], shapes[b]
                touch = A.buffer(0.6).intersection(B.buffer(0.6)).area > 0.5
                gap[i, j] = gap[j, i] = 0.0 if touch else A.distance(B)
    return gap


def main(json_path):
    base = os.path.dirname(os.path.abspath(json_path))
    data = json.load(open(json_path, encoding='utf-8'))
    lo, hi = data['range']
    districts = data['districts']
    n = len(districts)
    idx = {d: i for i, d in enumerate(districts)}

    sections = load_sections(os.path.join(base, data['index_dir']), lo, hi)
    errors = validate(data, sections)
    if errors:
        print("STOPPED: the matrix is incomplete or inconsistent:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    numbers = list(range(lo, hi + 1))
    if len(numbers) != n:
        sys.exit(f"{len(numbers)} numbers but {n} districts")

    # data fit: district x number
    fit = np.zeros((n, len(numbers)))
    for s in sections:
        sc = data['ratings'][s['key']]['scores']
        for i in range(n):
            fit[i, s['num'] - lo] += sc[i]

    # sound: from -> to
    sound = np.zeros((n, n))
    sound_note = {}
    for a, b, v, note in data['sound'].get('both', []):
        sound[idx[a], idx[b]] = sound[idx[b], idx[a]] = v
        sound_note[(a, b)] = sound_note[(b, a)] = note
    for a, b, v, note in data['sound'].get('forward', []):
        sound[idx[a], idx[b]] = v
        sound_note[(a, b)] = note

    # flow
    fr = data['flow_rule']
    gap = district_gaps(os.path.join(base, data['svg']), districts)
    flow = np.where(gap == 0, fr['touch'], np.where(gap <= fr['short_gap_max'], fr['short_gap'], fr['long_gap']))
    np.fill_diagonal(flow, 0)

    # every allowed order
    fixed = {int(k) - lo: [idx[d] for d in v] for k, v in data.get('fixed', {}).items()}
    pos_fixed = sorted(fixed)
    blocks = []
    for combo in itertools.product(*[fixed[p] for p in pos_fixed]):
        if len(set(combo)) < len(combo):
            continue
        rest = [i for i in range(n) if i not in combo]
        free_pos = [p for p in range(n) if p not in pos_fixed]
        perms = np.array(list(itertools.permutations(rest)), dtype=np.int8)
        P = np.zeros((len(perms), n), dtype=np.int8)
        P[:, free_pos] = perms
        for p, d in zip(pos_fixed, combo):
            P[:, p] = d
        blocks.append(P)
    P = np.vstack(blocks)
    S = sound[P[:, :-1], P[:, 1:]].sum(1)
    F = flow[P[:, :-1], P[:, 1:]].sum(1)
    D = fit[P, np.arange(n)].sum(1)
    w = data['weights']
    s_max, f_max, d_max = 2 * (n - 1), fr['touch'] * (n - 1), max(D.max(), 1)
    wsum = w['sound'] + w['flow'] + w['data']
    T = 100 * (w['sound'] * S / s_max + w['flow'] * F / f_max + w['data'] * D / d_max) / wsum
    top = np.argsort(-T, kind='stable')[:10]

    def score_order(seq):
        o = np.array([idx[d] for d in seq])
        s, f, dd = sound[o[:-1], o[1:]].sum(), flow[o[:-1], o[1:]].sum(), fit[o, np.arange(n)].sum()
        t = 100 * (w['sound'] * s / s_max + w['flow'] * f / f_max + w['data'] * dd / d_max) / wsum
        return s, f, dd, t

    stem = os.path.splitext(json_path)[0]

    # ---------- CSV ----------
    with open(stem + '.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['section', 'title'] + districts)
        for s in sections:
            row = data['ratings'][s['key']]
            wr.writerow([s['key'], s['title']] + [
                f"{v} {row['why'][d]}" if v else '0' for d, v in zip(districts, row['scores'])])

    # ---------- HTML ----------
    E = html.escape
    facts = data['facts']
    col = {0: '#ffffff', 1: '#ffe9a8', 2: '#7fd18b'}

    def joint_text(seq):
        parts = []
        for a, b in zip(seq, seq[1:]):
            i, j = idx[a], idx[b]
            g = 'touch' if gap[i, j] == 0 else f'gap {gap[i, j]:.1f}'
            parts.append(f"{a} → {b}: sound {int(sound[i, j])} ({sound_note.get((a, b), '-')}), {g}")
        return '\n'.join(parts)

    def order_row(label, seq, s, f, dd, t, hl=False):
        cells = ''.join(
            f'<td style="background:{col[min(int(fit[idx[d], k]), 2)] if fit[idx[d], k] else "#fff"}" '
            f'title="fit {int(fit[idx[d], k])} at {lo + k}">{E(d)}</td>' for k, d in enumerate(seq))
        return (f'<tr{" class=hl" if hl else ""}><td>{E(label)}</td><td title="{E(joint_text(seq))}">'
                f'<b>{t:.1f}</b></td><td>{int(s)}/{s_max}</td><td>{int(f)}/{f_max}</td>'
                f'<td>{int(dd)}/{int(d_max)}</td>{cells}</tr>')

    out = [f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>{E(data["title"])} matrix</title>
<style>
body{{font:13px system-ui,sans-serif;margin:16px;color:#222}} h2{{margin:22px 0 6px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:3px 5px;text-align:center}}
th{{background:#f3f3f3}} td.l{{text-align:left;white-space:nowrap}} tr.hl td{{outline:2px solid #3b6fd8}}
.f{{font-family:ui-monospace,monospace;background:#f6f6f6;padding:6px 8px;display:inline-block}}
.om{{background:repeating-linear-gradient(45deg,#eee,#eee 4px,#fff 4px,#fff 8px)!important}}
.leg span{{display:inline-block;width:14px;height:14px;vertical-align:middle;border:1px solid #aaa;margin:0 3px 0 10px}}
</style></head><body>
<h1>{E(data["title"])}</h1>
<div class="f">Total = 100 × ( {w["sound"]}·Sound/{s_max} + {w["flow"]}·Flow/{f_max} + {w["data"]}·Data/{int(d_max)} ) / {wsum}</div>
<div class="leg">fit: <span style="background:{col[2]}"></span>2 strong <span style="background:{col[1]}"></span>1 loose
<span style="background:#fff"></span>0 none &nbsp; · hover any cell for the fact, any score for the joints</div>''']

    # top orders
    out.append(f'<h2>Top 10 orders (out of {len(P):,} allowed)</h2><table><tr><th>#</th><th>Total</th>'
               f'<th>Sound</th><th>Flow</th><th>Data</th>' + ''.join(f'<th>{x}</th>' for x in numbers) + '</tr>')
    for r, k in enumerate(top, 1):
        seq = [districts[i] for i in P[k]]
        out.append(order_row(str(r), seq, S[k], F[k], D[k], T[k]))
    for label, seq in data.get('compare_orders', {}).items():
        out.append(order_row(label, seq, *score_order(seq), hl=True))
    out.append('</table>')

    # fit by number
    out.append('<h2>Data fit by number (sum over every section at that number)</h2><table><tr><th>District</th>'
               + ''.join(f'<th>{x}</th>' for x in numbers) + '<th>Best number</th></tr>')
    colmax = fit.max(0)
    for i, d in enumerate(districts):
        cells = ''
        for k in range(len(numbers)):
            v = int(fit[i, k])
            bold = v and v == colmax[k]
            cells += (f'<td style="background:{col[min(v, 2)] if v else "#fff"}">'
                      f'{"<b>" if bold else ""}{v or ""}{"</b>" if bold else ""}</td>')
        best = ', '.join(str(lo + k) for k in np.flatnonzero(fit[i] == fit[i].max())) if fit[i].max() else '-'
        out.append(f'<tr><td class="l">{E(d)}</td>{cells}<td>{best}</td></tr>')
    out.append('</table><p>Bold = best district for that number.</p>')

    # full matrix
    out.append('<h2>Full matrix: every section × every district</h2><table><tr><th>District</th>')
    for s in sections:
        om = ' class="om"' if data['ratings'][s['key']].get('omitted') else ''
        out.append(f'<th{om} title="{E(s["key"] + ": " + s["title"])}">{E(s["subject"])}<br>{E(s["code"])}</th>')
    out.append('</tr>')
    for i, d in enumerate(districts):
        out.append(f'<tr><td class="l">{E(d)}</td>')
        for s in sections:
            row = data['ratings'][s['key']]
            v = row['scores'][i]
            if v:
                fid, _, link = row['why'][d].partition('|')
                fid = fid.strip()
                tip = f"{s['key']}: {s['title']}\n{fid}: {facts[fid]['t']}\nLink: {link.strip()}"
                out.append(f'<td style="background:{col[v]}" title="{E(tip)}">{v}<br><small>{E(fid)}</small></td>')
            else:
                out.append(f'<td{" class=om" if row.get("omitted") else ""} title="{E(s["key"] + ": " + s["title"])}"></td>')
        out.append('</tr>')
    out.append('</table>')

    # sections list
    out.append('<h2>Sections in this block</h2><table><tr><th>Section</th><th>Title</th></tr>')
    for s in sections:
        om = ' (omitted)' if data['ratings'][s['key']].get('omitted') else ''
        out.append(f'<tr><td>{E(s["key"])}</td><td class="l">{E(s["title"])}{om}</td></tr>')
    out.append('</table>')

    # facts
    out.append('<h2>Facts</h2><table><tr><th>ID</th><th>District</th><th>Fact</th><th>Source</th></tr>')
    for fid, f in facts.items():
        src = f'<a href="{E(f["src"])}" target="_blank">link</a>' if f['src'] else 'not checked'
        out.append(f'<tr><td>{E(fid)}</td><td class="l">{E(f["d"])}</td><td class="l">{E(f["t"])}</td><td>{src}</td></tr>')
    out.append('</table>')

    # sound and flow
    out.append('<h2>Sound (row → column) and map gap</h2><table><tr><th></th>'
               + ''.join(f'<th title="{E(d)}">{E(d[:6])}</th>' for d in districts) + '</tr>')
    for i, a in enumerate(districts):
        out.append(f'<tr><td class="l">{E(a)}</td>')
        for j, b in enumerate(districts):
            if i == j:
                out.append('<td style="background:#ddd"></td>')
                continue
            g = 'touch' if gap[i, j] == 0 else f'{gap[i, j]:.1f}'
            sv = int(sound[i, j])
            out.append(f'<td style="background:{col[sv] if sv else "#fff"}" '
                       f'title="{E(a)} → {E(b)}: {E(sound_note.get((a, b), "no sound link"))}">{sv or ""}<br><small>{g}</small></td>')
        out.append('</tr>')
    out.append('</table><p>Top number = sound score, small number = map gap (touch = shared border).</p></body></html>')

    open(stem + '.html', 'w', encoding='utf-8').write('\n'.join(out))
    print(f"OK: {len(sections)} sections x {n} districts checked, {len(P):,} orders scored")
    print(f"wrote {stem}.html and {stem}.csv")
    for r, k in enumerate(top[:5], 1):
        print(f"{r}. {T[k]:.1f}  S{int(S[k])} F{int(F[k])} D{int(D[k])}  " + ' > '.join(districts[i] for i in P[k]))
    for label, seq in data.get('compare_orders', {}).items():
        s, f, dd, t = score_order(seq)
        print(f"[{label}] {t:.1f}  S{int(s)} F{int(f)} D{int(dd)}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
