#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

ROOT = Path(__file__).resolve().parent
ENV_SH = str(ROOT / "../../env.sh")
RESULT_ENV = "VENTUS_RESULT_FILE"
DEFAULT_RESULT_NAME = "result.hex"  # 程序若未设置环境变量时的默认文件名（不建议）

FLOAT_HEX_RE = re.compile(r"^[hH]([0-9a-fA-F]+)$")

def read_cases(cases_csv: Path) -> List[dict]:
    rows = []
    with cases_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        required = {"dir", "exe"}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            raise ValueError(f"cases.csv 缺少列: {missing}")
        for row in rdr:
            d = row.get("dir", "").strip()
            e = row.get("exe", "").strip()
            r = (row.get("run_cmd") or "").strip()
            if not d or not e:
                print(f"[WARN] 跳过无效行: {row}", file=sys.stderr)
                continue
            rows.append({"dir": d, "exe": e, "run_cmd": r})
    return rows

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _compose_cmd(cmd: str, use_env: bool=False, backend: Optional[str]=None, extra_env: Optional[dict]=None) -> str:
    prefix_parts = []
    if use_env:
        prefix_parts.append(f"source {shlex.quote(ENV_SH)}")
        if backend:
            prefix_parts.append(f"export VENTUS_BACKEND={shlex.quote(backend)}")
    if extra_env:
        for k, v in extra_env.items():
            prefix_parts.append(f"export {k}={shlex.quote(v)}")
    prefix = " && ".join(prefix_parts)
    full_cmd = cmd if not prefix else f"{prefix} && {cmd}"
    return f"bash -lc {shlex.quote(full_cmd)}"

def run_bash(cmd: str, cwd: Path, use_env: bool=False, backend: Optional[str]=None,
             timeout_s: Optional[float]=None, extra_env: Optional[dict]=None) -> Tuple[int, str, float, bool]:
    """
    在 cwd 下执行 cmd。
    use_env=True 时，会 `source ENV_SH` 并可设置 VENTUS_BACKEND。
    可通过 extra_env 额外注入环境变量（如 VENTUS_RESULT_FILE）。
    返回：returncode, stdout+stderr, elapsed_seconds, timed_out
    """
    full = _compose_cmd(cmd, use_env=use_env, backend=backend, extra_env=extra_env)
    start = time.perf_counter()
    timed_out = False
    proc = subprocess.Popen(
        full, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass
        try: out, _ = proc.communicate(timeout=5)
        except Exception: out = ""
        rc = 124
    elapsed = time.perf_counter() - start
    return rc, out, elapsed, timed_out

def make_build(case_dir: Path, jobs: Optional[int]) -> Tuple[bool, str, float]:
    cmd = f"make -j{jobs}" if jobs and jobs > 1 else "make"
    rc, out, el, _ = run_bash(cmd, cwd=case_dir, use_env=True, timeout_s=None)
    return rc == 0, out, el

def pick_run_cmd(case_dir: Path, exe_name: str, run_cmd_field: str) -> str:
    if run_cmd_field:  # 1) 指定 run_cmd
        return run_cmd_field
    for cand in ["run", "run.sh"]:  # 2) 存在 run/ run.sh
        p = case_dir / cand
        if p.exists() and os.access(str(p), os.X_OK):
            return f"./{cand}"
    return f"./{exe_name}"  # 3) 默认执行可执行文件

def parse_result_file(path: Path) -> List[float]:
    """
    读取一行结果。支持：
      - hex:  hxxxxxxxx 形式（与 ventus_result_io.h 写出一致）
      - float: 十进制浮点
    """
    if not path.exists():
        return []
    line = path.read_text().strip()
    if not line:
        return []
    vals = []
    for tok in line.split():
        m = FLOAT_HEX_RE.match(tok)
        if m:
            u = int(m.group(1), 16)
            import struct
            vals.append(struct.unpack('!f', struct.pack('!I', u))[0])
        else:
            try:
                vals.append(float(tok))
            except ValueError:
                # 忽略非数值 token
                pass
    return vals

def allclose(a: List[float], b: List[float], rtol: float, atol: float) -> Tuple[bool, dict]:
    n = min(len(a), len(b))
    if n == 0:
        return False, {"reason": "无可对比数值", "n": 0}
    max_abs_err = 0.0
    max_rel_err = 0.0
    idx = -1
    ok = True
    for i in range(n):
        ai, bi = a[i], b[i]
        abs_err = abs(ai - bi)
        if abs_err > (atol + rtol * abs(ai)):
            ok = False
        if abs_err > max_abs_err:
            max_abs_err = abs_err
            max_rel_err = abs_err / (abs(ai) + 1e-12)
            idx = i
    if len(a) != len(b):
        ok = False
    return ok, {
        "compared_elems": n,
        "len_a": len(a), "len_b": len(b),
        "max_abs_err": max_abs_err, "max_rel_err": max_rel_err,
        "worst_index": idx
    }

def main():
    import argparse
    ap = argparse.ArgumentParser(description="批量测试（只读程序写出的最终结果文件）")
    ap.add_argument("--cases", required=True, help="测例清单 CSV（dir, exe, 可选 run_cmd）")
    ap.add_argument("--backends", default="spike,rtlsim,cyclesim", help="VENTUS 后端，逗号分隔")
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--ref", help="引用基准结果目录，跳过 NVIDIA 基线运行")
    ap.add_argument("--result_name", default="result.hex",
                    help="若程序未使用环境变量，默认写入的文件名（相对 case 目录），仅兜底")
    args = ap.parse_args()

    cases = read_cases((ROOT / args.cases).resolve())
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]

    results_dir = ROOT / "results"
    ensure_dir(results_dir)

    # 1) 编译
    case_infos = []
    print("\n=== 阶段 1：编译所有测例 ===")
    for row in cases:
        case_dir = (ROOT / row["dir"]).resolve()
        exe_name = row["exe"]
        run_cmd_field = row.get("run_cmd", "")
        case_name = row["dir"].rstrip("/")

        ok, make_out, make_time = make_build(case_dir, args.jobs if args.jobs > 1 else None)
        if not ok:
            print(f"[ERROR]: Compile failed: {case_name}")
        case_infos.append({
            "case_name": case_name,
            "case_dir": case_dir,
            "exe": exe_name,
            "run_cmd": pick_run_cmd(case_dir, exe_name, run_cmd_field),
            "build_ok": ok,
            "build_time": make_time
        })
        # 记录一份编译日志（可选）
        (results_dir / case_name).mkdir(parents=True, exist_ok=True)
        (results_dir / case_name / "make.log").write_text(make_out)

    summary_rows = []
    summary_md = []
    summary_md += [
        "# 测试报告",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 容差：rtol={args.rtol}, atol={args.atol}",
        f"- 超时：{args.timeout} 秒",
        ""
    ]

    # 统一的结果文件写入路径：results/<case>/<env>/final.hex
    def result_path_for(case: str, env: str) -> Path:
        p = results_dir / case / env / "final.hex"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) NVIDIA 基线
    if args.ref:
        print(f"\n=== 阶段 2：NVIDIA 基线 (使用 --ref={args.ref}) 中的预定义结果===")
        ref_path = (ROOT / args.ref).resolve()
        if ref_path.exists():
            # 复制 ref_path 下所有内容到 results_dir
            cmd = f"cp -r {shlex.quote(str(ref_path))}/* {shlex.quote(str(results_dir))}/"
            print(f"Copying ref: {cmd}")
            run_bash(cmd, cwd=ROOT)
        else:
            print(f"[WARN] Ref path {ref_path} not found!")

        for info in case_infos:
            case = info["case_name"]
            out_file = result_path_for(case, "nvidia")
            vals = parse_result_file(out_file) if out_file.exists() else []
            summary_rows.append({
                "case": case, "env": "nvidia", "status": "skipped(ref)",
                "time_s": "0.000000", "match": "baseline" if vals else "missing",
                "max_abs_err": "", "max_rel_err": "", "details": f"from ref, {len(vals)} values"
            })
    else:
        print("\n=== 阶段 2：NVIDIA 基线 ===")
        for info in case_infos:
            case, case_dir, run_cmd = info["case_name"], info["case_dir"], info["run_cmd"]
            if not info["build_ok"]:
                summary_rows.append({
                    "case": case, "env": "build", "status": "build_failed",
                    "time_s": f"{info['build_time']:.6f}", "match": "", "max_abs_err": "", "max_rel_err": "",
                    "details": "make failed"
                })
                continue

            out_file = result_path_for(case, "nvidia")
            # 运行时用环境变量告诉程序把“最终结果”写到 out_file
            extra_env = {RESULT_ENV: str(out_file)}
            print(f"[NVIDIA] {case} -> {out_file}")
            rc, _, tsec, to = run_bash(run_cmd, cwd=case_dir, use_env=False,
                                       timeout_s=args.timeout, extra_env=extra_env)

            vals = parse_result_file(out_file) if out_file.exists() else []
            status = "timeout" if to else ("ok" if (rc == 0 and vals) else "run_failed")
            det = f"{len(vals)} values" if vals else "result file missing or empty"

            summary_rows.append({
                "case": case, "env": "nvidia", "status": status,
                "time_s": f"{tsec:.6f}", "match": "baseline" if status == "ok" else "baseline_failed",
                "max_abs_err": "", "max_rel_err": "", "details": det
            })

    # 3) VENTUS 各后端
    print("\n=== 阶段 3：VENTUS 后端 ===")
    for be in backends:
        print(f"\n--- VENTUS_BACKEND = {be} ---")
        for info in case_infos:
            case, case_dir, run_cmd = info["case_name"], info["case_dir"], info["run_cmd"]
            if not info["build_ok"]:
                summary_rows.append({
                    "case": case, "env": be, "status": "compile_failed",
                    "time_s": "-", "match": "fail",
                    "max_abs_err": "-", "max_rel_err": "-", "details": "Compile Failed" 
                })
                continue

            baseline_file = result_path_for(case, "nvidia")
            baseline_vals = parse_result_file(baseline_file) if baseline_file.exists() else []

            out_file = result_path_for(case, be)
            extra_env = {RESULT_ENV: str(out_file)}
            print(f"[{case}] {be} -> {out_file}")
            rc, _, tsec, to = run_bash(run_cmd, cwd=case_dir, use_env=True, backend=be,
                                       timeout_s=args.timeout, extra_env=extra_env)

            vals = parse_result_file(out_file) if out_file.exists() else []
            status = "timeout" if to else ("ok" if (rc == 0 and vals) else "run_failed")

            if baseline_vals and vals and status == "ok":
                okmatch, info_cmp = allclose(baseline_vals, vals, args.rtol, args.atol)
                match_str = "pass" if okmatch else "fail"
                max_abs = f"{info_cmp.get('max_abs_err', '')}"
                max_rel = f"{info_cmp.get('max_rel_err', '')}"
                details = json.dumps(info_cmp, ensure_ascii=False)
            elif not baseline_vals:
                match_str, max_abs, max_rel, details = "no_baseline", "", "", "baseline missing or failed"
            else:
                match_str, max_abs, max_rel, details = "fail", "", "", "run failed/timeout or result missing"

            summary_rows.append({
                "case": case, "env": be, "status": status,
                "time_s": f"{tsec:.6f}", "match": match_str,
                "max_abs_err": max_abs, "max_rel_err": max_rel, "details": details
            })

    # 4) 汇总输出
    for info in case_infos:
        case = info["case_name"]
        summary_md += [
            f"## {case}", "",
            "| 环境 | 状态 | 用时(s) | 匹配 | max_abs_err | 备注 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in [x for x in summary_rows if x["case"] == case]:
            summary_md.append(
                f"| {r['env']} | {r['status']} | {r['time_s']} | {r['match']} | {r['max_abs_err']} | {r['details']} |"
            )
        summary_md.append("")

    (results_dir / "summary.csv").write_text(
        "\n".join(
            [",".join(["case","env","status","time_s","match","max_abs_err","max_rel_err","details"])]
            + [",".join([
                r["case"], r["env"], r["status"], r["time_s"], r["match"],
                str(r["max_abs_err"]), str(r["max_rel_err"]), '"' + str(r["details"]).replace('"','""') + '"'
            ]) for r in summary_rows]
        ),
        encoding="utf-8"
    )
    (results_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print("\n=== 完成 ===")
    print(f"- 汇总 CSV: {results_dir / 'summary.csv'}")
    print(f"- 汇总 Markdown: {results_dir / 'summary.md'}")
    print(f"- 结果文件：results/<case>/<env>/final.hex (由程序直接生成)")

    failed_cases = set()
    for r in summary_rows:
        if r["match"] not in ["pass", "baseline"]:
            failed_cases.add(r["case"])
    if failed_cases:
        print(f"Failed cases: {failed_cases}")
        sys.exit(1)

if __name__ == "__main__":
    main()
